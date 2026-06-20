//! cTrader order RPCs: open a market position and close a position.
//!
//! **Real-money surface.** A market order produces a *sequence* of
//! `ProtoOAExecutionEvent`s (typically `ORDER_ACCEPTED` then `ORDER_FILLED`),
//! and cTrader echoes the request `clientMsgId` only on the first; later events
//! arrive unsolicited. So these functions read both the correlated reply (via
//! [`Connection::send_request`]) and subsequent events from the connection's
//! events channel until a terminal `ORDER_FILLED` (→ [`Fill`]) or
//! `ORDER_REJECTED` (→ error).
//!
//! ## Volume units (real-money critical, verified against cTrader demo)
//!
//! The cTrader API `volume` field is in **centi-units**: `volume = base
//! currency units × 100`. For a 100 000-unit standard lot that means
//! `volume = lots × 10_000_000`, and the USDJPY minimum (0.01 lot = 1000 base
//! units) is `volume = 100_000`. (Demo rejected `volume=1000` as
//! `TRADING_BAD_VOLUME: 10.00 < minimum 1000.00`, i.e. it read 1000 as 10.00,
//! confirming the ×100 factor. The `1000 = 0.01 lot` note in
//! `markets/traders/doublebottom.py` is misleading vs. the raw API.) Callers
//! pass the already-converted integer `volume`; use [`lots_to_volume`].

use std::time::{Duration, Instant};

use anyhow::{anyhow, Result};
use modelenv_proto::ctrader::{
    ProtoMessage, ProtoOaClosePositionReq, ProtoOaErrorRes, ProtoOaExecutionEvent,
    ProtoOaNewOrderReq, ProtoOaOrderErrorEvent,
};
use modelenv_proto::Fill;
use prost::Message;
use tokio::sync::mpsc;

use super::connection::Connection;
use super::wire::payload_type;

// cTrader ProtoOAOrderType / ProtoOATradeSide / ProtoOAExecutionType values.
const ORDER_TYPE_MARKET: i32 = 1;
const TRADE_SIDE_BUY: i32 = 1;
const TRADE_SIDE_SELL: i32 = 2;
const EXEC_ORDER_ACCEPTED: i32 = 2;
const EXEC_ORDER_FILLED: i32 = 3;
const EXEC_ORDER_REJECTED: i32 = 7;

/// Base-currency units in one standard FX lot (USDJPY and most FX). Symbol-
/// specific in general (cTrader exposes it per symbol); used here for the
/// lots→volume conversion of the FX majors this gateway trades.
pub const STANDARD_LOT_BASE_UNITS: f64 = 100_000.0;
/// cTrader API volume is in centi-units: `volume = base_units × 100`.
pub const CENTI_UNITS_PER_BASE_UNIT: f64 = 100.0;

/// Convert a lot size (e.g. 0.01) to the cTrader API integer `volume`
/// (`lots × 100_000 base units × 100`). 0.01 lot → 100_000 (the USDJPY min).
pub fn lots_to_volume(lots: f64) -> i64 {
    (lots * STANDARD_LOT_BASE_UNITS * CENTI_UNITS_PER_BASE_UNIT).round() as i64
}

/// Order direction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Side {
    Buy,
    Sell,
}

impl Side {
    fn trade_side(self) -> i32 {
        match self {
            Side::Buy => TRADE_SIDE_BUY,
            Side::Sell => TRADE_SIDE_SELL,
        }
    }
}

/// Outcome of a filled order/close: the resulting [`Fill`] plus the broker
/// `position_id` (needed to later close, or to track the open position).
#[derive(Clone, Debug)]
pub struct OrderResult {
    pub fill: Fill,
    pub position_id: i64,
}

/// Submit a MARKET order on `symbol_id` for `account_id` and wait for the fill.
/// `volume` is cTrader units (1000 = 0.01 lot). `client_order_id` is echoed for
/// broker-side idempotency. Returns the [`Fill`] + opened `position_id`.
pub async fn submit_market_order(
    conn: &Connection,
    events: &mut mpsc::UnboundedReceiver<ProtoMessage>,
    account_id: i64,
    symbol_id: i64,
    side: Side,
    volume: i64,
    client_order_id: &str,
    timeout: Duration,
) -> Result<OrderResult> {
    let req = ProtoOaNewOrderReq {
        payload_type: Some(payload_type::NEW_ORDER_REQ as i32),
        ctid_trader_account_id: account_id,
        symbol_id,
        order_type: ORDER_TYPE_MARKET,
        trade_side: side.trade_side(),
        volume,
        client_order_id: Some(client_order_id.to_string()),
        ..Default::default()
    };
    let first = conn
        .send_request(payload_type::NEW_ORDER_REQ, req.encode_to_vec(), timeout)
        .await?;
    await_fill(events, first, timeout, "submit_market_order").await
}

/// Close (fully) an open position by id. `volume` is the cTrader-unit volume to
/// close (use the position's opened volume for a full close).
pub async fn close_position(
    conn: &Connection,
    events: &mut mpsc::UnboundedReceiver<ProtoMessage>,
    account_id: i64,
    position_id: i64,
    volume: i64,
    timeout: Duration,
) -> Result<OrderResult> {
    let req = ProtoOaClosePositionReq {
        payload_type: Some(payload_type::CLOSE_POSITION_REQ as i32),
        ctid_trader_account_id: account_id,
        position_id,
        volume,
    };
    let first = conn
        .send_request(payload_type::CLOSE_POSITION_REQ, req.encode_to_vec(), timeout)
        .await?;
    await_fill(events, first, timeout, "close_position").await
}

/// Drive an order/close to a terminal outcome: inspect the correlated `first`
/// reply, then drain `events` until `ORDER_FILLED` (→ `OrderResult`) or
/// `ORDER_REJECTED`/error (→ Err), bounded by `timeout`.
async fn await_fill(
    events: &mut mpsc::UnboundedReceiver<ProtoMessage>,
    first: ProtoMessage,
    timeout: Duration,
    ctx: &str,
) -> Result<OrderResult> {
    match interpret(&first, ctx)? {
        Exec::Filled(res) => return Ok(res),
        Exec::Rejected(e) => return Err(anyhow!("cTrader {ctx} rejected: {e}")),
        Exec::Pending => {}
    }
    let deadline = Instant::now() + timeout;
    loop {
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            return Err(anyhow!("cTrader {ctx}: no fill before timeout"));
        }
        match tokio::time::timeout(remaining, events.recv()).await {
            Ok(Some(frame)) => match interpret(&frame, ctx)? {
                Exec::Filled(res) => return Ok(res),
                Exec::Rejected(e) => return Err(anyhow!("cTrader {ctx} rejected: {e}")),
                Exec::Pending => continue,
            },
            Ok(None) => return Err(anyhow!("cTrader {ctx}: connection closed before fill")),
            Err(_) => return Err(anyhow!("cTrader {ctx}: no fill before timeout")),
        }
    }
}

enum Exec {
    Filled(OrderResult),
    Rejected(String),
    /// Not terminal for us (ACCEPTED, a heartbeat, an unrelated event): keep waiting.
    Pending,
}

/// Interpret one inbound frame in the context of an in-flight order.
fn interpret(frame: &ProtoMessage, ctx: &str) -> Result<Exec> {
    // A request-level error response is a hard rejection.
    if frame.payload_type == payload_type::ERROR_RES
        || frame.payload_type == payload_type::OA_ERROR_RES
    {
        if let Ok(err) = ProtoOaErrorRes::decode(frame.payload.as_deref().unwrap_or_default()) {
            return Ok(Exec::Rejected(format!(
                "{} ({})",
                err.error_code,
                err.description.unwrap_or_default()
            )));
        }
        return Ok(Exec::Rejected("unknown broker error".into()));
    }

    // cTrader reports a failed order via ProtoOAOrderErrorEvent (e.g. market
    // closed, not enough money) rather than an ORDER_REJECTED execution event.
    if frame.payload_type == payload_type::ORDER_ERROR_EVENT {
        if let Ok(err) =
            ProtoOaOrderErrorEvent::decode(frame.payload.as_deref().unwrap_or_default())
        {
            return Ok(Exec::Rejected(format!(
                "{} ({})",
                err.error_code,
                err.description.unwrap_or_default()
            )));
        }
        return Ok(Exec::Rejected("unknown order error".into()));
    }

    if frame.payload_type != payload_type::EXECUTION_EVENT {
        return Ok(Exec::Pending); // heartbeat / unrelated
    }

    let evt = ProtoOaExecutionEvent::decode(frame.payload.as_deref().unwrap_or_default())
        .map_err(|e| anyhow!("cTrader {ctx}: decode ProtoOAExecutionEvent failed: {e}"))?;

    match evt.execution_type {
        EXEC_ORDER_FILLED => {
            let deal = evt
                .deal
                .as_ref()
                .ok_or_else(|| anyhow!("cTrader {ctx}: ORDER_FILLED without a deal"))?;
            let position_id = deal.position_id;
            let fill = Fill {
                order_id: deal.order_id.to_string(),
                // cTrader timestamps are epoch milliseconds; modelenv uses ns.
                timestamp_ns: deal.execution_timestamp.saturating_mul(1_000_000),
                price: deal.execution_price.unwrap_or(0.0),
                // Raw cTrader filled volume (units; 1000 = 0.01 lot). The gateway
                // converts to modelenv's lot convention.
                size: deal.filled_volume as f64,
                // modelenv side: 0 = buy, 1 = sell (cTrader trade_side 1/2 - 1).
                side: deal.trade_side - 1,
                partial: deal.filled_volume < deal.volume,
            };
            Ok(Exec::Filled(OrderResult { fill, position_id }))
        }
        EXEC_ORDER_REJECTED => Ok(Exec::Rejected(
            evt.error_code.unwrap_or_else(|| "ORDER_REJECTED".into()),
        )),
        EXEC_ORDER_ACCEPTED => Ok(Exec::Pending), // wait for the subsequent FILLED
        _ => Ok(Exec::Pending),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::broker_gateway::ctrader::wire;
    use modelenv_proto::ctrader::ProtoOaDeal;
    use tokio::io::{AsyncRead, AsyncWrite};

    fn exec_event_frame(
        exec_type: i32,
        client_msg_id: Option<String>,
        deal: Option<ProtoOaDeal>,
        error_code: Option<String>,
    ) -> ProtoMessage {
        let evt = ProtoOaExecutionEvent {
            payload_type: Some(payload_type::EXECUTION_EVENT as i32),
            ctid_trader_account_id: 47678494,
            execution_type: exec_type,
            deal,
            error_code,
            ..Default::default()
        };
        wire::envelope(payload_type::EXECUTION_EVENT, evt.encode_to_vec(), client_msg_id)
    }

    fn usdjpy_buy_deal() -> ProtoOaDeal {
        ProtoOaDeal {
            deal_id: 9001,
            order_id: 7001,
            position_id: 5001,
            volume: 1000,
            filled_volume: 1000,
            symbol_id: 4,
            execution_timestamp: 1_700_000_000_000, // ms
            execution_price: Some(150.123),
            trade_side: TRADE_SIDE_BUY,
            ..Default::default()
        }
    }

    /// Mock server: replies to the order request with ORDER_ACCEPTED (correlated),
    /// then pushes an unsolicited ORDER_FILLED; the real cTrader two-step.
    async fn accept_then_fill_server<S>(mut s: S)
    where
        S: AsyncRead + AsyncWrite + Unpin + Send + 'static,
    {
        if let Ok(req) = wire::read_frame(&mut s).await {
            let accepted = exec_event_frame(EXEC_ORDER_ACCEPTED, req.client_msg_id.clone(), None, None);
            let _ = wire::write_frame(&mut s, &accepted).await;
            let filled = exec_event_frame(EXEC_ORDER_FILLED, None, Some(usdjpy_buy_deal()), None);
            let _ = wire::write_frame(&mut s, &filled).await;
        }
    }

    async fn rejecting_server<S>(mut s: S)
    where
        S: AsyncRead + AsyncWrite + Unpin + Send + 'static,
    {
        if let Ok(req) = wire::read_frame(&mut s).await {
            let rejected = exec_event_frame(
                EXEC_ORDER_REJECTED,
                req.client_msg_id.clone(),
                None,
                Some("NOT_ENOUGH_MONEY".into()),
            );
            let _ = wire::write_frame(&mut s, &rejected).await;
        }
    }

    #[tokio::test]
    async fn market_order_fills_via_accepted_then_unsolicited_filled() {
        let (client_io, server_io) = tokio::io::duplex(8192);
        tokio::spawn(accept_then_fill_server(server_io));
        let (cr, cw) = tokio::io::split(client_io);
        let (conn, mut events) = Connection::start(cr, cw);

        let res = submit_market_order(
            &conn, &mut events, 47678494, 4, Side::Buy, 1000, "ord-1",
            Duration::from_secs(5),
        )
        .await
        .unwrap();
        assert_eq!(res.position_id, 5001);
        assert_eq!(res.fill.order_id, "7001");
        assert_eq!(res.fill.price, 150.123);
        assert_eq!(res.fill.size, 1000.0);
        assert_eq!(res.fill.side, 0); // buy -> 0
        assert!(!res.fill.partial);
        assert_eq!(res.fill.timestamp_ns, 1_700_000_000_000 * 1_000_000);
    }

    #[tokio::test]
    async fn market_order_surfaces_rejection() {
        let (client_io, server_io) = tokio::io::duplex(8192);
        tokio::spawn(rejecting_server(server_io));
        let (cr, cw) = tokio::io::split(client_io);
        let (conn, mut events) = Connection::start(cr, cw);

        let err = submit_market_order(
            &conn, &mut events, 47678494, 4, Side::Sell, 1000, "ord-2",
            Duration::from_secs(5),
        )
        .await
        .unwrap_err()
        .to_string();
        assert!(err.contains("NOT_ENOUGH_MONEY"), "got: {err}");
    }

    #[test]
    fn lots_to_volume_matches_demo_minimum() {
        // Demo confirmed 0.01 lot = volume 100_000 (and rejected 1000).
        assert_eq!(lots_to_volume(0.01), 100_000);
        assert_eq!(lots_to_volume(1.0), 10_000_000);
    }

    /// Mock cTrader rejecting the order via ProtoOAOrderErrorEvent (2132); the
    /// real shape of a MARKET_CLOSED / BAD_VOLUME failure, not an exec event.
    async fn order_error_server<S>(mut s: S)
    where
        S: AsyncRead + AsyncWrite + Unpin + Send + 'static,
    {
        if let Ok(req) = wire::read_frame(&mut s).await {
            let err = modelenv_proto::ctrader::ProtoOaOrderErrorEvent {
                payload_type: Some(payload_type::ORDER_ERROR_EVENT as i32),
                ctid_trader_account_id: 47678494,
                error_code: "MARKET_CLOSED".into(),
                description: Some("Trading is not available: Market is closed.".into()),
                ..Default::default()
            };
            let env = wire::envelope(
                payload_type::ORDER_ERROR_EVENT,
                err.encode_to_vec(),
                req.client_msg_id.clone(),
            );
            let _ = wire::write_frame(&mut s, &env).await;
        }
    }

    #[tokio::test]
    async fn market_order_surfaces_order_error_event() {
        let (client_io, server_io) = tokio::io::duplex(8192);
        tokio::spawn(order_error_server(server_io));
        let (cr, cw) = tokio::io::split(client_io);
        let (conn, mut events) = Connection::start(cr, cw);

        let err = submit_market_order(
            &conn, &mut events, 47678494, 4, Side::Buy, 100_000, "ord-e",
            Duration::from_secs(5),
        )
        .await
        .unwrap_err()
        .to_string();
        assert!(err.contains("MARKET_CLOSED"), "got: {err}");
    }

    #[tokio::test]
    async fn close_position_fills() {
        // Closing also resolves via a FILLED execution event with a deal.
        let (client_io, server_io) = tokio::io::duplex(8192);
        tokio::spawn(async move {
            let mut s = server_io;
            if let Ok(req) = wire::read_frame(&mut s).await {
                let mut deal = usdjpy_buy_deal();
                deal.trade_side = TRADE_SIDE_SELL; // closing a long = a sell deal
                let filled = exec_event_frame(EXEC_ORDER_FILLED, req.client_msg_id.clone(), Some(deal), None);
                let _ = wire::write_frame(&mut s, &filled).await;
            }
        });
        let (cr, cw) = tokio::io::split(client_io);
        let (conn, mut events) = Connection::start(cr, cw);

        let res = close_position(&conn, &mut events, 47678494, 5001, 1000, Duration::from_secs(5))
            .await
            .unwrap();
        assert_eq!(res.position_id, 5001);
        assert_eq!(res.fill.side, 1); // sell -> 1
    }
}
