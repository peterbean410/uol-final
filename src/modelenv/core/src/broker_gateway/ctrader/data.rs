//! cTrader Open API read-only data RPCs.
//!
//! These never place or modify orders; they read account/market state over an
//! authenticated [`Connection`]. The first and simplest, [`get_account_list`]
//!, also doubles as the safe discovery + validation step: it needs only
//! application auth (no specific account, no account-auth) and returns every
//! trading account the access token can reach, each flagged demo vs live, so a
//! caller can pick a demo account before any order path is exercised.

use std::time::Duration;

use anyhow::{anyhow, Result};
use modelenv_proto::ctrader::{
    ProtoOaCtidTraderAccount, ProtoOaDeal, ProtoOaDealListReq, ProtoOaDealListRes, ProtoOaErrorRes,
    ProtoOaGetAccountListByAccessTokenReq, ProtoOaGetAccountListByAccessTokenRes,
    ProtoOaGetTrendbarsReq, ProtoOaGetTrendbarsRes, ProtoOaPosition, ProtoOaReconcileReq,
    ProtoOaReconcileRes, ProtoOaSpotEvent, ProtoOaSubscribeSpotsReq, ProtoOaSubscribeSpotsRes,
    ProtoOaSymbolsListReq, ProtoOaSymbolsListRes, ProtoOaTrendbar,
};
use modelenv_proto::{Bar, Fill, Position};
use prost::Message;

use super::connection::Connection;
use super::orders::volume_to_lots;
use super::wire::payload_type;
use crate::position::{ClosedPosition, Side};

// cTrader money fields (swap, commission) are integers in `money_digits`
// precision; the default for most accounts is 2 (i.e. hundredths).
const MONEY_SCALE: f64 = 100.0;
// cTrader trendbar prices are integers scaled by 10^5 (fixed, regardless of the
// symbol's display digits), delta-encoded off the bar `low`.
const TRENDBAR_PRICE_SCALE: f64 = 100_000.0;
/// cTrader trendbar period discriminants (`ProtoOATrendbarPeriod`).
pub const TRENDBAR_M1: i32 = 1;
pub const TRENDBAR_M5: i32 = 5;
pub const TRENDBAR_M15: i32 = 7;

/// Map a cTrader [`ProtoOaPosition`] to a modelenv [`Position`]. Volume is
/// converted from cTrader units to lots; `unrealised_pnl` is left 0.0 (modelenv
/// recomputes it from the current price); side is `trade_side - 1` (0=buy,
/// 1=sell).
fn to_modelenv_position(p: &ProtoOaPosition) -> Position {
    Position {
        position_id: p.position_id.to_string(),
        entry_price: p.price.unwrap_or(0.0),
        unrealised_pnl: 0.0,
        swap: p.swap as f64 / MONEY_SCALE,
        open_timestamp_ns: p
            .trade_data
            .open_timestamp
            .unwrap_or(0)
            .saturating_mul(1_000_000),
        volume: volume_to_lots(p.trade_data.volume),
        side: p.trade_data.trade_side - 1,
    }
}

/// Reconcile: fetch the account's currently OPEN positions and map them to
/// modelenv [`Position`]s. Used by live `Reset()` to sync the broker book.
/// Read-only.
pub async fn sync_positions(
    conn: &Connection,
    account_id: i64,
    timeout: Duration,
) -> Result<Vec<Position>> {
    let req = ProtoOaReconcileReq {
        payload_type: Some(payload_type::RECONCILE_REQ as i32),
        ctid_trader_account_id: account_id,
        ..Default::default()
    };
    let resp = conn
        .send_request(payload_type::RECONCILE_REQ, req.encode_to_vec(), timeout)
        .await?;

    if resp.payload_type == payload_type::RECONCILE_RES {
        let decoded = ProtoOaReconcileRes::decode(resp.payload.as_deref().unwrap_or_default())
            .map_err(|e| anyhow!("decode ProtoOAReconcileRes failed: {e}"))?;
        return Ok(decoded.position.iter().map(to_modelenv_position).collect());
    }
    if resp.payload_type == payload_type::ERROR_RES
        || resp.payload_type == payload_type::OA_ERROR_RES
    {
        if let Ok(err) = ProtoOaErrorRes::decode(resp.payload.as_deref().unwrap_or_default()) {
            return Err(anyhow!(
                "cTrader reconcile rejected: {} ({})",
                err.error_code,
                err.description.unwrap_or_default()
            ));
        }
    }
    Err(anyhow!(
        "cTrader reconcile: unexpected payload_type {} (expected {})",
        resp.payload_type,
        payload_type::RECONCILE_RES
    ))
}

/// Fetch all trading accounts authorized by `access_token`. Read-only; requires
/// only prior application auth. Each [`ProtoOaCtidTraderAccount`] carries its
/// `ctid_trader_account_id` and an `is_live` flag (false = demo).
pub async fn get_account_list(
    conn: &Connection,
    access_token: &str,
    timeout: Duration,
) -> Result<Vec<ProtoOaCtidTraderAccount>> {
    let req = ProtoOaGetAccountListByAccessTokenReq {
        payload_type: Some(payload_type::GET_ACCOUNTS_BY_ACCESS_TOKEN_REQ as i32),
        access_token: access_token.to_string(),
    };
    let resp = conn
        .send_request(
            payload_type::GET_ACCOUNTS_BY_ACCESS_TOKEN_REQ,
            req.encode_to_vec(),
            timeout,
        )
        .await?;

    if resp.payload_type == payload_type::GET_ACCOUNTS_BY_ACCESS_TOKEN_RES {
        let decoded = ProtoOaGetAccountListByAccessTokenRes::decode(
            resp.payload.as_deref().unwrap_or_default(),
        )
        .map_err(|e| anyhow!("decode ProtoOAGetAccountListByAccessTokenRes failed: {e}"))?;
        return Ok(decoded.ctid_trader_account);
    }

    if resp.payload_type == payload_type::ERROR_RES
        || resp.payload_type == payload_type::OA_ERROR_RES
    {
        if let Ok(err) = ProtoOaErrorRes::decode(resp.payload.as_deref().unwrap_or_default()) {
            return Err(anyhow!(
                "cTrader account-list rejected: {} ({})",
                err.error_code,
                err.description.unwrap_or_default()
            ));
        }
    }
    Err(anyhow!(
        "cTrader account-list: unexpected payload_type {} (expected {})",
        resp.payload_type,
        payload_type::GET_ACCOUNTS_BY_ACCESS_TOKEN_RES
    ))
}

/// Resolve a broker symbol name (e.g. "USDJPY") to its numeric `symbol_id` on
/// `account_id`. Read-only; requires prior account auth. Symbol ids are
/// broker-specific, so an order must use the id from this lookup, never a
/// hard-coded guess. Matches case-insensitively on `symbol_name`.
pub async fn get_symbol_id(
    conn: &Connection,
    account_id: i64,
    symbol_name: &str,
    timeout: Duration,
) -> Result<i64> {
    let req = ProtoOaSymbolsListReq {
        payload_type: Some(payload_type::SYMBOLS_LIST_REQ as i32),
        ctid_trader_account_id: account_id,
        include_archived_symbols: Some(false),
    };
    let resp = conn
        .send_request(payload_type::SYMBOLS_LIST_REQ, req.encode_to_vec(), timeout)
        .await?;

    if resp.payload_type == payload_type::SYMBOLS_LIST_RES {
        let decoded = ProtoOaSymbolsListRes::decode(resp.payload.as_deref().unwrap_or_default())
            .map_err(|e| anyhow!("decode ProtoOASymbolsListRes failed: {e}"))?;
        let want = symbol_name.trim().to_ascii_uppercase();
        let found = decoded.symbol.iter().find(|s| {
            s.symbol_name
                .as_deref()
                .map(|n| n.trim().to_ascii_uppercase() == want)
                .unwrap_or(false)
        });
        return found
            .map(|s| s.symbol_id)
            .ok_or_else(|| anyhow!("symbol {symbol_name:?} not found on account {account_id}"));
    }

    if resp.payload_type == payload_type::ERROR_RES
        || resp.payload_type == payload_type::OA_ERROR_RES
    {
        if let Ok(err) = ProtoOaErrorRes::decode(resp.payload.as_deref().unwrap_or_default()) {
            return Err(anyhow!(
                "cTrader symbols-list rejected: {} ({})",
                err.error_code,
                err.description.unwrap_or_default()
            ));
        }
    }
    Err(anyhow!(
        "cTrader symbols-list: unexpected payload_type {} (expected {})",
        resp.payload_type,
        payload_type::SYMBOLS_LIST_RES
    ))
}

/// Decode one cTrader [`ProtoOaTrendbar`] (delta-encoded off `low`, prices ×10⁵,
/// timestamp in minutes) into a modelenv [`Bar`].
fn to_modelenv_bar(tb: &ProtoOaTrendbar) -> Bar {
    let low = tb.low.unwrap_or(0);
    let price = |delta: u64| (low as f64 + delta as f64) / TRENDBAR_PRICE_SCALE;
    Bar {
        // utc_timestamp_in_minutes (minutes since epoch) → ns.
        timestamp_ns: tb.utc_timestamp_in_minutes.unwrap_or(0) as i64 * 60_000_000_000,
        open: price(tb.delta_open.unwrap_or(0)),
        high: price(tb.delta_high.unwrap_or(0)),
        low: low as f64 / TRENDBAR_PRICE_SCALE,
        close: price(tb.delta_close.unwrap_or(0)),
        volume: tb.volume as f64,
    }
}

/// Fetch the most recent `count` trendbars of `period` for `symbol_id` on
/// `account_id`, oldest→newest, as modelenv [`Bar`]s. Read-only; works even
/// when the market is closed (historical data). `now_ms` is the current epoch
/// time in milliseconds (the window is `[now - count·period, now]`).
pub async fn get_trendbars(
    conn: &Connection,
    account_id: i64,
    symbol_id: i64,
    period: i32,
    count: u32,
    now_ms: i64,
    timeout: Duration,
) -> Result<Vec<Bar>> {
    let period_ms: i64 = match period {
        TRENDBAR_M1 => 60_000,
        TRENDBAR_M5 => 300_000,
        TRENDBAR_M15 => 900_000,
        _ => 60_000,
    };
    // Generous window so cTrader returns at least `count` completed bars.
    let from = now_ms - (count as i64 + 2) * period_ms;
    let req = ProtoOaGetTrendbarsReq {
        payload_type: Some(payload_type::GET_TRENDBARS_REQ as i32),
        ctid_trader_account_id: account_id,
        from_timestamp: from,
        to_timestamp: now_ms,
        period,
        symbol_id,
        count: Some(count),
        ..Default::default()
    };
    let resp = conn
        .send_request(payload_type::GET_TRENDBARS_REQ, req.encode_to_vec(), timeout)
        .await?;

    if resp.payload_type == payload_type::GET_TRENDBARS_RES {
        let decoded = ProtoOaGetTrendbarsRes::decode(resp.payload.as_deref().unwrap_or_default())
            .map_err(|e| anyhow!("decode ProtoOAGetTrendbarsRes failed: {e}"))?;
        return Ok(decoded.trendbar.iter().map(to_modelenv_bar).collect());
    }
    if resp.payload_type == payload_type::ERROR_RES
        || resp.payload_type == payload_type::OA_ERROR_RES
    {
        if let Ok(err) = ProtoOaErrorRes::decode(resp.payload.as_deref().unwrap_or_default()) {
            return Err(anyhow!(
                "cTrader trendbars rejected: {} ({})",
                err.error_code,
                err.description.unwrap_or_default()
            ));
        }
    }
    Err(anyhow!(
        "cTrader trendbars: unexpected payload_type {} (expected {})",
        resp.payload_type,
        payload_type::GET_TRENDBARS_RES
    ))
}

/// cTrader spot (bid/ask) prices are integers scaled by 10^5, like trendbars.
const SPOT_PRICE_SCALE: f64 = 100_000.0;

/// Subscribe to streaming spot (bid/ask) events for `symbol_id` on `account_id`.
/// Once this returns Ok, cTrader pushes unsolicited `ProtoOASpotEvent`s over the
/// connection (delivered on its events channel). Required for live tick data.
pub async fn subscribe_spots(
    conn: &Connection,
    account_id: i64,
    symbol_id: i64,
    timeout: Duration,
) -> Result<()> {
    let req = ProtoOaSubscribeSpotsReq {
        payload_type: Some(payload_type::SUBSCRIBE_SPOTS_REQ as i32),
        ctid_trader_account_id: account_id,
        symbol_id: vec![symbol_id],
        subscribe_to_spot_timestamp: Some(true),
    };
    let resp = conn
        .send_request(payload_type::SUBSCRIBE_SPOTS_REQ, req.encode_to_vec(), timeout)
        .await?;
    if resp.payload_type == payload_type::SUBSCRIBE_SPOTS_RES {
        // Validate the account echo (defensive; the RES only carries the account).
        let _ = ProtoOaSubscribeSpotsRes::decode(resp.payload.as_deref().unwrap_or_default());
        return Ok(());
    }
    if resp.payload_type == payload_type::ERROR_RES
        || resp.payload_type == payload_type::OA_ERROR_RES
    {
        if let Ok(err) = ProtoOaErrorRes::decode(resp.payload.as_deref().unwrap_or_default()) {
            return Err(anyhow!(
                "cTrader subscribe-spots rejected: {} ({})",
                err.error_code,
                err.description.unwrap_or_default()
            ));
        }
    }
    Err(anyhow!(
        "cTrader subscribe-spots: unexpected payload_type {} (expected {})",
        resp.payload_type,
        payload_type::SUBSCRIBE_SPOTS_RES
    ))
}

/// Decode a [`ProtoOaSpotEvent`]'s bid/ask (each optional, ×10^5) and timestamp
/// (ms→ns). Returns `(bid, ask, ts_ns)`; a spot event updates only the side(s)
/// that changed, so either price may be `None`. `ts_ns` is 0 when absent (the
/// caller substitutes the receive time).
pub fn spot_bid_ask(evt: &ProtoOaSpotEvent) -> (Option<f64>, Option<f64>, i64) {
    let bid = evt.bid.map(|b| b as f64 / SPOT_PRICE_SCALE);
    let ask = evt.ask.map(|a| a as f64 / SPOT_PRICE_SCALE);
    let ts_ns = evt.timestamp.unwrap_or(0).saturating_mul(1_000_000);
    (bid, ask, ts_ns)
}

/// Map a cTrader [`ProtoOaDeal`] to a modelenv [`Fill`] (lots, side = trade_side
/// − 1, ms→ns). Shared with the order path.
fn deal_to_fill(d: &ProtoOaDeal) -> Fill {
    Fill {
        order_id: d.order_id.to_string(),
        timestamp_ns: d.execution_timestamp.saturating_mul(1_000_000),
        price: d.execution_price.unwrap_or(0.0),
        size: volume_to_lots(d.filled_volume),
        side: d.trade_side - 1,
        partial: d.filled_volume < d.volume,
    }
}

/// Map a *closing* deal (one carrying a `close_position_detail`) to a modelenv
/// [`ClosedPosition`]. The closing deal's `trade_side` is OPPOSITE the
/// position's (closing a long is a sell deal), so the position side is the
/// inverse. `realised_pnl`/`swap` are scaled by the detail's `money_digits`.
/// NOTE: the realised-P&L scaling is mock-verified; confirm against a real
/// closed demo deal (needs the market open) before relying on it live.
fn deal_to_closed_position(d: &ProtoOaDeal) -> Option<ClosedPosition> {
    let detail = d.close_position_detail.as_ref()?;
    let money_scale = 10f64.powi(detail.money_digits.unwrap_or(2) as i32);
    let side = if d.trade_side == 1 { Side::Sell } else { Side::Buy };
    Some(ClosedPosition {
        position_id: d.position_id.to_string(),
        entry_price: detail.entry_price,
        close_price: d.execution_price.unwrap_or(0.0),
        volume: volume_to_lots(detail.closed_volume.unwrap_or(d.filled_volume)),
        side,
        realised_pnl: detail.gross_profit as f64 / money_scale,
        swap: detail.swap as f64 / money_scale,
        open_timestamp_ns: 0, // not carried on the close deal
        close_timestamp_ns: d.execution_timestamp.saturating_mul(1_000_000),
    })
}

/// Fetch the account's deal (execution) history in `[from_ms, to_ms]`.
/// Read-only. Underlies both `recent_fills` and `closed_positions`.
async fn get_deals(
    conn: &Connection,
    account_id: i64,
    from_ms: i64,
    to_ms: i64,
    timeout: Duration,
) -> Result<Vec<ProtoOaDeal>> {
    let req = ProtoOaDealListReq {
        payload_type: Some(payload_type::DEAL_LIST_REQ as i32),
        ctid_trader_account_id: account_id,
        from_timestamp: from_ms,
        to_timestamp: to_ms,
        ..Default::default()
    };
    let resp = conn
        .send_request(payload_type::DEAL_LIST_REQ, req.encode_to_vec(), timeout)
        .await?;
    if resp.payload_type == payload_type::DEAL_LIST_RES {
        let decoded = ProtoOaDealListRes::decode(resp.payload.as_deref().unwrap_or_default())
            .map_err(|e| anyhow!("decode ProtoOADealListRes failed: {e}"))?;
        return Ok(decoded.deal);
    }
    if resp.payload_type == payload_type::ERROR_RES
        || resp.payload_type == payload_type::OA_ERROR_RES
    {
        if let Ok(err) = ProtoOaErrorRes::decode(resp.payload.as_deref().unwrap_or_default()) {
            return Err(anyhow!(
                "cTrader deal-list rejected: {} ({})",
                err.error_code,
                err.description.unwrap_or_default()
            ));
        }
    }
    Err(anyhow!(
        "cTrader deal-list: unexpected payload_type {} (expected {})",
        resp.payload_type,
        payload_type::DEAL_LIST_RES
    ))
}

/// Up to `count` most-recent fills (deals) in `[from_ms, to_ms]`, newest first.
pub async fn recent_fills(
    conn: &Connection,
    account_id: i64,
    from_ms: i64,
    to_ms: i64,
    count: usize,
    timeout: Duration,
) -> Result<Vec<Fill>> {
    let mut deals = get_deals(conn, account_id, from_ms, to_ms, timeout).await?;
    deals.sort_by_key(|d| std::cmp::Reverse(d.execution_timestamp));
    Ok(deals.iter().take(count).map(deal_to_fill).collect())
}

/// Closed positions in `[from_ms, to_ms]` (deals carrying a close detail),
/// used by live `Reset()` to rebuild the rolling realised-P&L window.
pub async fn closed_positions(
    conn: &Connection,
    account_id: i64,
    from_ms: i64,
    to_ms: i64,
    timeout: Duration,
) -> Result<Vec<ClosedPosition>> {
    let deals = get_deals(conn, account_id, from_ms, to_ms, timeout).await?;
    Ok(deals.iter().filter_map(deal_to_closed_position).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::broker_gateway::ctrader::wire;
    use modelenv_proto::ctrader::{ProtoOaClosePositionDetail, ProtoOaLightSymbol};
    use tokio::io::{AsyncRead, AsyncWrite};

    /// Mock cTrader that returns a two-account list (one demo, one live).
    async fn account_list_server<S>(mut s: S)
    where
        S: AsyncRead + AsyncWrite + Unpin + Send + 'static,
    {
        if let Ok(req) = wire::read_frame(&mut s).await {
            let res = ProtoOaGetAccountListByAccessTokenRes {
                payload_type: Some(payload_type::GET_ACCOUNTS_BY_ACCESS_TOKEN_RES as i32),
                access_token: "tok".into(),
                permission_scope: None,
                ctid_trader_account: vec![
                    ProtoOaCtidTraderAccount {
                        ctid_trader_account_id: 111,
                        is_live: Some(false),
                        trader_login: Some(5001),
                        ..Default::default()
                    },
                    ProtoOaCtidTraderAccount {
                        ctid_trader_account_id: 222,
                        is_live: Some(true),
                        trader_login: Some(9001),
                        ..Default::default()
                    },
                ],
            };
            let env = wire::envelope(
                payload_type::GET_ACCOUNTS_BY_ACCESS_TOKEN_RES,
                res.encode_to_vec(),
                req.client_msg_id.clone(),
            );
            let _ = wire::write_frame(&mut s, &env).await;
        }
    }

    #[tokio::test]
    async fn get_account_list_parses_demo_and_live_accounts() {
        let (client_io, server_io) = tokio::io::duplex(4096);
        tokio::spawn(account_list_server(server_io));
        let (cr, cw) = tokio::io::split(client_io);
        let (conn, _events) = Connection::start(cr, cw);

        let accounts = get_account_list(&conn, "tok", Duration::from_secs(5))
            .await
            .unwrap();
        assert_eq!(accounts.len(), 2);
        let demo: Vec<_> = accounts.iter().filter(|a| a.is_live == Some(false)).collect();
        assert_eq!(demo.len(), 1);
        assert_eq!(demo[0].ctid_trader_account_id, 111);
    }

    /// Mock cTrader returning a small symbol list including USDJPY=id 4.
    async fn symbols_server<S>(mut s: S)
    where
        S: AsyncRead + AsyncWrite + Unpin + Send + 'static,
    {
        if let Ok(req) = wire::read_frame(&mut s).await {
            let res = ProtoOaSymbolsListRes {
                payload_type: Some(payload_type::SYMBOLS_LIST_RES as i32),
                ctid_trader_account_id: 47678494,
                symbol: vec![
                    ProtoOaLightSymbol {
                        symbol_id: 1,
                        symbol_name: Some("EURUSD".into()),
                        ..Default::default()
                    },
                    ProtoOaLightSymbol {
                        symbol_id: 4,
                        symbol_name: Some("USDJPY".into()),
                        ..Default::default()
                    },
                ],
                archived_symbol: vec![],
            };
            let env = wire::envelope(
                payload_type::SYMBOLS_LIST_RES,
                res.encode_to_vec(),
                req.client_msg_id.clone(),
            );
            let _ = wire::write_frame(&mut s, &env).await;
        }
    }

    #[tokio::test]
    async fn get_symbol_id_resolves_case_insensitively() {
        let (client_io, server_io) = tokio::io::duplex(4096);
        tokio::spawn(symbols_server(server_io));
        let (cr, cw) = tokio::io::split(client_io);
        let (conn, _events) = Connection::start(cr, cw);

        let id = get_symbol_id(&conn, 47678494, "usdjpy", Duration::from_secs(5))
            .await
            .unwrap();
        assert_eq!(id, 4);
    }

    #[tokio::test]
    async fn get_symbol_id_errors_on_unknown_symbol() {
        let (client_io, server_io) = tokio::io::duplex(4096);
        tokio::spawn(symbols_server(server_io));
        let (cr, cw) = tokio::io::split(client_io);
        let (conn, _events) = Connection::start(cr, cw);

        let res = get_symbol_id(&conn, 47678494, "XAUUSD", Duration::from_secs(5)).await;
        assert!(res.is_err());
    }

    /// Mock cTrader returning one open long position (0.01 lot USDJPY).
    async fn reconcile_server<S>(mut s: S)
    where
        S: AsyncRead + AsyncWrite + Unpin + Send + 'static,
    {
        use modelenv_proto::ctrader::ProtoOaTradeData;
        if let Ok(req) = wire::read_frame(&mut s).await {
            let res = ProtoOaReconcileRes {
                payload_type: Some(payload_type::RECONCILE_RES as i32),
                ctid_trader_account_id: 47678494,
                position: vec![ProtoOaPosition {
                    position_id: 5001,
                    trade_data: ProtoOaTradeData {
                        symbol_id: 4,
                        volume: 100_000, // 0.01 lot
                        trade_side: 1,   // BUY
                        open_timestamp: Some(1_700_000_000_000),
                        ..Default::default()
                    },
                    price: Some(150.5),
                    swap: 25, // 0.25 in money units
                    ..Default::default()
                }],
                order: vec![],
            };
            let env = wire::envelope(
                payload_type::RECONCILE_RES,
                res.encode_to_vec(),
                req.client_msg_id.clone(),
            );
            let _ = wire::write_frame(&mut s, &env).await;
        }
    }

    #[test]
    fn closing_deal_maps_to_closed_position_with_realised_pnl() {
        // A SELL deal (trade_side=2) that closes a LONG position, with a close
        // detail: entry 150.00, +1234 gross (money_digits=2 -> 12.34), swap -25.
        let deal = ProtoOaDeal {
            deal_id: 1,
            order_id: 7001,
            position_id: 5001,
            volume: 100_000,
            filled_volume: 100_000,
            symbol_id: 4,
            execution_timestamp: 1_700_000_100_000,
            execution_price: Some(150.50),
            trade_side: 2, // SELL closes a long
            close_position_detail: Some(ProtoOaClosePositionDetail {
                entry_price: 150.00,
                gross_profit: 1234,
                swap: -25,
                commission: 0,
                balance: 0,
                closed_volume: Some(100_000),
                money_digits: Some(2),
                ..Default::default()
            }),
            ..Default::default()
        };
        let cp = deal_to_closed_position(&deal).unwrap();
        assert_eq!(cp.position_id, "5001");
        assert_eq!(cp.side, Side::Buy); // SELL deal closed a BUY (long)
        assert!((cp.entry_price - 150.00).abs() < 1e-9);
        assert!((cp.close_price - 150.50).abs() < 1e-9);
        assert!((cp.realised_pnl - 12.34).abs() < 1e-9);
        assert!((cp.swap - (-0.25)).abs() < 1e-9);
        assert!((cp.volume - 0.01).abs() < 1e-9);
    }

    /// Mock cTrader returning a deal list with one open (no close detail) and
    /// one closing deal.
    async fn deal_list_server<S>(mut s: S)
    where
        S: AsyncRead + AsyncWrite + Unpin + Send + 'static,
    {
        if let Ok(req) = wire::read_frame(&mut s).await {
            let open = ProtoOaDeal {
                order_id: 7001,
                position_id: 5001,
                volume: 100_000,
                filled_volume: 100_000,
                symbol_id: 4,
                execution_timestamp: 1_700_000_000_000,
                execution_price: Some(150.00),
                trade_side: 1,
                ..Default::default()
            };
            let close = ProtoOaDeal {
                order_id: 7002,
                position_id: 5001,
                volume: 100_000,
                filled_volume: 100_000,
                symbol_id: 4,
                execution_timestamp: 1_700_000_100_000,
                execution_price: Some(150.50),
                trade_side: 2,
                close_position_detail: Some(ProtoOaClosePositionDetail {
                    entry_price: 150.00,
                    gross_profit: 1234,
                    swap: 0,
                    commission: 0,
                    balance: 0,
                    closed_volume: Some(100_000),
                    money_digits: Some(2),
                    ..Default::default()
                }),
                ..Default::default()
            };
            let res = ProtoOaDealListRes {
                payload_type: Some(payload_type::DEAL_LIST_RES as i32),
                ctid_trader_account_id: 47678494,
                deal: vec![open, close],
                has_more: false,
            };
            let env = wire::envelope(
                payload_type::DEAL_LIST_RES,
                res.encode_to_vec(),
                req.client_msg_id.clone(),
            );
            let _ = wire::write_frame(&mut s, &env).await;
        }
    }

    #[tokio::test]
    async fn recent_fills_and_closed_positions_from_deal_list() {
        let (client_io, server_io) = tokio::io::duplex(8192);
        tokio::spawn(deal_list_server(server_io));
        let (cr, cw) = tokio::io::split(client_io);
        let (conn, _events) = Connection::start(cr, cw);

        let fills = recent_fills(&conn, 47678494, 0, 1_800_000_000_000, 10, Duration::from_secs(5))
            .await
            .unwrap();
        assert_eq!(fills.len(), 2);
        assert_eq!(fills[0].order_id, "7002"); // newest first

        // closed_positions filters to the one closing deal (fresh connection).
        let (client_io2, server_io2) = tokio::io::duplex(8192);
        tokio::spawn(deal_list_server(server_io2));
        let (cr3, cw3) = tokio::io::split(client_io2);
        let (conn2, _e2) = Connection::start(cr3, cw3);
        let closed = closed_positions(&conn2, 47678494, 0, 1_800_000_000_000, Duration::from_secs(5))
            .await
            .unwrap();
        assert_eq!(closed.len(), 1);
        assert_eq!(closed[0].position_id, "5001");
        assert!((closed[0].realised_pnl - 12.34).abs() < 1e-9);
    }

    #[test]
    fn trendbar_delta_decoding_reconstructs_ohlc() {
        // low=15010000 (=150.10 at x10^5); deltas give O=150.105 H=150.123 C=150.118.
        let tb = ProtoOaTrendbar {
            volume: 42,
            low: Some(15_010_000),
            delta_open: Some(500),   // 150.105
            delta_high: Some(2_300), // 150.123
            delta_close: Some(1_800),// 150.118
            utc_timestamp_in_minutes: Some(28_350_000),
            ..Default::default()
        };
        let bar = to_modelenv_bar(&tb);
        assert!((bar.low - 150.10).abs() < 1e-6);
        assert!((bar.open - 150.105).abs() < 1e-6);
        assert!((bar.high - 150.123).abs() < 1e-6);
        assert!((bar.close - 150.118).abs() < 1e-6);
        assert_eq!(bar.volume, 42.0);
        assert_eq!(bar.timestamp_ns, 28_350_000i64 * 60_000_000_000);
    }

    #[tokio::test]
    async fn sync_positions_maps_ctrader_position_to_modelenv() {
        let (client_io, server_io) = tokio::io::duplex(4096);
        tokio::spawn(reconcile_server(server_io));
        let (cr, cw) = tokio::io::split(client_io);
        let (conn, _events) = Connection::start(cr, cw);

        let positions = sync_positions(&conn, 47678494, Duration::from_secs(5))
            .await
            .unwrap();
        assert_eq!(positions.len(), 1);
        let p = &positions[0];
        assert_eq!(p.position_id, "5001");
        assert_eq!(p.entry_price, 150.5);
        assert_eq!(p.volume, 0.01); // 100_000 units -> 0.01 lot
        assert_eq!(p.side, 0); // buy -> 0
        assert_eq!(p.swap, 0.25); // 25 / 100
        assert_eq!(p.open_timestamp_ns, 1_700_000_000_000 * 1_000_000);
    }
}
