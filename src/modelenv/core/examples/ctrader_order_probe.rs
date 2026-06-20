//! Diagnostic: submit a demo MARKET order and dump every execution frame
//! cTrader returns for ~10s (correlated reply + unsolicited events), without
//! interpreting them. Used to observe the real response when the market is
//! closed (no fill expected), confirming the submission is well-formed and
//! accepted. Demo only.

use std::time::Duration;

use modelenv_core::broker_gateway::ctrader::{
    auth, connection::Connection, data, transport::Transport, wire,
};
use modelenv_proto::ctrader::{ProtoMessage, ProtoOaExecutionEvent, ProtoOaNewOrderReq};
use prost::Message;

fn describe(f: &ProtoMessage) -> String {
    if f.payload_type == wire::payload_type::EXECUTION_EVENT {
        if let Ok(e) = ProtoOaExecutionEvent::decode(f.payload.as_deref().unwrap_or_default()) {
            return format!(
                "EXECUTION_EVENT execution_type={} error_code={:?} has_deal={} has_order={}",
                e.execution_type,
                e.error_code,
                e.deal.is_some(),
                e.order.is_some()
            );
        }
    }
    format!("payload_type={}", f.payload_type)
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let client_id = std::env::var("CTRADER_APP_CLIENT_ID")?;
    let client_secret = std::env::var("CTRADER_APP_CLIENT_SECRET")?;
    let access_token = std::env::var("CTRADER_ACCESS_TOKEN")?;
    let timeout = Duration::from_secs(20);

    let (r, w) = Transport::connect_env(false).await?.into_split();
    let (conn, mut events) = Connection::start(r, w);
    auth::app_authenticate(&conn, &client_id, &client_secret, timeout).await?;
    let accounts = data::get_account_list(&conn, &access_token, timeout).await?;
    let demo = accounts.iter().find(|a| a.is_live == Some(false)).unwrap();
    let account_id = demo.ctid_trader_account_id as i64;
    auth::account_authenticate(&conn, &access_token, account_id, timeout).await?;
    let symbol_id = data::get_symbol_id(&conn, account_id, "USDJPY", timeout).await?;

    let req = ProtoOaNewOrderReq {
        payload_type: Some(wire::payload_type::NEW_ORDER_REQ as i32),
        ctid_trader_account_id: account_id,
        symbol_id,
        order_type: 1, // MARKET
        trade_side: 1, // BUY
        volume: 1000,  // 0.01 lot
        client_order_id: Some("probe".into()),
        ..Default::default()
    };
    eprintln!("[probe] sending MARKET BUY 1000 USDJPY (symbol_id={symbol_id}) ...");
    let reply = conn
        .send_request(wire::payload_type::NEW_ORDER_REQ, req.encode_to_vec(), timeout)
        .await?;
    println!("[probe] correlated reply: {}", describe(&reply));

    // Drain follow-up events for ~10s.
    let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
    loop {
        let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
        if remaining.is_zero() {
            break;
        }
        match tokio::time::timeout(remaining, events.recv()).await {
            Ok(Some(f)) => println!("[probe] event: {}", describe(&f)),
            Ok(None) => break,
            Err(_) => break,
        }
    }
    println!("[probe] done.");
    Ok(())
}
