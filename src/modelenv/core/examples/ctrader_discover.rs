//! Read-only cTrader discovery + foundation validation.
//!
//! Connects to the cTrader **demo** endpoint (unless `CTRADER_LIVE=true`),
//! performs application auth, and lists every trading account the access token
//! can reach, printing each account's id and demo/live flag. Places **no
//! orders**. This is the first real-network exercise of the transport, framing,
//! auth and request/response correlation layers.
//!
//! Run (creds from env, never hard-coded):
//! ```sh
//! CTRADER_APP_CLIENT_ID=… CTRADER_APP_CLIENT_SECRET=… CTRADER_ACCESS_TOKEN=… \
//!   cargo run -p modelenv-core --example ctrader_discover
//! ```

use std::time::Duration;

use modelenv_core::broker_gateway::ctrader::{auth, connection::Connection, data, transport::Transport};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let client_id = std::env::var("CTRADER_APP_CLIENT_ID")?;
    let client_secret = std::env::var("CTRADER_APP_CLIENT_SECRET")?;
    let access_token = std::env::var("CTRADER_ACCESS_TOKEN")?;
    let live = std::env::var("CTRADER_LIVE").ok().as_deref() == Some("true");
    let timeout = Duration::from_secs(15);

    eprintln!("[discover] connecting to cTrader {} ...", if live { "LIVE" } else { "demo" });
    let transport = Transport::connect_env(live).await?;
    eprintln!("[discover] TLS connected: {}", transport.endpoint());

    let (r, w) = transport.into_split();
    let (conn, _events) = Connection::start(r, w);

    eprintln!("[discover] application auth ...");
    auth::app_authenticate(&conn, &client_id, &client_secret, timeout).await?;
    eprintln!("[discover] application auth OK");

    eprintln!("[discover] fetching account list ...");
    let accounts = data::get_account_list(&conn, &access_token, timeout).await?;

    println!("[discover] {} account(s):", accounts.len());
    for a in &accounts {
        let kind = match a.is_live {
            Some(true) => "LIVE",
            Some(false) => "demo",
            None => "unknown",
        };
        println!(
            "  ctidTraderAccountId={} kind={} traderLogin={:?}",
            a.ctid_trader_account_id, kind, a.trader_login
        );
    }

    // Pick the first DEMO account and exercise account-auth + symbol resolution
    // (still read-only). Refuse to proceed on a live account here.
    let demo = accounts.iter().find(|a| a.is_live == Some(false));
    let Some(demo) = demo else {
        eprintln!("[discover] no demo account reachable; stopping (read-only).");
        return Ok(());
    };
    let account_id = demo.ctid_trader_account_id as i64;
    eprintln!("[discover] account auth on demo account {account_id} ...");
    // App auth already happened above; do ONLY the account-auth stage (a second
    // app auth on the same connection would be rejected as a duplicate).
    auth::account_authenticate(&conn, &access_token, account_id, timeout).await?;
    eprintln!("[discover] account auth OK");

    let symbol = std::env::var("CTRADER_SYMBOL").unwrap_or_else(|_| "USDJPY".into());
    eprintln!("[discover] resolving symbol {symbol} ...");
    let symbol_id = data::get_symbol_id(&conn, account_id, &symbol, timeout).await?;
    println!("[discover] {symbol} symbol_id={symbol_id} on demo account {account_id}");
    Ok(())
}
