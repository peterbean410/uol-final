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
    Ok(())
}
