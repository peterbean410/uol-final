//! cTrader DEMO order round-trip validation (T-9.1-06 step 7).
//!
//! Connects to the cTrader **demo** endpoint, authenticates, resolves the
//! symbol, places ONE minimum-size MARKET order, prints the fill, then
//! immediately closes the resulting position and prints that fill. Hard refuses
//! to run against a live account.
//!
//! Run (creds from env; volume in cTrader units, 1000 = 0.01 lot):
//! ```sh
//! CTRADER_APP_CLIENT_ID=… CTRADER_APP_CLIENT_SECRET=… CTRADER_ACCESS_TOKEN=… \
//!   cargo run -p modelenv-core --example ctrader_demo_order
//! ```

use std::time::Duration;

use modelenv_core::broker_gateway::ctrader::{
    auth, connection::Connection, data, orders, transport::Transport,
};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let client_id = std::env::var("CTRADER_APP_CLIENT_ID")?;
    let client_secret = std::env::var("CTRADER_APP_CLIENT_SECRET")?;
    let access_token = std::env::var("CTRADER_ACCESS_TOKEN")?;
    let symbol = std::env::var("CTRADER_SYMBOL").unwrap_or_else(|_| "USDJPY".into());
    // cTrader API volume is centi-units (base_units × 100); 0.01 lot = 100_000
    // (the USDJPY minimum, verified on demo). See orders::lots_to_volume.
    let volume: i64 = std::env::var("CTRADER_VOLUME")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or_else(|| orders::lots_to_volume(0.01));
    let timeout = Duration::from_secs(20);

    eprintln!("[demo-order] connecting to cTrader demo ...");
    let (r, w) = Transport::connect_env().await?.into_split();
    let (conn, mut events) = Connection::start(r, w);

    auth::app_authenticate(&conn, &client_id, &client_secret, timeout).await?;
    let accounts = data::get_account_list(&conn, &access_token, timeout).await?;
    let demo = accounts
        .iter()
        .find(|a| a.is_live == Some(false))
        .ok_or_else(|| anyhow::anyhow!("no demo account reachable, refusing to trade"))?;
    let account_id = demo.ctid_trader_account_id as i64;
    // Safety: never trade a live account from this validation runner.
    if demo.is_live == Some(true) {
        anyhow::bail!("account {account_id} is LIVE, refusing");
    }
    eprintln!("[demo-order] demo account {account_id}; account auth ...");
    auth::account_authenticate(&conn, &access_token, account_id, timeout).await?;

    let symbol_id = data::get_symbol_id(&conn, account_id, &symbol, timeout).await?;
    eprintln!("[demo-order] {symbol} symbol_id={symbol_id}; placing MARKET BUY volume={volume} ...");

    let opened = orders::submit_market_order(
        &conn,
        &mut events,
        account_id,
        symbol_id,
        orders::Side::Buy,
        volume,
        &format!("demo-validate-{}", std::process::id()),
        timeout,
    )
    .await?;
    println!(
        "[demo-order] FILLED open: position_id={} order_id={} price={} size={} side={}",
        opened.position_id, opened.fill.order_id, opened.fill.price, opened.fill.size, opened.fill.side
    );

    eprintln!("[demo-order] closing position {} ...", opened.position_id);
    let closed = orders::close_position(
        &conn,
        &mut events,
        account_id,
        opened.position_id,
        volume,
        timeout,
    )
    .await?;
    println!(
        "[demo-order] FILLED close: position_id={} order_id={} price={} size={} side={}",
        closed.position_id, closed.fill.order_id, closed.fill.price, closed.fill.size, closed.fill.side
    );

    println!("[demo-order] round-trip OK, opened then closed a demo {symbol} position.");
    Ok(())
}
