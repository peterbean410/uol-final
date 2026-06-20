//! cTrader Open API connection lifecycle: application/account authentication
//! and heartbeats, over a [`super::connection::Connection`].
//!
//! Handshake order after the TLS connect:
//! 1. `ProtoOAApplicationAuthReq` (app `client_id` + `client_secret`) → `…Res`
//! 2. `ProtoOAAccountAuthReq` (`ctidTraderAccountId` + OAuth `access_token`) → `…Res`
//!
//! Then a heartbeat must be sent periodically (cTrader drops idle sockets after
//! ~30 s). **OAuth token refresh is out of scope here**: the access token is
//! refreshed externally by the Airflow `refresh_token` DAG into the
//! `ctrader-secrets` secret; modelenv consumes whatever current token it is
//! given. Reconnect is driven at the client level by re-running TLS connect +
//! [`authenticate`] with the latest token.

use std::time::Duration;

use anyhow::{anyhow, Result};
use modelenv_proto::ctrader::{
    ProtoOaAccountAuthReq, ProtoOaApplicationAuthReq, ProtoOaErrorRes,
};
use modelenv_proto::ctrader::ProtoMessage;
use prost::Message;
use tokio::task::JoinHandle;

use super::connection::Connection;
use super::wire::payload_type;

/// cTrader keeps a socket alive only if it sees traffic; send a heartbeat at
/// least this often. 10 s is the SDK-recommended cadence (well inside the
/// ~30 s idle-drop window).
pub const HEARTBEAT_INTERVAL: Duration = Duration::from_secs(10);

/// Credentials for the two-stage cTrader auth handshake.
#[derive(Clone)]
pub struct Credentials {
    /// Open API application client id.
    pub client_id: String,
    /// Open API application client secret.
    pub client_secret: String,
    /// OAuth access token for the trading account (refreshed externally).
    pub access_token: String,
    /// `ctidTraderAccountId`; the numeric trading account id.
    pub account_id: i64,
}

/// Run the application-auth then account-auth handshake over `conn`. Returns
/// `Ok(())` once both are accepted; surfaces the broker error code if cTrader
/// rejects either stage.
pub async fn authenticate(
    conn: &Connection,
    creds: &Credentials,
    timeout: Duration,
) -> Result<()> {
    let app_req = ProtoOaApplicationAuthReq {
        payload_type: Some(payload_type::APPLICATION_AUTH_REQ as i32),
        client_id: creds.client_id.clone(),
        client_secret: creds.client_secret.clone(),
    };
    let resp = conn
        .send_request(
            payload_type::APPLICATION_AUTH_REQ,
            app_req.encode_to_vec(),
            timeout,
        )
        .await?;
    expect_payload_type(&resp, payload_type::APPLICATION_AUTH_RES, "application auth")?;

    let acct_req = ProtoOaAccountAuthReq {
        payload_type: Some(payload_type::ACCOUNT_AUTH_REQ as i32),
        ctid_trader_account_id: creds.account_id,
        access_token: creds.access_token.clone(),
    };
    let resp = conn
        .send_request(
            payload_type::ACCOUNT_AUTH_REQ,
            acct_req.encode_to_vec(),
            timeout,
        )
        .await?;
    expect_payload_type(&resp, payload_type::ACCOUNT_AUTH_RES, "account auth")?;
    Ok(())
}

/// Verify a response is the expected type; otherwise map an error response to a
/// descriptive broker error, or report the unexpected type.
fn expect_payload_type(resp: &ProtoMessage, want: u32, ctx: &str) -> Result<()> {
    if resp.payload_type == want {
        return Ok(());
    }
    if resp.payload_type == payload_type::ERROR_RES
        || resp.payload_type == payload_type::OA_ERROR_RES
    {
        if let Ok(err) = ProtoOaErrorRes::decode(resp.payload.as_deref().unwrap_or_default()) {
            return Err(anyhow!(
                "cTrader {ctx} rejected: {} ({})",
                err.error_code,
                err.description.unwrap_or_default()
            ));
        }
    }
    Err(anyhow!(
        "cTrader {ctx}: unexpected payload_type {} (expected {want})",
        resp.payload_type
    ))
}

/// Spawn a background task that sends a heartbeat every `interval` until the
/// connection write fails (socket closed), keeping the cTrader session alive.
/// `ProtoHeartbeatEvent` is an empty message, so an empty payload suffices.
pub fn spawn_heartbeat(conn: Connection, interval: Duration) -> JoinHandle<()> {
    tokio::spawn(async move {
        let mut tick = tokio::time::interval(interval);
        // Skip the immediate first tick; we only need periodic keep-alive.
        tick.tick().await;
        loop {
            tick.tick().await;
            if conn
                .send_event(payload_type::HEARTBEAT_EVENT, Vec::new())
                .await
                .is_err()
            {
                break;
            }
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::broker_gateway::ctrader::wire;
    use tokio::io::{AsyncRead, AsyncWrite};

    fn creds() -> Credentials {
        Credentials {
            client_id: "app-id".into(),
            client_secret: "app-secret".into(),
            access_token: "tok".into(),
            account_id: 12345,
        }
    }

    /// Mock cTrader that accepts both auth stages (echoes the matching *_RES).
    async fn accepting_server<S>(mut s: S)
    where
        S: AsyncRead + AsyncWrite + Unpin + Send + 'static,
    {
        loop {
            match wire::read_frame(&mut s).await {
                Ok(req) => {
                    let res_type = req.payload_type + 1; // REQ -> RES (2100->2101, 2102->2103)
                    let res = wire::envelope(res_type, vec![], req.client_msg_id.clone());
                    if wire::write_frame(&mut s, &res).await.is_err() {
                        break;
                    }
                }
                Err(_) => break,
            }
        }
    }

    /// Mock cTrader that rejects application auth with an ErrorRes.
    async fn rejecting_server<S>(mut s: S)
    where
        S: AsyncRead + AsyncWrite + Unpin + Send + 'static,
    {
        if let Ok(req) = wire::read_frame(&mut s).await {
            let err = ProtoOaErrorRes {
                payload_type: Some(payload_type::OA_ERROR_RES as i32),
                error_code: "CH_CLIENT_AUTH_FAILURE".into(),
                description: Some("bad app credentials".into()),
                ..Default::default()
            };
            let res = wire::envelope(
                payload_type::OA_ERROR_RES,
                err.encode_to_vec(),
                req.client_msg_id.clone(),
            );
            let _ = wire::write_frame(&mut s, &res).await;
        }
    }

    #[tokio::test]
    async fn authenticate_succeeds_when_both_stages_accepted() {
        let (client_io, server_io) = tokio::io::duplex(4096);
        tokio::spawn(accepting_server(server_io));
        let (cr, cw) = tokio::io::split(client_io);
        let (conn, _events) = Connection::start(cr, cw);
        authenticate(&conn, &creds(), Duration::from_secs(5))
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn authenticate_surfaces_broker_rejection() {
        let (client_io, server_io) = tokio::io::duplex(4096);
        tokio::spawn(rejecting_server(server_io));
        let (cr, cw) = tokio::io::split(client_io);
        let (conn, _events) = Connection::start(cr, cw);
        let err = authenticate(&conn, &creds(), Duration::from_secs(5))
            .await
            .unwrap_err()
            .to_string();
        assert!(err.contains("CH_CLIENT_AUTH_FAILURE"), "got: {err}");
        assert!(err.contains("application auth"), "got: {err}");
    }

    #[tokio::test]
    async fn heartbeat_sends_periodically_until_socket_closes() {
        let (client_io, mut server_io) = tokio::io::duplex(4096);
        let (cr, cw) = tokio::io::split(client_io);
        let (conn, _events) = Connection::start(cr, cw);
        let handle = spawn_heartbeat(conn, Duration::from_millis(20));

        // Read two heartbeats off the server end.
        for _ in 0..2 {
            let hb = wire::read_frame(&mut server_io).await.unwrap();
            assert_eq!(hb.payload_type, payload_type::HEARTBEAT_EVENT);
        }
        // Closing the socket ends the heartbeat task.
        drop(server_io);
        let _ = tokio::time::timeout(Duration::from_secs(2), handle).await;
    }
}
