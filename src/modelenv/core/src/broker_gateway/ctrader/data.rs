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
    ProtoOaCtidTraderAccount, ProtoOaErrorRes, ProtoOaGetAccountListByAccessTokenReq,
    ProtoOaGetAccountListByAccessTokenRes, ProtoOaSymbolsListReq, ProtoOaSymbolsListRes,
};
use prost::Message;

use super::connection::Connection;
use super::wire::payload_type;

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::broker_gateway::ctrader::wire;
    use modelenv_proto::ctrader::ProtoOaLightSymbol;
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
}
