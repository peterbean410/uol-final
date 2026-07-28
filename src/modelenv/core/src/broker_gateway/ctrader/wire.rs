//! cTrader Open API wire framing.
//!
//! The cTrader Open API speaks protobuf over a TLS TCP socket. Every frame on
//! the wire is a **4-byte big-endian length** `N` followed by `N` bytes of a
//! serialized [`ProtoMessage`] envelope:
//!
//! ```text
//! ┌──────────────┬───────────────────────────────────────────┐
//! │ len: u32 BE  │ ProtoMessage { payload_type, payload, .. } │
//! └──────────────┴───────────────────────────────────────────┘
//! ```
//!
//! The application message (e.g. `ProtoOANewOrderReq`) is prost-encoded into
//! `ProtoMessage.payload` and identified by `ProtoMessage.payload_type` (one of
//! the [`payload_type`] constants). This module is the pure, network-free
//! framing layer: envelope construction, frame encode/decode, length parsing,
//! and async read/write helpers over any `AsyncRead`/`AsyncWrite`. It places no
//! orders and holds no connection state, so it is fully unit-testable.

use anyhow::{anyhow, Result};
use modelenv_proto::ctrader::ProtoMessage;
use prost::Message;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

/// Defensive ceiling on an inbound frame body length. cTrader application
/// messages are small (orders, ticks, bar batches); 16 MiB is far above any
/// legitimate frame and guards against a corrupt/hostile length prefix turning
/// into an unbounded allocation.
pub const MAX_FRAME_LEN: usize = 16 * 1024 * 1024;

/// cTrader Open API payload-type discriminants carried in
/// `ProtoMessage.payload_type`. Mirrors the `ProtoOAPayloadType` /
/// `ProtoPayloadType` enums in the vendored protos; only the values this client
/// sends or routes on are named here.
pub mod payload_type {
    // Common (OpenApiCommonModelMessages)
    pub const PROTO_MESSAGE: u32 = 5;
    pub const ERROR_RES: u32 = 50;
    pub const HEARTBEAT_EVENT: u32 = 51;

    // cTrader Open API (OpenApiModelMessages), requests/responses/events
    pub const APPLICATION_AUTH_REQ: u32 = 2100;
    pub const APPLICATION_AUTH_RES: u32 = 2101;
    pub const ACCOUNT_AUTH_REQ: u32 = 2102;
    pub const ACCOUNT_AUTH_RES: u32 = 2103;
    pub const VERSION_REQ: u32 = 2104;
    pub const NEW_ORDER_REQ: u32 = 2106;
    pub const SYMBOLS_LIST_REQ: u32 = 2114;
    pub const SYMBOLS_LIST_RES: u32 = 2115;
    pub const GET_TRENDBARS_REQ: u32 = 2137;
    pub const GET_TRENDBARS_RES: u32 = 2138;
    pub const DEAL_LIST_REQ: u32 = 2133;
    pub const DEAL_LIST_RES: u32 = 2134;
    pub const RECONCILE_REQ: u32 = 2124;
    pub const RECONCILE_RES: u32 = 2125;
    pub const EXECUTION_EVENT: u32 = 2126;
    pub const CLOSE_POSITION_REQ: u32 = 2111;
    /// Order-level failure event (e.g. market closed, insufficient funds). cTrader
    /// reports a bad order via THIS, not an `ORDER_REJECTED` execution event.
    pub const ORDER_ERROR_EVENT: u32 = 2132;
    pub const OA_ERROR_RES: u32 = 2142;
    pub const GET_ACCOUNTS_BY_ACCESS_TOKEN_REQ: u32 = 2149;
    pub const GET_ACCOUNTS_BY_ACCESS_TOKEN_RES: u32 = 2150;
    // Spot (streaming bid/ask) subscription + unsolicited spot events.
    pub const SUBSCRIBE_SPOTS_REQ: u32 = 2127;
    pub const SUBSCRIBE_SPOTS_RES: u32 = 2128;
    pub const SPOT_EVENT: u32 = 2131;
}

/// Build a [`ProtoMessage`] envelope around an already-encoded application
/// payload. `client_msg_id` correlates a response to its request (echoed back
/// by the server); pass `None` for fire-and-forget sends.
pub fn envelope(
    payload_type: u32,
    payload: Vec<u8>,
    client_msg_id: Option<String>,
) -> ProtoMessage {
    ProtoMessage {
        payload_type,
        payload: Some(payload),
        client_msg_id,
    }
}

/// Encode a [`ProtoMessage`] into a complete length-prefixed wire frame
/// (`4-byte BE length` + serialized body).
pub fn encode_frame(msg: &ProtoMessage) -> Vec<u8> {
    let body = msg.encode_to_vec();
    let mut out = Vec::with_capacity(4 + body.len());
    out.extend_from_slice(&(body.len() as u32).to_be_bytes());
    out.extend_from_slice(&body);
    out
}

/// Validate + parse the 4-byte big-endian length prefix.
pub fn parse_len(prefix: [u8; 4]) -> Result<usize> {
    let n = u32::from_be_bytes(prefix) as usize;
    if n > MAX_FRAME_LEN {
        return Err(anyhow!(
            "cTrader frame length {n} exceeds MAX_FRAME_LEN {MAX_FRAME_LEN}"
        ));
    }
    Ok(n)
}

/// Decode a [`ProtoMessage`] from a complete frame body (the bytes after the
/// length prefix).
pub fn decode_body(body: &[u8]) -> Result<ProtoMessage> {
    ProtoMessage::decode(body).map_err(|e| anyhow!("cTrader ProtoMessage decode failed: {e}"))
}

/// Write one framed [`ProtoMessage`] to an async writer (length prefix + body),
/// flushing so the server sees it immediately.
pub async fn write_frame<W: AsyncWriteExt + Unpin>(w: &mut W, msg: &ProtoMessage) -> Result<()> {
    let frame = encode_frame(msg);
    w.write_all(&frame).await?;
    w.flush().await?;
    Ok(())
}

/// Read exactly one framed [`ProtoMessage`] from an async reader. Reads the
/// 4-byte length, then the exact body, then decodes, correctly reassembling a
/// frame that arrives split across multiple TCP reads.
pub async fn read_frame<R: AsyncReadExt + Unpin>(r: &mut R) -> Result<ProtoMessage> {
    let mut len_buf = [0u8; 4];
    r.read_exact(&mut len_buf).await?;
    let n = parse_len(len_buf)?;
    let mut body = vec![0u8; n];
    r.read_exact(&mut body).await?;
    decode_body(&body)
}

#[cfg(test)]
mod tests {
    use super::*;
    use modelenv_proto::ctrader::ProtoOaNewOrderReq;
    use tokio::io::AsyncWriteExt;

    fn sample_envelope() -> ProtoMessage {
        envelope(
            payload_type::NEW_ORDER_REQ,
            vec![1, 2, 3, 4, 5],
            Some("corr-123".to_string()),
        )
    }

    #[test]
    fn envelope_round_trips_through_a_frame() {
        let msg = sample_envelope();
        let frame = encode_frame(&msg);
        // first 4 bytes are the BE length of the remaining body
        let (len_bytes, body) = frame.split_at(4);
        let n = parse_len(len_bytes.try_into().unwrap()).unwrap();
        assert_eq!(n, body.len());
        let decoded = decode_body(body).unwrap();
        assert_eq!(decoded.payload_type, payload_type::NEW_ORDER_REQ);
        assert_eq!(decoded.payload.as_deref(), Some(&[1, 2, 3, 4, 5][..]));
        assert_eq!(decoded.client_msg_id.as_deref(), Some("corr-123"));
    }

    #[test]
    fn inner_application_message_nests_and_unnests() {
        // Encode a real ProtoOANewOrderReq into the envelope payload and recover it.
        let order = ProtoOaNewOrderReq {
            ctid_trader_account_id: 42,
            symbol_id: 4, // USDJPY on most cTrader brokers
            order_type: 1, // MARKET
            trade_side: 1, // BUY
            volume: 100,   // 1.00 lots in cTrader cents-of-lot units
            ..Default::default()
        };
        let env = envelope(
            payload_type::NEW_ORDER_REQ,
            order.encode_to_vec(),
            Some("o-1".into()),
        );
        let frame = encode_frame(&env);
        let decoded_env = decode_body(&frame[4..]).unwrap();
        let decoded_order =
            ProtoOaNewOrderReq::decode(decoded_env.payload.as_deref().unwrap()).unwrap();
        assert_eq!(decoded_order.ctid_trader_account_id, 42);
        assert_eq!(decoded_order.symbol_id, 4);
        assert_eq!(decoded_order.trade_side, 1);
        assert_eq!(decoded_order.volume, 100);
    }

    #[test]
    fn parse_len_rejects_oversized_prefix() {
        let huge = ((MAX_FRAME_LEN as u32) + 1).to_be_bytes();
        assert!(parse_len(huge).is_err());
        let ok = (1234u32).to_be_bytes();
        assert_eq!(parse_len(ok).unwrap(), 1234);
    }

    #[tokio::test]
    async fn async_write_then_read_round_trips() {
        let (mut a, mut b) = tokio::io::duplex(1024);
        let msg = sample_envelope();
        write_frame(&mut a, &msg).await.unwrap();
        let got = read_frame(&mut b).await.unwrap();
        assert_eq!(got.payload_type, msg.payload_type);
        assert_eq!(got.payload, msg.payload);
        assert_eq!(got.client_msg_id, msg.client_msg_id);
    }

    #[tokio::test]
    async fn read_frame_reassembles_a_split_frame() {
        // Write the length prefix and body in separate chunks with a yield in
        // between, so read_frame must reassemble across reads.
        let frame = encode_frame(&sample_envelope());
        let (mut a, mut b) = tokio::io::duplex(1024);
        let writer = tokio::spawn(async move {
            a.write_all(&frame[..2]).await.unwrap();
            a.flush().await.unwrap();
            tokio::task::yield_now().await;
            a.write_all(&frame[2..6]).await.unwrap();
            a.flush().await.unwrap();
            tokio::task::yield_now().await;
            a.write_all(&frame[6..]).await.unwrap();
            a.flush().await.unwrap();
        });
        let got = read_frame(&mut b).await.unwrap();
        writer.await.unwrap();
        assert_eq!(got.payload_type, payload_type::NEW_ORDER_REQ);
        assert_eq!(got.client_msg_id.as_deref(), Some("corr-123"));
    }
}
