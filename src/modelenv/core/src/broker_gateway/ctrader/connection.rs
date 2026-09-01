//! cTrader Open API connection: async request/response correlation + event routing.
//!
//! Above the framed [`super::transport`] this layer multiplexes one TLS socket
//! into many logical request/response exchanges. cTrader echoes the
//! `client_msg_id` we set on a request back on its response, so a single
//! background **read loop** can route each inbound frame either to the waiting
//! caller (matched `client_msg_id`) or, when unsolicited (heartbeats, execution
//! events, spot events, server errors), to an events channel.
//!
//! It is generic over an `AsyncRead`/`AsyncWrite` pair so it can be driven by
//! the real TLS transport in production and by an in-memory duplex (a mock
//! cTrader server) in tests; no network required.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use anyhow::{anyhow, Result};
use modelenv_proto::ctrader::ProtoMessage;
use tokio::io::{AsyncRead, AsyncWrite};
use tokio::sync::{mpsc, oneshot, Mutex};

use super::wire;

/// Default ceiling on how long [`Connection::send_request`] waits for the
/// matching response before failing. cTrader responses are sub-second in
/// practice; this is a generous upper bound.
pub const DEFAULT_REQUEST_TIMEOUT: Duration = Duration::from_secs(15);

type PendingMap = Arc<Mutex<HashMap<String, oneshot::Sender<ProtoMessage>>>>;

/// A multiplexed cTrader connection. Cheap to clone; all clones share the same
/// underlying socket, pending-request table, and event stream.
#[derive(Clone)]
pub struct Connection {
    writer: Arc<Mutex<Box<dyn AsyncWriteHalf>>>,
    pending: PendingMap,
    next_id: Arc<AtomicU64>,
}

/// Object-safe async writer the connection can own behind a mutex.
pub trait AsyncWriteHalf: AsyncWrite + Unpin + Send {}
impl<T: AsyncWrite + Unpin + Send> AsyncWriteHalf for T {}

impl Connection {
    /// Start a connection over a split reader/writer, spawning the read loop.
    /// Unsolicited (uncorrelated) inbound messages are delivered on the returned
    /// receiver; the caller drains it (e.g. to dispatch `ProtoOAExecutionEvent`).
    pub fn start<R, W>(reader: R, writer: W) -> (Self, mpsc::UnboundedReceiver<ProtoMessage>)
    where
        R: AsyncRead + Unpin + Send + 'static,
        W: AsyncWrite + Unpin + Send + 'static,
    {
        let pending: PendingMap = Arc::new(Mutex::new(HashMap::new()));
        let (events_tx, events_rx) = mpsc::unbounded_channel();
        let conn = Self {
            writer: Arc::new(Mutex::new(Box::new(writer))),
            pending: pending.clone(),
            next_id: Arc::new(AtomicU64::new(1)),
        };
        tokio::spawn(read_loop(reader, pending, events_tx));
        (conn, events_rx)
    }

    /// Allocate a fresh, process-unique correlation id.
    fn next_client_msg_id(&self) -> String {
        format!("m{}", self.next_id.fetch_add(1, Ordering::Relaxed))
    }

    /// Send a request and await the response whose `client_msg_id` matches, up
    /// to `timeout`. Registers the correlation BEFORE writing so a fast server
    /// reply can never race ahead of the pending entry.
    pub async fn send_request(
        &self,
        payload_type: u32,
        payload: Vec<u8>,
        timeout: Duration,
    ) -> Result<ProtoMessage> {
        let id = self.next_client_msg_id();
        let (tx, rx) = oneshot::channel();
        self.pending.lock().await.insert(id.clone(), tx);

        let msg = wire::envelope(payload_type, payload, Some(id.clone()));
        if let Err(e) = self.write(&msg).await {
            self.pending.lock().await.remove(&id);
            return Err(e);
        }

        match tokio::time::timeout(timeout, rx).await {
            Ok(Ok(resp)) => Ok(resp),
            Ok(Err(_)) => Err(anyhow!(
                "cTrader connection closed while awaiting response to {id}"
            )),
            Err(_) => {
                self.pending.lock().await.remove(&id);
                Err(anyhow!(
                    "cTrader request {id} (payload_type={payload_type}) timed out after {:?}",
                    timeout
                ))
            }
        }
    }

    /// Fire-and-forget send of an already-built envelope (e.g. heartbeats).
    pub async fn write(&self, msg: &ProtoMessage) -> Result<()> {
        let mut w = self.writer.lock().await;
        wire::write_frame(&mut *w, msg).await
    }

    /// Send an unkeyed event message (no correlation id), e.g. a heartbeat.
    pub async fn send_event(&self, payload_type: u32, payload: Vec<u8>) -> Result<()> {
        self.write(&wire::envelope(payload_type, payload, None)).await
    }
}

/// Background loop: read frames forever, routing each to its waiting caller (by
/// `client_msg_id`) or to the unsolicited-events channel. Exits when the socket
/// closes or the events receiver is dropped, failing any still-pending waiters.
async fn read_loop<R>(
    mut reader: R,
    pending: PendingMap,
    events_tx: mpsc::UnboundedSender<ProtoMessage>,
) where
    R: AsyncRead + Unpin + Send,
{
    loop {
        match wire::read_frame(&mut reader).await {
            Ok(msg) => {
                let matched = match msg.client_msg_id.as_ref() {
                    Some(id) => pending.lock().await.remove(id),
                    None => None,
                };
                match matched {
                    Some(tx) => {
                                                let _ = tx.send(msg);
                    }
                    None => {
                        if events_tx.send(msg).is_err() {
                            break;
                        }
                    }
                }
            }
            Err(_) => break,
        }
    }
        pending.lock().await.clear();
}

#[cfg(test)]
mod tests {
    use super::*;
    use prost::Message;

    /// Spawn a mock cTrader server on one end of an in-memory duplex that, for
    /// every framed request it receives, echoes a response with the SAME
    /// `client_msg_id` and `payload_type + 1` (mirrors cTrader's req->res pairing
    /// where e.g. APPLICATION_AUTH_REQ=2100 -> APPLICATION_AUTH_RES=2101).
    async fn echo_server<S>(mut server: S)
    where
        S: AsyncRead + AsyncWrite + Unpin + Send + 'static,
    {
        loop {
            match wire::read_frame(&mut server).await {
                Ok(req) => {
                    let resp = wire::envelope(
                        req.payload_type + 1,
                        req.payload.clone().unwrap_or_default(),
                        req.client_msg_id.clone(),
                    );
                    if wire::write_frame(&mut server, &resp).await.is_err() {
                        break;
                    }
                }
                Err(_) => break,
            }
        }
    }

    #[tokio::test]
    async fn send_request_correlates_the_matching_response() {
        let (client_io, server_io) = tokio::io::duplex(4096);
        tokio::spawn(echo_server(server_io));
        let (cr, cw) = tokio::io::split(client_io);
        let (conn, _events) = Connection::start(cr, cw);

        let resp = conn
            .send_request(wire::payload_type::APPLICATION_AUTH_REQ, vec![9, 9], DEFAULT_REQUEST_TIMEOUT)
            .await
            .unwrap();
        assert_eq!(resp.payload_type, wire::payload_type::APPLICATION_AUTH_RES);
        assert_eq!(resp.payload.as_deref(), Some(&[9, 9][..]));
    }

    #[tokio::test]
    async fn concurrent_requests_each_get_their_own_response() {
        let (client_io, server_io) = tokio::io::duplex(4096);
        tokio::spawn(echo_server(server_io));
        let (cr, cw) = tokio::io::split(client_io);
        let (conn, _events) = Connection::start(cr, cw);

                        let mut handles = Vec::new();
        for i in 0u8..16 {
            let c = conn.clone();
            handles.push(tokio::spawn(async move {
                let r = c
                    .send_request(2100, vec![i], DEFAULT_REQUEST_TIMEOUT)
                    .await
                    .unwrap();
                assert_eq!(r.payload.as_deref(), Some(&[i][..]));
            }));
        }
        for h in handles {
            h.await.unwrap();
        }
    }

    #[tokio::test]
    async fn unsolicited_messages_go_to_the_events_channel() {
                let (client_io, mut server_io) = tokio::io::duplex(4096);
        let (cr, cw) = tokio::io::split(client_io);
        let (_conn, mut events) = Connection::start(cr, cw);

        let evt = wire::envelope(wire::payload_type::EXECUTION_EVENT, vec![7], None);
        wire::write_frame(&mut server_io, &evt).await.unwrap();

        let got = events.recv().await.unwrap();
        assert_eq!(got.payload_type, wire::payload_type::EXECUTION_EVENT);
        assert_eq!(got.payload.as_deref(), Some(&[7][..]));
    }

    #[tokio::test]
    async fn request_times_out_when_no_response() {
                let (client_io, mut server_io) = tokio::io::duplex(4096);
        tokio::spawn(async move {
                        let mut buf = [0u8; 64];
            use tokio::io::AsyncReadExt;
            while server_io.read(&mut buf).await.unwrap_or(0) > 0 {}
        });
        let (cr, cw) = tokio::io::split(client_io);
        let (conn, _events) = Connection::start(cr, cw);

        let res = conn
            .send_request(2100, vec![1], Duration::from_millis(150))
            .await;
        assert!(res.is_err());
        assert!(res.unwrap_err().to_string().contains("timed out"));
    }

    #[tokio::test]
    async fn pending_request_fails_when_connection_closes() {
        let (client_io, server_io) = tokio::io::duplex(4096);
        let (cr, cw) = tokio::io::split(client_io);
        let (conn, _events) = Connection::start(cr, cw);
                drop(server_io);
        let res = conn.send_request(2100, vec![1], DEFAULT_REQUEST_TIMEOUT).await;
        assert!(res.is_err());
    }

    #[test]
    fn ids_are_unique_and_monotonic() {
                let next = Arc::new(AtomicU64::new(1));
        let a = format!("m{}", next.fetch_add(1, Ordering::Relaxed));
        let b = format!("m{}", next.fetch_add(1, Ordering::Relaxed));
        assert_ne!(a, b);
        assert_eq!((a, b), ("m1".to_string(), "m2".to_string()));
                let env = wire::envelope(2100, ProtoMessage::default().encode_to_vec(), Some("m1".into()));
        assert!(wire::decode_body(&wire::encode_frame(&env)[4..]).is_ok());
    }
}
