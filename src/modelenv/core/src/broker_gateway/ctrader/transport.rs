//! cTrader Open API transport: protobuf [`ProtoMessage`]s over a TLS TCP socket.
//!
//! Thin layer over [`super::wire`]: it owns the TLS stream and exposes
//! [`Transport::send`] / [`Transport::recv`], which delegate to the unit-tested
//! framing codec. Connection lifecycle (auth, heartbeat, reconnect) and
//! request/response correlation live above this in the client.
//!
//! cTrader exposes the trading endpoint on port **5035** (protobuf) for both
//! demo and live; the host selects the environment:
//! - demo: `demo.ctraderapi.com`

use std::sync::Arc;

use anyhow::{anyhow, Context, Result};
use modelenv_proto::ctrader::ProtoMessage;
use tokio::net::TcpStream;
use tokio_rustls::client::TlsStream;
use tokio_rustls::rustls::pki_types::ServerName;
use tokio_rustls::rustls::{ClientConfig, RootCertStore};
use tokio_rustls::TlsConnector;

use super::wire;

/// cTrader demo trading host (protobuf API).
pub const DEMO_HOST: &str = "demo.ctraderapi.com";
/// cTrader live trading host (protobuf API).
/// cTrader protobuf trading port (demo and live).
pub const PORT: u16 = 5035;

/// An authenticated-transport-agnostic TLS framed connection to cTrader.
pub struct Transport {
    stream: TlsStream<TcpStream>,
    host: String,
    port: u16,
}

impl Transport {
    /// Open a TLS connection to `host:port` and prepare it for framed
    /// [`ProtoMessage`] exchange. Verifies the server certificate against the
    /// Mozilla root store (webpki-roots); no application auth is performed here.
    pub async fn connect(host: &str, port: u16) -> Result<Self> {
        let mut roots = RootCertStore::empty();
        roots.extend(webpki_roots::TLS_SERVER_ROOTS.iter().cloned());
        let config = ClientConfig::builder()
            .with_root_certificates(roots)
            .with_no_client_auth();
        let connector = TlsConnector::from(Arc::new(config));

        let server_name = ServerName::try_from(host.to_string())
            .with_context(|| format!("invalid cTrader host name {host:?}"))?;

        let tcp = TcpStream::connect((host, port))
            .await
            .with_context(|| format!("TCP connect to cTrader {host}:{port} failed"))?;
        // cTrader sends frequent small frames (heartbeats, ticks); disable
        // Nagle so requests/heartbeats are not delayed.
        tcp.set_nodelay(true).ok();

        let stream = connector
            .connect(server_name, tcp)
            .await
            .with_context(|| format!("TLS handshake with cTrader {host}:{port} failed"))?;

        Ok(Self {
            stream,
            host: host.to_string(),
            port,
        })
    }

    /// Connect to the demo trading endpoint.
    pub async fn connect_env() -> Result<Self> {
        Self::connect(DEMO_HOST, PORT).await
    }

    /// Send one framed [`ProtoMessage`].
    pub async fn send(&mut self, msg: &ProtoMessage) -> Result<()> {
        wire::write_frame(&mut self.stream, msg)
            .await
            .map_err(|e| anyhow!("cTrader send to {}:{} failed: {e}", self.host, self.port))
    }

    /// Receive one framed [`ProtoMessage`] (blocks until a full frame arrives).
    pub async fn recv(&mut self) -> Result<ProtoMessage> {
        wire::read_frame(&mut self.stream)
            .await
            .map_err(|e| anyhow!("cTrader recv from {}:{} failed: {e}", self.host, self.port))
    }

    /// The connected endpoint, for logging.
    pub fn endpoint(&self) -> String {
        format!("{}:{}", self.host, self.port)
    }

    /// Consume the transport into independent read/write halves, suitable for
    /// driving a [`super::connection::Connection`] (which reads in a background
    /// task while callers write).
    pub fn into_split(
        self,
    ) -> (
        tokio::io::ReadHalf<TlsStream<TcpStream>>,
        tokio::io::WriteHalf<TlsStream<TcpStream>>,
    ) {
        tokio::io::split(self.stream)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn endpoint_constants_are_the_protobuf_port() {
        assert_eq!(PORT, 5035);
        assert_eq!(DEMO_HOST, "demo.ctraderapi.com");
    }

    // Building the rustls client config (root store + connector) must not panic;
    // this exercises everything up to the socket, with no network dependency.
    // A refused connection on loopback returns immediately (vs. a non-routable
    // address which would hang on SYN retries), so the test stays fast.
    #[tokio::test]
    async fn connect_to_refused_port_errors_cleanly() {
        // Bind then immediately drop a listener to obtain a definitely-closed
        // local port, then connect to it, instant ECONNREFUSED.
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        drop(listener);
        let res = Transport::connect("127.0.0.1", port).await;
        assert!(res.is_err());
    }
}
