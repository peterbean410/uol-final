// Broker connection module for Production Mode
use anyhow::Result;
use log::{error, info};
use std::sync::Arc;

use modelenv_core::broker_gateway::BrokerGateway;

/// Create a broker gateway connection based on configuration
///
/// This function creates the appropriate broker gateway implementation
/// based on the configured broker gateway type.
///
/// # Arguments
/// * `broker_gateway` - The broker gateway type (e.g., "ctrader", "metatrader", "ib")
/// * `broker_addr` - Optional broker gateway address (host:port)
/// * `ctrader_app_client_id` - Optional cTrader Open API app client ID
/// * `ctrader_app_client_secret` - Optional cTrader Open API app client secret
/// * `ctrader_access_token` - Optional cTrader Open API access token
/// * `ctrader_refresh_token` - Optional cTrader Open API refresh token
/// * `ctrader_account` - Optional cTrader trader account ID
/// * `symbol` - Trading symbol
///
/// # Returns
/// A boxed BrokerGateway implementation
pub async fn create_broker_gateway(
    broker_gateway: &str,
    broker_addr: Option<&str>,
    ctrader_app_client_id: Option<&str>,
    ctrader_app_client_secret: Option<&str>,
    ctrader_access_token: Option<&str>,
    ctrader_refresh_token: Option<&str>,
    ctrader_account: Option<&str>,
    symbol: &str,
) -> Result<Arc<dyn BrokerGateway + Send + Sync>> {
    info!(
        "Creating broker gateway: type={}, symbol={}",
        broker_gateway, symbol
    );

    match broker_gateway.to_lowercase().as_str() {
        "ctrader" => {
            let app_client_id = ctrader_app_client_id.map(|s| s.to_string());
            let app_client_secret = ctrader_app_client_secret.map(|s| s.to_string());
            let access_token = ctrader_access_token.map(|s| s.to_string());
            let refresh_token = ctrader_refresh_token.map(|s| s.to_string());
            let account = ctrader_account.map(|s| s.to_string());

            let gateway = modelenv_core::broker_gateway::create_broker_gateway_instance(
                broker_gateway,
                app_client_id,
                app_client_secret,
                access_token,
                refresh_token,
                account,
                symbol,
            )?;
            info!("cTrader broker gateway created for symbol {}", symbol);
            Ok(Arc::from(gateway))
        }
        "metatrader" => {
            let _broker_addr = broker_addr.ok_or_else(|| {
                anyhow::anyhow!("Broker gateway type specified but address not configured")
            })?;
            // TODO: Implement MetaTrader API gateway
            let err = anyhow::anyhow!(
                "MetaTrader API gateway not yet implemented. \
                Please configure a different broker gateway or implement the MetaTrader API integration."
            );
            error!("Failed to create MetaTrader broker gateway: {}", err);
            Err(err)
        }
        "ib" | "interactive_brokers" => {
            let _broker_addr = broker_addr.ok_or_else(|| {
                anyhow::anyhow!("Broker gateway type specified but address not configured")
            })?;
            // TODO: Implement Interactive Brokers API gateway
            let err = anyhow::anyhow!(
                "Interactive Brokers API gateway not yet implemented. \
                Please configure a different broker gateway or implement the IB API integration."
            );
            error!("Failed to create Interactive Brokers gateway: {}", err);
            Err(err)
        }
        _ => {
            let err = anyhow::anyhow!(
                "Unknown broker gateway type '{}'. Supported types: ctrader, metatrader, ib",
                broker_gateway
            );
            error!("Failed to create broker gateway: {}", err);
            Err(err)
        }
    }
}

/// Try to create a broker gateway connection with error handling
///
/// This function attempts to create a broker gateway connection and returns
/// an appropriate result that can be used by the main application.
///
/// # Arguments
/// * `broker_gateway` - Optional broker gateway type
/// * `broker_addr` - Optional broker gateway address
/// * `ctrader_app_client_id` - Optional cTrader Open API app client ID
/// * `ctrader_app_client_secret` - Optional cTrader Open API app client secret
/// * `ctrader_access_token` - Optional cTrader Open API access token
/// * `ctrader_refresh_token` - Optional cTrader Open API refresh token
/// * `ctrader_account` - Optional cTrader trader account ID
/// * `symbol` - Trading symbol
///
/// # Returns
/// * `Ok(Some(gateway))` - Successfully created broker gateway
/// * `Ok(None)` - Broker gateway not configured (not an error)
/// * `Err(e)` - Failed to create broker gateway
pub async fn try_create_broker_gateway(
    broker_gateway: Option<&str>,
    broker_addr: Option<&str>,
    ctrader_app_client_id: Option<&str>,
    ctrader_app_client_secret: Option<&str>,
    ctrader_access_token: Option<&str>,
    ctrader_refresh_token: Option<&str>,
    ctrader_account: Option<&str>,
    symbol: &str,
) -> Result<Option<Arc<dyn BrokerGateway + Send + Sync>>> {
    match broker_gateway {
        Some(gateway) => {
            let gateway = create_broker_gateway(
                gateway,
                broker_addr,
                ctrader_app_client_id,
                ctrader_app_client_secret,
                ctrader_access_token,
                ctrader_refresh_token,
                ctrader_account,
                symbol,
            )
            .await?;
            Ok(Some(gateway))
        }
        None if broker_addr.is_none() => {
            // Broker gateway not configured - this is fine for Training Mode
            info!("Broker gateway not configured; starting without external broker");
            Ok(None)
        }
        None => {
            let err = anyhow::anyhow!("Broker gateway address specified but type not configured");
            error!("Invalid broker gateway configuration: {}", err);
            Err(err)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn try_create_ctrader_gateway_without_broker_addr() {
        let gateway = try_create_broker_gateway(
            Some("ctrader"),
            None,
            Some("app-client-id"),
            Some("app-client-secret"),
            Some("access-token"),
            Some("refresh-token"),
            Some("account"),
            "USDJPY",
        )
        .await
        .unwrap();

        assert!(gateway.is_some());
    }

    #[tokio::test]
    async fn non_ctrader_gateway_requires_broker_addr() {
        let result = try_create_broker_gateway(
            Some("metatrader"),
            None,
            None,
            None,
            None,
            None,
            None,
            "USDJPY",
        )
        .await;

        match result {
            Ok(_) => panic!("expected broker gateway creation to fail without broker address"),
            Err(err) => assert!(err.to_string().contains("address not configured")),
        }
    }
}
