// FX RL Model Environment Server
mod broker;
mod server;

use anyhow::Result;
use env_logger::Env;
use log::{error, info};
use modelenv_core::{
    config::{Config, Mode},
    environment::Environment,
};

use crate::server::EnvironmentService;
use modelenv_proto::environment_server::EnvironmentServer;
use tonic::transport::Server;

fn init_logging() {
    let _ = env_logger::Builder::from_env(Env::default().default_filter_or("info"))
        .format_timestamp_secs()
        .try_init();
}

fn build_environment(config: &Config) -> Environment {
    let mut environment = Environment::new(
        config.mode.clone(),
        config.symbol.clone(),
        config.s3_prefix.clone(),
    )
    .with_local_cache_dir(config.local_cache_dir.clone())
    .with_reward_lambda(config.reward_lambda)
    .with_reward_action_penalty(config.reward_action_penalty)
    .with_reward_holding_penalty(config.reward_holding_penalty)
    .with_disable_hedging(config.disable_hedging);

    if let Some(price_snapshot_ts) = config.price_snapshot_ts {
        environment = environment.with_price_snapshot_ts(price_snapshot_ts);
    }

    environment
}

#[tokio::main]
async fn main() -> Result<()> {
    init_logging();

    // Load configuration from command-line arguments and environment variables
    let config = Config::load().map_err(|err| {
        error!("Failed to load configuration: {}", err);
        err
    })?;

    // Log configuration at startup
    config.log();

    // Create the environment
    let mut environment = build_environment(&config);
    if config.mode == Mode::Training {
        info!("Preloading training market data on startup...");
        environment.preload_training_data().await.map_err(|err| {
            error!("Failed to preload training market data: {}", err);
            err
        })?;
        info!("Training market data preload complete");
    }

    // Configure broker gateway if in Production Mode
    if config.mode == Mode::Live {
        info!("Connecting to broker gateway...");

        // Try to create broker gateway connection
        let broker_gateway = broker::try_create_broker_gateway(
            config.broker_gateway.broker_gateway.as_deref(),
            config.broker_gateway.broker_addr.as_deref(),
            config.broker_gateway.ctrader_app_client_id.as_deref(),
            config.broker_gateway.ctrader_app_client_secret.as_deref(),
            config.broker_gateway.ctrader_access_token.as_deref(),
            config.broker_gateway.ctrader_refresh_token.as_deref(),
            config.broker_gateway.ctrader_account.as_deref(),
            config.symbol.as_str(),
        )
        .await
        .map_err(|err| {
            error!("Failed to initialize broker gateway: {}", err);
            err
        })?;

        match broker_gateway {
            Some(bg) => {
                environment = environment.with_broker_gateway(bg);
                info!("Broker gateway connected successfully");
            }
            None => {
                // This shouldn't happen since we checked is_broker_gateway_configured()
                let err = anyhow::anyhow!("Broker gateway not configured but mode is Live");
                error!("Failed to start in live mode: {}", err);
                return Err(err);
            }
        }
    }

    // Create the gRPC service
    let environment_service = EnvironmentService::new(environment);

    // Start the gRPC server
    let addr = config.addr.parse().map_err(|err| {
        error!("Invalid gRPC server address '{}': {}", config.addr, err);
        err
    })?;
    info!("Starting gRPC server on {}", addr);

    Server::builder()
        .add_service(EnvironmentServer::new(environment_service))
        .serve(addr)
        .await
        .map_err(|err| {
            error!("gRPC server exited with error: {}", err);
            err
        })?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_environment_applies_reward_config() {
        let mut config = Config::default();
        config.reward_lambda = 2.5;
        config.reward_action_penalty = 0.05;
        config.reward_holding_penalty = 0.0002;

        let environment = build_environment(&config);

        assert_eq!(environment.reward_parameters(), (2.5, 0.05, 0.0002));
    }
}
