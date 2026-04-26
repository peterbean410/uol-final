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
    let mode = config.mode.clone();
    let symbol = config.symbol.clone();
    let s3_prefix = config.s3_prefix.clone();

    let mut environment = Environment::new(mode, symbol, s3_prefix);

    // Configure broker gateway if in Production Mode
    if config.mode == Mode::Live {
        info!("Connecting to broker gateway...");

        // Try to create broker gateway connection
        let broker_gateway = broker::try_create_broker_gateway(
            config.broker_gateway.broker_gateway.as_deref(),
            config.broker_gateway.broker_addr.as_deref(),
            config.broker_gateway.broker_username.as_deref(),
            config.broker_gateway.broker_password.as_deref(),
            config.broker_gateway.broker_account.as_deref(),
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
