// FX RL Model Environment Server
mod server;

use anyhow::Result;
use modelenv_core::{config::Config, environment::Environment, Mode};
use std::time::Duration;
use tonic::transport::Server;
use modelenv_proto::{environment_server::EnvironmentServer, EnvironmentService};

#[tokio::main]
async fn main() -> Result<()> {
    // Load configuration from command-line arguments and environment variables
    let config = Config::load()?;
    
    // Log configuration at startup
    config.log();
    
    // Create the environment
    let mode = config.mode.clone();
    let symbol = config.symbol.clone();
    let s3_prefix = config.s3_prefix.clone();
    
    let mut environment = Environment::new(mode, symbol, s3_prefix);
    
    // Configure broker gateway if in Production Mode
    if config.mode == Mode::Live && config.is_broker_gateway_configured() {
        // TODO: Create broker gateway implementation
        // For now, log that broker gateway is configured but not implemented
        println!("Broker gateway configured: {:?}", config.broker_gateway);
    }
    
    // Create the gRPC service
    let environment_service = EnvironmentService::new(environment);
    
    // Start the gRPC server
    let addr = config.addr.parse()?;
    println!("Starting gRPC server on {}", addr);
    
    Server::builder()
        .add_service(EnvironmentServer::new(environment_service))
        .serve(addr)
        .await?;
    
    Ok(())
}
