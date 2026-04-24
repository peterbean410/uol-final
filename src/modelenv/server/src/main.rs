// FX RL Model Environment Server
mod server;
mod broker;

use anyhow::Result;
use modelenv_core::{config::{Config, Mode}, environment::Environment};

use tonic::transport::Server;
use modelenv_proto::environment_server::EnvironmentServer;
use crate::server::EnvironmentService;

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
        println!("Connecting to broker gateway...");
        
        // Try to create broker gateway connection
        let broker_gateway = broker::try_create_broker_gateway(
            config.broker_gateway.broker_gateway.as_deref(),
            config.broker_gateway.broker_addr.as_deref(),
            config.broker_gateway.broker_username.as_deref(),
            config.broker_gateway.broker_password.as_deref(),
            config.broker_gateway.broker_account.as_deref(),
        )
        .await?;
        
        match broker_gateway {
            Some(bg) => {
                environment = environment.with_broker_gateway(bg);
                println!("Broker gateway connected successfully");
            }
            None => {
                // This shouldn't happen since we checked is_broker_gateway_configured()
                return Err(anyhow::anyhow!(
                    "Broker gateway not configured but mode is Live"
                ));
            }
        }
    } else if config.mode == Mode::Live && !config.is_broker_gateway_configured() {
        return Err(anyhow::anyhow!(
            "Production Mode requires broker gateway configuration. \
            Please provide --broker-gateway and --broker-addr arguments."
        ));
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
