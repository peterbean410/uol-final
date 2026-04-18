use anyhow::Result;
use clap::Parser;
use std::sync::Arc;
use tokio::sync::Mutex;
use tonic::transport::Server;
use tracing_subscriber::EnvFilter;

mod data;
mod env;
mod proto;
mod service;

use env::{live::LiveBackend, training::TrainingBackend, Environment};
use proto::environment_server::EnvironmentServer;
use service::EnvironmentService;

#[derive(Debug, Clone, clap::ValueEnum)]
enum Mode {
    Training,
    Live,
}

#[derive(Parser, Debug)]
#[command(name = "modelenv", about = "FX RL environment / live gateway")]
struct Cli {
    #[arg(long, value_enum, env = "MODELENV_MODE", default_value = "training")]
    mode: Mode,

    #[arg(long, env = "MODELENV_ADDR", default_value = "0.0.0.0:50051")]
    addr: String,

    #[arg(
        long,
        env = "MODELENV_S3_PREFIX",
        default_value = "s3://prod-fintech-forex-sg-731833471586/marketdata/"
    )]
    s3_prefix: String,

    #[arg(long, env = "MODELENV_SYMBOL", default_value = "USDJPY")]
    symbol: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    let cli = Cli::parse();
    let addr = cli.addr.parse()?;

    let backend: Arc<Mutex<dyn Environment + Send>> = match cli.mode {
        Mode::Training => {
            tracing::info!(s3_prefix = %cli.s3_prefix, "starting in training mode");
            let backend =
                TrainingBackend::new(cli.s3_prefix.clone(), cli.symbol.clone()).await?;
            Arc::new(Mutex::new(backend))
        }
        Mode::Live => {
            tracing::info!("starting in live mode");
            let backend = LiveBackend::new(cli.symbol.clone())?;
            Arc::new(Mutex::new(backend))
        }
    };

    let svc = EnvironmentService::new(backend);

    tracing::info!(%addr, "modelenv gRPC server listening");
    Server::builder()
        .add_service(EnvironmentServer::new(svc))
        .serve(addr)
        .await?;

    Ok(())
}
