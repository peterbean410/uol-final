// gRPC server implementation for FX RL Model Environment
use anyhow::Result;
use futures::stream::{self, Stream};
use std::pin::Pin;
use std::sync::Arc;
use tokio::sync::mpsc;
use tokio::sync::Mutex;
use tonic::{Request, Response, Status};

use modelenv_proto::{
    environment_server::Environment, Action, ObserveRequest, RecentBarsRequest,
    RecentBarsResponse, RecentNewsRequest, RecentNewsResponse, RecentTicksRequest,
    RecentTicksResponse, Reference,
    ResetRequest, StepResponse,
};

use modelenv_core::environment::Environment as CoreEnvironment;

fn format_error_chain(err: &anyhow::Error) -> String {
    err.chain()
        .map(|cause| cause.to_string())
        .collect::<Vec<_>>()
        .join(": ")
}

fn internal_error_status(operation: &str, err: anyhow::Error) -> Status {
    Status::internal(format!("{operation} failed: {}", format_error_chain(&err)))
}

/// Environment service implementation
pub struct EnvironmentService {
    environment: Arc<Mutex<CoreEnvironment>>,
}

impl EnvironmentService {
    /// Create a new EnvironmentService
    pub fn new(environment: CoreEnvironment) -> Self {
        EnvironmentService {
            environment: Arc::new(Mutex::new(environment)),
        }
    }
}

/// Response type for streaming observations
type StreamReferencesResponse = Pin<Box<dyn Stream<Item = Result<Reference, Status>> + Send>>;

#[tonic::async_trait]
impl Environment for EnvironmentService {
    type StreamReferencesStream = StreamReferencesResponse;

    async fn reset(&self, req: Request<ResetRequest>) -> Result<Response<Reference>, Status> {
        let reset_req = req.into_inner();

        // Call the environment reset method
        let mut env = self.environment.lock().await;
        let observation = env
            .reset(reset_req)
            .await
            .map_err(|e| internal_error_status("Reset", e))?;

        Ok(Response::new(observation))
    }

    async fn step(&self, req: Request<Action>) -> Result<Response<StepResponse>, Status> {
        let action = req.into_inner();

        // Call the environment step method
        let mut env = self.environment.lock().await;
        let step_response = env
            .step(action)
            .await
            .map_err(|e| internal_error_status("Step", e))?;

        Ok(Response::new(step_response))
    }

    async fn observe(&self, req: Request<ObserveRequest>) -> Result<Response<Reference>, Status> {
        let observe_req = req.into_inner();

        // Call the environment observe method
        let mut env = self.environment.lock().await;
        let observation = env
            .observe(observe_req)
            .await
            .map_err(|e| internal_error_status("Observe", e))?;

        Ok(Response::new(observation))
    }

    async fn recent_bars(
        &self,
        req: Request<RecentBarsRequest>,
    ) -> Result<Response<RecentBarsResponse>, Status> {
        let request = req.into_inner();
        let env = self.environment.lock().await;
        let response = env
            .recent_bars(request)
            .await
            .map_err(|e| internal_error_status("RecentBars", e))?;
        Ok(Response::new(response))
    }

    async fn recent_ticks(
        &self,
        req: Request<RecentTicksRequest>,
    ) -> Result<Response<RecentTicksResponse>, Status> {
        let request = req.into_inner();
        let env = self.environment.lock().await;
        let response = env
            .recent_ticks(request)
            .await
            .map_err(|e| internal_error_status("RecentTicks", e))?;
        Ok(Response::new(response))
    }

    async fn recent_news(
        &self,
        req: Request<RecentNewsRequest>,
    ) -> Result<Response<RecentNewsResponse>, Status> {
        let request = req.into_inner();
        let env = self.environment.lock().await;
        let response = env
            .recent_news(request)
            .await
            .map_err(|e| internal_error_status("RecentNews", e))?;
        Ok(Response::new(response))
    }

    async fn stream_references(
        &self,
        req: Request<ObserveRequest>,
    ) -> Result<Response<StreamReferencesResponse>, Status> {
        let observe_req = req.into_inner();

        // Create a channel for streaming observations
        let (tx, rx) = mpsc::channel(16);

        // Clone the environment for the streaming task
        let env = self.environment.clone();
        let symbol = observe_req.symbol;

        // Spawn a task to generate observations
        tokio::spawn(async move {
            // TODO: Implement streaming for Production Mode
            // For now, just send a single observation and close the stream

            // Get initial observation
            let observation = {
                let mut env = env.lock().await;
                env.observe(ObserveRequest { symbol }).await
            };

            let observation = match observation {
                Ok(obs) => obs,
                Err(e) => {
                    let _ = tx.send(Err(internal_error_status("Observe", e))).await;
                    return;
                }
            };

            // Send the observation
            if tx.send(Ok(observation)).await.is_err() {
                return;
            }

            // Close the channel
        });

        let stream = stream::unfold(rx, |mut rx| async move {
            match rx.recv().await {
                Some(item) => Some((item, rx)),
                None => None,
            }
        });
        let stream = Box::pin(stream);
        Ok(Response::new(stream))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn format_error_chain_includes_causes() {
        let err = anyhow::anyhow!("root cause")
            .context("middle context")
            .context("top context");

        assert_eq!(
            format_error_chain(&err),
            "top context: middle context: root cause"
        );
    }
}
