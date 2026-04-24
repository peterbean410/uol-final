// gRPC server implementation for FX RL Model Environment
use anyhow::Result;
use std::pin::Pin;
use tokio::sync::mpsc;
use tonic::{Request, Response, Status};
use futures::Stream;

use modelenv_proto::{
    environment_server::Environment, ResetRequest, ObserveRequest, Action, StepResponse,
    Observation,
};

use modelenv_core::environment::Environment as CoreEnvironment;

/// Environment service implementation
pub struct EnvironmentService {
    environment: CoreEnvironment,
}

impl EnvironmentService {
    /// Create a new EnvironmentService
    pub fn new(environment: CoreEnvironment) -> Self {
        EnvironmentService { environment }
    }
}

/// Response type for streaming observations
type StreamObservationsResponse = Pin<Box<dyn Stream<Item = Result<Observation, Status>> + Send>>;

#[tonic::async_trait]
impl Environment for EnvironmentService {
    async fn reset(&self, req: Request<ResetRequest>) -> Result<Response<Observation>, Status> {
        let reset_req = req.into_inner();
        
        // Call the environment reset method
        let observation = self.environment.reset(reset_req)
            .await
            .map_err(|e| Status::internal(format!("Reset failed: {}", e)))?;
        
        Ok(Response::new(observation))
    }

    async fn step(&self, req: Request<Action>) -> Result<Response<StepResponse>, Status> {
        let action = req.into_inner();
        
        // Call the environment step method
        let step_response = self.environment.step(action)
            .await
            .map_err(|e| Status::internal(format!("Step failed: {}", e)))?;
        
        Ok(Response::new(step_response))
    }

    async fn observe(&self, req: Request<ObserveRequest>) -> Result<Response<Observation>, Status> {
        let observe_req = req.into_inner();
        
        // Call the environment observe method
        let observation = self.environment.observe(observe_req)
            .await
            .map_err(|e| Status::internal(format!("Observe failed: {}", e)))?;
        
        Ok(Response::new(observation))
    }

    async fn stream_observations(
        &self,
        req: Request<ObserveRequest>,
    ) -> Result<Response<StreamObservationsResponse>, Status> {
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
            let observation = match env.observe(ObserveRequest { symbol }).await {
                Ok(obs) => obs,
                Err(e) => {
                    let _ = tx.send(Err(Status::internal(format!("Observe failed: {}", e)))).await;
                    return;
                }
            };
            
            // Send the observation
            if tx.send(Ok(observation)).await.is_err() {
                return;
            }
            
            // Close the channel
        });
        
        let stream = Box::pin(tokio_stream::from_iter(rx));
        Ok(Response::new(stream))
    }
}
