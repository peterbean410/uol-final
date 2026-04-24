// gRPC server implementation for FX RL Model Environment
use anyhow::Result;
use std::pin::Pin;
use std::sync::Arc;
use tokio::sync::Mutex;
use tokio::sync::mpsc;
use tonic::{Request, Response, Status};
use futures::stream::{self, Stream};

use modelenv_proto::{
    environment_server::Environment, ResetRequest, ObserveRequest, Action, StepResponse,
    Observation,
};

use modelenv_core::environment::Environment as CoreEnvironment;

/// Environment service implementation
pub struct EnvironmentService {
    environment: Arc<Mutex<CoreEnvironment>>,
}

impl EnvironmentService {
    /// Create a new EnvironmentService
    pub fn new(environment: CoreEnvironment) -> Self {
        EnvironmentService { 
            environment: Arc::new(Mutex::new(environment)) 
        }
    }
}

/// Response type for streaming observations
type StreamObservationsResponse = Pin<Box<dyn Stream<Item = Result<Observation, Status>> + Send>>;

#[tonic::async_trait]
impl Environment for EnvironmentService {
    type StreamObservationsStream = StreamObservationsResponse;

    async fn reset(&self, req: Request<ResetRequest>) -> Result<Response<Observation>, Status> {
        let reset_req = req.into_inner();
        
        // Call the environment reset method
        let mut env = self.environment.lock().await;
        let observation = env.reset(reset_req)
            .await
            .map_err(|e| Status::internal(format!("Reset failed: {}", e)))?;
        
        Ok(Response::new(observation))
    }

    async fn step(&self, req: Request<Action>) -> Result<Response<StepResponse>, Status> {
        let action = req.into_inner();
        
        // Call the environment step method
        let mut env = self.environment.lock().await;
        let step_response = env.step(action)
            .await
            .map_err(|e| Status::internal(format!("Step failed: {}", e)))?;
        
        Ok(Response::new(step_response))
    }

    async fn observe(&self, req: Request<ObserveRequest>) -> Result<Response<Observation>, Status> {
        let observe_req = req.into_inner();
        
        // Call the environment observe method
        let env = self.environment.lock().await;
        let observation = env.observe(observe_req)
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
            let observation = {
                let env = env.lock().await;
                env.observe(ObserveRequest { symbol }).await
            };
            
            let observation = match observation {
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
