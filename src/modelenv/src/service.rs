use crate::env::Environment;
use crate::proto;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Mutex;
use tokio_stream::{wrappers::ReceiverStream, Stream};
use tonic::{Request, Response, Status};

pub struct EnvironmentService {
    backend: Arc<Mutex<dyn Environment + Send>>,
}

impl EnvironmentService {
    pub fn new(backend: Arc<Mutex<dyn Environment + Send>>) -> Self {
        Self { backend }
    }
}

fn to_status(err: anyhow::Error) -> Status {
    Status::internal(format!("{err:#}"))
}

#[tonic::async_trait]
impl proto::environment_server::Environment for EnvironmentService {
    type StreamObservationsStream =
        Pin<Box<dyn Stream<Item = Result<proto::Observation, Status>> + Send + 'static>>;

    async fn reset(
        &self,
        req: Request<proto::ResetRequest>,
    ) -> Result<Response<proto::Observation>, Status> {
        let mut guard = self.backend.lock().await;
        let obs = guard.reset(req.into_inner()).await.map_err(to_status)?;
        Ok(Response::new(obs))
    }

    async fn step(
        &self,
        req: Request<proto::Action>,
    ) -> Result<Response<proto::StepResponse>, Status> {
        let mut guard = self.backend.lock().await;
        let outcome = guard.step(req.into_inner()).await.map_err(to_status)?;
        Ok(Response::new(proto::StepResponse {
            observation: Some(outcome.observation),
            reward: outcome.reward,
            done: outcome.done,
            info: outcome.info,
        }))
    }

    async fn observe(
        &self,
        _req: Request<proto::ObserveRequest>,
    ) -> Result<Response<proto::Observation>, Status> {
        let guard = self.backend.lock().await;
        let obs = guard.observe().await.map_err(to_status)?;
        Ok(Response::new(obs))
    }

    async fn stream_observations(
        &self,
        _req: Request<proto::ObserveRequest>,
    ) -> Result<Response<Self::StreamObservationsStream>, Status> {
        let (tx, rx) = tokio::sync::mpsc::channel(32);
        let backend = self.backend.clone();
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_millis(500));
            loop {
                interval.tick().await;
                let obs = {
                    let guard = backend.lock().await;
                    guard.observe().await
                };
                let msg = obs.map_err(to_status);
                if tx.send(msg).await.is_err() {
                    break;
                }
            }
        });
        let stream = ReceiverStream::new(rx);
        Ok(Response::new(Box::pin(stream) as Self::StreamObservationsStream))
    }
}
