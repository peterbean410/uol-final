use std::sync::Arc;

use anyhow::Result;
use modelenv_core::{
    broker_gateway::create_broker_gateway_instance, config::Mode, environment::Environment,
};
use modelenv_proto::{Action, ActionType, FillSide, ObserveRequest, ResetRequest};

fn create_mock_ctrader_gateway(
) -> Result<Arc<dyn modelenv_core::broker_gateway::BrokerGateway + Send + Sync>> {
    let gateway = create_broker_gateway_instance(
        "ctrader",
        Some("app-client-id".to_string()),
        Some("app-client-secret".to_string()),
        Some("access-token".to_string()),
        Some("refresh-token".to_string()),
        Some("account".to_string()),
        "USDJPY",
    )?;

    Ok(Arc::from(gateway))
}

#[tokio::test]
async fn ctrader_gateway_trait_returns_mock_api_responses() -> Result<()> {
    let gateway = create_mock_ctrader_gateway()?;

    let positions = gateway.sync_positions("USDJPY").await?;
    let bar = gateway.current_bar("USDJPY").await?;
    let ticks = gateway.current_ticks("USDJPY").await?;
    let fill = gateway
        .submit(&Action {
            action: ActionType::ActionBuy1 as i32,
            client_order_id: "integration-order".to_string(),
        })
        .await?;

    assert!(positions.is_empty());
    assert!(bar.close > 100.0);
    assert!(bar.high >= bar.close);
    assert!(!ticks.is_empty());
    assert!(ticks
        .windows(2)
        .all(|window| window[0].timestamp_ns < window[1].timestamp_ns));
    assert_eq!(fill.order_id, "order-USDJPY-integration-order");
    assert_eq!(fill.side, FillSide::Buy as i32);
    assert_eq!(fill.size, 1.0);

    Ok(())
}

#[tokio::test]
async fn live_environment_reset_and_step_use_ctrader_gateway_end_to_end() -> Result<()> {
    let broker_gateway = create_mock_ctrader_gateway()?;
    let mut environment =
        Environment::new(Mode::Live, "USDJPY".to_string(), "s3://unused".to_string())
            .with_broker_gateway(broker_gateway);

    let reset_observation = environment
        .reset(ResetRequest {
            symbol: "USDJPY".to_string(),
            episode_start_ts: 0,
            episode_end_ts: 0,
            seed: 0,
            step_size_seconds: 0,
        })
        .await?;

    assert_eq!(reset_observation.symbol, "USDJPY");
    assert!(reset_observation.positions.is_empty());
    assert!(reset_observation.live_bars.contains_key("M1"));
    assert!(reset_observation.live_bars["M1"].close > 100.0);
    assert!(!reset_observation.live_ticks.is_empty());

    let step_response = environment
        .step(Action {
            action: ActionType::ActionBuy1 as i32,
            client_order_id: "live-step-order".to_string(),
        })
        .await?;

    assert!(!step_response.done);
    let step_observation = step_response.observation.expect("step observation");
    assert_eq!(step_observation.recent_fills.len(), 1);
    assert_eq!(
        step_observation.recent_fills[0].order_id,
        "order-USDJPY-live-step-order"
    );
    assert_eq!(
        step_observation.recent_fills[0].side,
        FillSide::Buy as i32
    );
    assert!(step_observation.live_bars.contains_key("M1"));
    assert!(!step_observation.live_ticks.is_empty());

    let observe_response = environment
        .observe(ObserveRequest {
            symbol: "USDJPY".to_string(),
        })
        .await?;

    assert_eq!(observe_response.symbol, "USDJPY");
    assert_eq!(observe_response.recent_fills.len(), 1);
    assert_eq!(
        observe_response.recent_fills[0].order_id,
        "order-USDJPY-live-step-order"
    );
    assert!(observe_response.live_bars["M1"].close > 100.0);
    assert!(!observe_response.recent_ticks.is_empty());

    Ok(())
}
