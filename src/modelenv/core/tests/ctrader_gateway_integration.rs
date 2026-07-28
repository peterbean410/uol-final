use std::sync::Arc;

use anyhow::Result;
use modelenv_core::{
    broker_gateway::create_broker_gateway_instance, config::Mode, environment::Environment,
};
use modelenv_proto::{Action, ActionType, ResetRequest};

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
        false, // demo endpoint
        0.01,  // lots per unit
    )?;

    Ok(Arc::from(gateway))
}

/// The real client must NOT fabricate broker data: without a reachable broker
/// (here the account id is non-numeric, so connect fails before any network),
/// every trading operation surfaces an error rather than returning a synthetic
/// value. (The old simulation returned fake positions/bars/ticks/fills; that is
/// gone by design, real wiring is unit-tested in `ctrader::client`.)
#[tokio::test]
async fn ctrader_gateway_refuses_to_fabricate_without_broker() -> Result<()> {
    let gateway = create_mock_ctrader_gateway()?;

    assert!(gateway.sync_positions("USDJPY").await.is_err());
    assert!(gateway.current_bar("USDJPY").await.is_err());
    assert!(gateway.current_ticks("USDJPY").await.is_err());
    assert!(gateway
        .submit(&Action {
            action: ActionType::ActionBuy1 as i32,
            client_order_id: "integration-order".to_string(),
        })
        .await
        .is_err());

    Ok(())
}

fn col_index(columns: &[String], name: &str) -> usize {
    columns.iter().position(|c| c == name).unwrap()
}

fn first_row_value(columns: &[String], values: &[f64], name: &str) -> f64 {
    values[col_index(columns, name)]
}

// Needs a reachable broker (demo/live) or a pub MockBrokerGateway to drive the
// live Reset/Step arms; the old version relied on the removed in-client
// simulation. Tracked as T-9.2-06 (env integration test with a mock gateway).
#[ignore = "needs a reachable demo/live broker or a pub MockBrokerGateway (T-9.2-06)"]
#[tokio::test]
async fn live_environment_reset_and_step_use_ctrader_gateway_end_to_end() -> Result<()> {
    let broker_gateway = create_mock_ctrader_gateway()?;
    let mut environment =
        Environment::new(Mode::Live, "USDJPY".to_string(), "s3://unused".to_string())
            .with_broker_gateway(broker_gateway);

    let obs = environment
        .reset(ResetRequest {
            symbol: "USDJPY".to_string(),
            episode_start_ts: 0,
            episode_end_ts: 0,
            seed: 0,
            step_size_seconds: 0,
        })
        .await?;

    let cols = &obs.state_columns;
    let vals = &obs.state_data[0].values;
    assert!(!cols.is_empty());
    assert_eq!(vals.len(), cols.len());
    // tick_ask is present (z-scored → 0.0 before warmup)
    assert!(cols.contains(&"tick_ask".to_string()));
    assert!(first_row_value(cols, vals, "tick_ask").is_finite());

    let step_response = environment
        .step(Action {
            action: ActionType::ActionBuy1 as i32,
            client_order_id: "live-step-order".to_string(),
        })
        .await?;

    assert!(!step_response.data.as_ref().unwrap().done);
    let step_obs = step_response.data.expect("step observation");
    let step_cols = &step_obs.state_columns;
    let step_vals = &step_obs.state_data[0].values;
    assert_eq!(step_vals.len(), step_cols.len());

    Ok(())
}
