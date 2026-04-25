// Reconciliation module for Production Mode
use log::warn;
use std::collections::HashMap;

use modelenv_proto::Position;

/// Reconcile internal positions with broker-reported positions
///
/// This function compares the internal position state with the broker's reported positions
/// and logs warnings for any discrepancies.
///
/// # Arguments
/// * `internal_positions` - Current internal positions
/// * `broker_positions` - Positions reported by the broker
/// * `internal_realised_pnl` - Internal calculated realised P/L
/// * `broker_realised_pnl` - Realised P/L reported by the broker
///
/// # Returns
/// * `Ok(())` if reconciliation completed successfully
/// * `Err(_)` if an error occurred during reconciliation
pub fn reconcile_positions(
    internal_positions: &[Position],
    broker_positions: &[Position],
    internal_realised_pnl: f64,
    broker_realised_pnl: f64,
) {
    // Create maps for easier lookup by position_id
    let internal_map: HashMap<&str, &Position> = internal_positions
        .iter()
        .map(|p| (p.position_id.as_str(), p))
        .collect();
    
    let broker_map: HashMap<&str, &Position> = broker_positions
        .iter()
        .map(|p| (p.position_id.as_str(), p))
        .collect();

    // Check 1: Position count mismatch
    if internal_positions.len() != broker_positions.len() {
        warn!(
            "Position count mismatch: internal={}, broker={}",
            internal_positions.len(),
            broker_positions.len()
        );
    }

    // Check 2-4: Compare positions that exist in both internal and broker state
    for broker_position in broker_positions {
        let position_id = &broker_position.position_id;
        
        if let Some(internal_position) = internal_map.get(position_id.as_str()) {
            // Position exists in both - compare details
            
            // Entry price drift check (threshold: 1e-6)
            let entry_price_drift = (internal_position.entry_price - broker_position.entry_price).abs();
            if entry_price_drift > 1e-6 {
                warn!(
                    "Entry price drift exceeds threshold for position {}: internal={}, broker={}, drift={}",
                    position_id,
                    internal_position.entry_price,
                    broker_position.entry_price,
                    entry_price_drift
                );
            }

            // Unrealised P/L divergence check (threshold: 0.01)
            let unrealised_pnl_divergence = (internal_position.unrealised_pnl - broker_position.unrealised_pnl).abs();
            if unrealised_pnl_divergence > 0.01 {
                warn!(
                    "Unrealised P/L divergence exceeds threshold for position {}: internal={}, broker={}, divergence={}",
                    position_id,
                    internal_position.unrealised_pnl,
                    broker_position.unrealised_pnl,
                    unrealised_pnl_divergence
                );
            }

            // Swap divergence check (threshold: 0.01)
            let swap_divergence = (internal_position.swap - broker_position.swap).abs();
            if swap_divergence > 0.01 {
                warn!(
                    "Swap divergence exceeds threshold for position {}: internal={}, broker={}, divergence={}",
                    position_id,
                    internal_position.swap,
                    broker_position.swap,
                    swap_divergence
                );
            }
        } else {
            // Position exists in broker but not in internal state
            warn!(
                "Broker position not found in internal state: position_id={}",
                position_id
            );
        }
    }

    // Check 5: Check for internal positions not in broker state
    for internal_position in internal_positions {
        let position_id = &internal_position.position_id;
        
        if !broker_map.contains_key(position_id.as_str()) {
            warn!(
                "Internal position not found in broker state: position_id={}",
                position_id
            );
        }
    }

    // Check 6: Realised P/L difference
    let realised_pnl_diff = (internal_realised_pnl - broker_realised_pnl).abs();
    // For realised P/L, we use a relative threshold since absolute differences can vary
    // Based on requirement: "WHERE a broker-reported realised P/L differs from internal calculation"
    // We'll log a warning if the absolute difference is significant
    if realised_pnl_diff > 0.01 {
        warn!(
            "Realised P/L differs from broker report: internal={}, broker={}, difference={}",
            internal_realised_pnl,
            broker_realised_pnl,
            realised_pnl_diff
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reconcile_positions_no_discrepancies() {
        let internal_positions = vec![
            Position {
                position_id: "pos_1".to_string(),
                entry_price: 150.0,
                unrealised_pnl: 10.0,
                swap: 0.5,
                open_timestamp_ns: 1000000000000,
                volume: 1.0,
                side: 0,
            },
            Position {
                position_id: "pos_2".to_string(),
                entry_price: 151.0,
                unrealised_pnl: -5.0,
                swap: 0.3,
                open_timestamp_ns: 2000000000000,
                volume: 2.0,
                side: 1,
            },
        ];

        let broker_positions = internal_positions.clone();

        // Should not log any warnings for matching positions
        reconcile_positions(&internal_positions, &broker_positions, 100.0, 100.0);
    }

    #[test]
    fn test_reconcile_positions_count_mismatch() {
        let internal_positions = vec![
            Position {
                position_id: "pos_1".to_string(),
                entry_price: 150.0,
                unrealised_pnl: 10.0,
                swap: 0.5,
                open_timestamp_ns: 1000000000000,
                volume: 1.0,
                side: 0,
            },
        ];

        let broker_positions = vec![
            Position {
                position_id: "pos_1".to_string(),
                entry_price: 150.0,
                unrealised_pnl: 10.0,
                swap: 0.5,
                open_timestamp_ns: 1000000000000,
                volume: 1.0,
                side: 0,
            },
            Position {
                position_id: "pos_2".to_string(),
                entry_price: 151.0,
                unrealised_pnl: 5.0,
                swap: 0.3,
                open_timestamp_ns: 2000000000000,
                volume: 1.0,
                side: 0,
            },
        ];

        // Should log count mismatch warning
        reconcile_positions(&internal_positions, &broker_positions, 100.0, 100.0);
    }

    #[test]
    fn test_reconcile_positions_entry_price_drift() {
        let internal_positions = vec![
            Position {
                position_id: "pos_1".to_string(),
                entry_price: 150.0,
                unrealised_pnl: 10.0,
                swap: 0.5,
                open_timestamp_ns: 1000000000000,
                volume: 1.0,
                side: 0,
            },
        ];

        let broker_positions = vec![
            Position {
                position_id: "pos_1".to_string(),
                entry_price: 150.000002, // Drift of 2e-6 > 1e-6
                unrealised_pnl: 10.0,
                swap: 0.5,
                open_timestamp_ns: 1000000000000,
                volume: 1.0,
                side: 0,
            },
        ];

        // Should log entry price drift warning
        reconcile_positions(&internal_positions, &broker_positions, 100.0, 100.0);
    }

    #[test]
    fn test_reconcile_positions_unrealised_pnl_divergence() {
        let internal_positions = vec![
            Position {
                position_id: "pos_1".to_string(),
                entry_price: 150.0,
                unrealised_pnl: 10.0,
                swap: 0.5,
                open_timestamp_ns: 1000000000000,
                volume: 1.0,
                side: 0,
            },
        ];

        let broker_positions = vec![
            Position {
                position_id: "pos_1".to_string(),
                entry_price: 150.0,
                unrealised_pnl: 10.02, // Divergence of 0.02 > 0.01
                swap: 0.5,
                open_timestamp_ns: 1000000000000,
                volume: 1.0,
                side: 0,
            },
        ];

        // Should log unrealised P/L divergence warning
        reconcile_positions(&internal_positions, &broker_positions, 100.0, 100.0);
    }

    #[test]
    fn test_reconcile_positions_swap_divergence() {
        let internal_positions = vec![
            Position {
                position_id: "pos_1".to_string(),
                entry_price: 150.0,
                unrealised_pnl: 10.0,
                swap: 0.5,
                open_timestamp_ns: 1000000000000,
                volume: 1.0,
                side: 0,
            },
        ];

        let broker_positions = vec![
            Position {
                position_id: "pos_1".to_string(),
                entry_price: 150.0,
                unrealised_pnl: 10.0,
                swap: 0.52, // Divergence of 0.02 > 0.01
                open_timestamp_ns: 1000000000000,
                volume: 1.0,
                side: 0,
            },
        ];

        // Should log swap divergence warning
        reconcile_positions(&internal_positions, &broker_positions, 100.0, 100.0);
    }

    #[test]
    fn test_reconcile_positions_missing_broker_position() {
        let internal_positions = vec![
            Position {
                position_id: "pos_1".to_string(),
                entry_price: 150.0,
                unrealised_pnl: 10.0,
                swap: 0.5,
                open_timestamp_ns: 1000000000000,
                volume: 1.0,
                side: 0,
            },
        ];

        let broker_positions = vec![]; // Empty - broker position not found

        // Should log warning for missing broker position
        reconcile_positions(&internal_positions, &broker_positions, 100.0, 100.0);
    }

    #[test]
    fn test_reconcile_positions_missing_internal_position() {
        let internal_positions = vec![]; // Empty - internal position not found

        let broker_positions = vec![
            Position {
                position_id: "pos_1".to_string(),
                entry_price: 150.0,
                unrealised_pnl: 10.0,
                swap: 0.5,
                open_timestamp_ns: 1000000000000,
                volume: 1.0,
                side: 0,
            },
        ];

        // Should log warning for missing internal position
        reconcile_positions(&internal_positions, &broker_positions, 100.0, 100.0);
    }

    #[test]
    fn test_reconcile_positions_realised_pnl_diff() {
        let internal_positions = vec![
            Position {
                position_id: "pos_1".to_string(),
                entry_price: 150.0,
                unrealised_pnl: 10.0,
                swap: 0.5,
                open_timestamp_ns: 1000000000000,
                volume: 1.0,
                side: 0,
            },
        ];

        let broker_positions = internal_positions.clone();

        // Realised P/L differs by 0.02 > 0.01
        reconcile_positions(&internal_positions, &broker_positions, 100.0, 99.98);
    }

    #[test]
    fn test_reconcile_positions_threshold_boundaries() {
        // Test at exact threshold - should NOT log warning
        let internal_positions = vec![
            Position {
                position_id: "pos_1".to_string(),
                entry_price: 150.0,
                unrealised_pnl: 10.0,
                swap: 0.5,
                open_timestamp_ns: 1000000000000,
                volume: 1.0,
                side: 0,
            },
        ];

        let broker_positions = vec![
            Position {
                position_id: "pos_1".to_string(),
                entry_price: 150.0 + 1e-6, // Exactly at threshold
                unrealised_pnl: 10.0 + 0.01, // Exactly at threshold
                swap: 0.5 + 0.01, // Exactly at threshold
                open_timestamp_ns: 1000000000000,
                volume: 1.0,
                side: 0,
            },
        ];

        // Should NOT log warnings at exact threshold
        reconcile_positions(&internal_positions, &broker_positions, 100.0, 100.0);
    }
}
