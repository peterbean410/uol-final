// Configuration module for FX RL Model Environment
use anyhow::Result;
use log::info;
use std::env;

use crate::data_loader::DEFAULT_LOCAL_CACHE_DIR;

const CTRADER_GATEWAY: &str = "ctrader";

/// Operating mode for the environment
#[derive(Debug, Clone, PartialEq)]
pub enum Mode {
    Training,
    Live,
}

impl Default for Mode {
    fn default() -> Self {
        Mode::Training
    }
}

impl Mode {
    /// Parse mode from string
    pub fn from_str(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "training" => Ok(Mode::Training),
            "live" | "production" => Ok(Mode::Live),
            _ => Err(anyhow::anyhow!(
                "Invalid mode '{}'. Expected 'training' or 'live'",
                s
            )),
        }
    }

    /// Convert mode to string
    pub fn as_str(&self) -> &'static str {
        match self {
            Mode::Training => "training",
            Mode::Live => "live",
        }
    }
}

/// Broker gateway configuration
#[derive(Debug, Clone)]
pub struct BrokerGatewayConfig {
    /// Broker gateway type (e.g., "ctrader", "metatrader", "ib")
    pub broker_gateway: Option<String>,
    /// Broker gateway address (host:port)
    pub broker_addr: Option<String>,
    /// cTrader Open API application client ID
    pub ctrader_app_client_id: Option<String>,
    /// cTrader Open API application client secret
    pub ctrader_app_client_secret: Option<String>,
    /// cTrader Open API access token
    pub ctrader_access_token: Option<String>,
    /// cTrader Open API refresh token
    pub ctrader_refresh_token: Option<String>,
    /// cTrader trader account ID
    pub ctrader_account: Option<String>,
}

impl Default for BrokerGatewayConfig {
    fn default() -> Self {
        BrokerGatewayConfig {
            broker_gateway: None,
            broker_addr: None,
            ctrader_app_client_id: None,
            ctrader_app_client_secret: None,
            ctrader_access_token: None,
            ctrader_refresh_token: None,
            ctrader_account: None,
        }
    }
}

impl BrokerGatewayConfig {
    fn gateway_type(&self) -> Option<&str> {
        self.broker_gateway
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty())
    }

    fn is_ctrader(&self) -> bool {
        matches!(self.gateway_type(), Some(gateway) if gateway.eq_ignore_ascii_case(CTRADER_GATEWAY))
    }

    fn set_ctrader_gateway_if_unset(&mut self) {
        if self.gateway_type().is_none() {
            self.broker_gateway = Some(CTRADER_GATEWAY.to_string());
        }
    }

    fn app_client_id(&self) -> Option<&str> {
        self.ctrader_app_client_id
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty())
    }

    fn app_client_secret(&self) -> Option<&str> {
        self.ctrader_app_client_secret
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty())
    }

    fn access_token(&self) -> Option<&str> {
        self.ctrader_access_token
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty())
    }

    fn refresh_token(&self) -> Option<&str> {
        self.ctrader_refresh_token
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty())
    }

    fn account(&self) -> Option<&str> {
        self.ctrader_account
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty())
    }

    fn broker_addr(&self) -> Option<&str> {
        self.broker_addr
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty())
    }
}

/// Main configuration for the environment
#[derive(Debug, Clone)]
pub struct Config {
    /// Operating mode (training or live)
    pub mode: Mode,
    /// gRPC server address
    pub addr: String,
    /// S3 bucket or prefix for training data. Bucket roots are resolved to
    /// the correct marketdata snapshot branch automatically.
    pub s3_prefix: String,
    /// Optional training price snapshot selection timestamp in nanoseconds.
    /// When set, the server loads the latest snapshot at or before this timestamp.
    pub price_snapshot_ts: Option<i64>,
    /// Local directory for downloaded parquet cache files.
    pub local_cache_dir: String,
    /// Trading symbol
    pub symbol: String,
    /// Broker gateway configuration
    pub broker_gateway: BrokerGatewayConfig,
    /// Reward function configuration
    pub reward_lambda: f64,
    pub reward_action_penalty: f64,
    pub reward_holding_penalty: f64,
    /// When true, close opposite-side positions before opening new ones (no hedging)
    pub disable_hedging: bool,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            mode: Mode::default(),
            addr: "0.0.0.0:50051".to_string(),
            s3_prefix: "s3://prod-fintech-forex-sg-731833471586".to_string(),
            price_snapshot_ts: None,
            local_cache_dir: DEFAULT_LOCAL_CACHE_DIR.to_string(),
            symbol: "USDJPY".to_string(),
            broker_gateway: BrokerGatewayConfig::default(),
            reward_lambda: 1.0,
            reward_action_penalty: 0.001,
            reward_holding_penalty: 1e-6,
            disable_hedging: true,
        }
    }
}

impl Config {
    /// Create a new config with default values
    pub fn new() -> Self {
        Config::default()
    }

    /// Load configuration from command-line arguments and environment variables
    pub fn load() -> Result<Self> {
        let args: Vec<String> = env::args().collect();
        Self::load_from_sources(&args, |key| env::var(key).ok())
    }

    fn load_from_sources<F>(args: &[String], env_get: F) -> Result<Self>
    where
        F: Fn(&str) -> Option<String>,
    {
        let mut config = Config::new();

        config.apply_env_defaults_from(&env_get);
        config.parse_args(args)?;
        config.validate()?;

        Ok(config)
    }

    fn parse_args(&mut self, args: &[String]) -> Result<()> {
        // Parse arguments manually (no external crate dependency)
        let mut i = 1; // Skip program name
        while i < args.len() {
            match args[i].as_str() {
                "--mode" => {
                    if i + 1 < args.len() {
                        self.mode = Mode::from_str(&args[i + 1])?;
                        i += 2;
                    } else {
                        return Err(anyhow::anyhow!("--mode requires a value"));
                    }
                }
                "--addr" => {
                    if i + 1 < args.len() {
                        self.addr = args[i + 1].clone();
                        i += 2;
                    } else {
                        return Err(anyhow::anyhow!("--addr requires a value"));
                    }
                }
                "--s3-prefix" => {
                    if i + 1 < args.len() {
                        self.s3_prefix = args[i + 1].clone();
                        i += 2;
                    } else {
                        return Err(anyhow::anyhow!("--s3-prefix requires a value"));
                    }
                }
                "--price-snapshot-ts" => {
                    if i + 1 < args.len() {
                        self.price_snapshot_ts = Some(args[i + 1].parse()?);
                        i += 2;
                    } else {
                        return Err(anyhow::anyhow!("--price-snapshot-ts requires a value"));
                    }
                }
                "--local-cache-dir" => {
                    if i + 1 < args.len() {
                        self.local_cache_dir = args[i + 1].clone();
                        i += 2;
                    } else {
                        return Err(anyhow::anyhow!("--local-cache-dir requires a value"));
                    }
                }
                "--symbol" => {
                    if i + 1 < args.len() {
                        self.symbol = args[i + 1].clone();
                        i += 2;
                    } else {
                        return Err(anyhow::anyhow!("--symbol requires a value"));
                    }
                }
                "--broker-gateway" => {
                    if i + 1 < args.len() {
                        self.broker_gateway.broker_gateway = Some(args[i + 1].clone());
                        i += 2;
                    } else {
                        return Err(anyhow::anyhow!("--broker-gateway requires a value"));
                    }
                }
                "--broker-addr" => {
                    if i + 1 < args.len() {
                        self.broker_gateway.broker_addr = Some(args[i + 1].clone());
                        i += 2;
                    } else {
                        return Err(anyhow::anyhow!("--broker-addr requires a value"));
                    }
                }
                "--ctrader-app-client-id" => {
                    if i + 1 < args.len() {
                        self.broker_gateway.set_ctrader_gateway_if_unset();
                        self.broker_gateway.ctrader_app_client_id = Some(args[i + 1].clone());
                        i += 2;
                    } else {
                        return Err(anyhow::anyhow!("--ctrader-app-client-id requires a value"));
                    }
                }
                "--ctrader-app-client-secret" => {
                    if i + 1 < args.len() {
                        self.broker_gateway.set_ctrader_gateway_if_unset();
                        self.broker_gateway.ctrader_app_client_secret = Some(args[i + 1].clone());
                        i += 2;
                    } else {
                        return Err(anyhow::anyhow!(
                            "--ctrader-app-client-secret requires a value"
                        ));
                    }
                }
                "--ctrader-access-token" => {
                    if i + 1 < args.len() {
                        self.broker_gateway.set_ctrader_gateway_if_unset();
                        self.broker_gateway.ctrader_access_token = Some(args[i + 1].clone());
                        i += 2;
                    } else {
                        return Err(anyhow::anyhow!("--ctrader-access-token requires a value"));
                    }
                }
                "--ctrader-refresh-token" => {
                    if i + 1 < args.len() {
                        self.broker_gateway.set_ctrader_gateway_if_unset();
                        self.broker_gateway.ctrader_refresh_token = Some(args[i + 1].clone());
                        i += 2;
                    } else {
                        return Err(anyhow::anyhow!("--ctrader-refresh-token requires a value"));
                    }
                }
                "--ctrader-account" => {
                    if i + 1 < args.len() {
                        self.broker_gateway.set_ctrader_gateway_if_unset();
                        self.broker_gateway.ctrader_account = Some(args[i + 1].clone());
                        i += 2;
                    } else {
                        return Err(anyhow::anyhow!("--ctrader-account requires a value"));
                    }
                }
                "--reward-lambda" => {
                    if i + 1 < args.len() {
                        self.reward_lambda = args[i + 1].parse()?;
                        i += 2;
                    } else {
                        return Err(anyhow::anyhow!("--reward-lambda requires a value"));
                    }
                }
                "--reward-action-penalty" => {
                    if i + 1 < args.len() {
                        self.reward_action_penalty = args[i + 1].parse()?;
                        i += 2;
                    } else {
                        return Err(anyhow::anyhow!("--reward-action-penalty requires a value"));
                    }
                }
                "--reward-holding-penalty" => {
                    if i + 1 < args.len() {
                        self.reward_holding_penalty = args[i + 1].parse()?;
                        i += 2;
                    } else {
                        return Err(anyhow::anyhow!("--reward-holding-penalty requires a value"));
                    }
                }
                "--help" => {
                    print_help();
                    std::process::exit(0);
                }
                _ => {
                    eprintln!("Unknown argument: {}", args[i]);
                    print_help();
                    std::process::exit(1);
                }
            }
        }
        Ok(())
    }

    /// Apply environment variable defaults for configuration
    fn apply_env_defaults_from<F>(&mut self, env_get: &F)
    where
        F: Fn(&str) -> Option<String>,
    {
        // Mode
        if let Some(mode_env) = Self::non_empty_env(env_get, "MODELENV_MODE") {
            if let Ok(mode) = Mode::from_str(&mode_env) {
                self.mode = mode;
            }
        }

        // Address
        if let Some(addr_env) = Self::non_empty_env(env_get, "MODELENV_ADDR") {
            self.addr = addr_env;
        }

        // S3 prefix
        if let Some(s3_prefix_env) = Self::non_empty_env(env_get, "MODELENV_S3_PREFIX") {
            self.s3_prefix = s3_prefix_env;
        }

        if let Some(price_snapshot_ts_env) =
            Self::non_empty_env(env_get, "MODELENV_PRICE_SNAPSHOT_TS")
        {
            if let Ok(value) = price_snapshot_ts_env.parse::<i64>() {
                self.price_snapshot_ts = Some(value);
            }
        }

        if let Some(local_cache_dir_env) = Self::non_empty_env(env_get, "MODELENV_LOCAL_CACHE_DIR")
        {
            self.local_cache_dir = local_cache_dir_env;
        }

        // Symbol
        if let Some(symbol_env) = Self::non_empty_env(env_get, "MODELENV_SYMBOL") {
            self.symbol = symbol_env;
        }

        // Broker gateway configuration
        if let Some(broker_gateway_env) = Self::non_empty_env(env_get, "MODELENV_BROKER_GATEWAY") {
            self.broker_gateway.broker_gateway = Some(broker_gateway_env);
        }

        if let Some(broker_addr_env) = Self::non_empty_env(env_get, "MODELENV_BROKER_ADDR") {
            self.broker_gateway.broker_addr = Some(broker_addr_env);
        }

        if let Some(app_client_id) = Self::non_empty_env(env_get, "APP_CLIENT_ID") {
            self.broker_gateway.ctrader_app_client_id = Some(app_client_id);
        }

        if let Some(app_client_secret) = Self::non_empty_env(env_get, "APP_CLIENT_SECRET") {
            self.broker_gateway.ctrader_app_client_secret = Some(app_client_secret);
        }

        if let Some(access_token) = Self::non_empty_env(env_get, "ACCESS_TOKEN") {
            self.broker_gateway.ctrader_access_token = Some(access_token);
        }

        if let Some(refresh_token) = Self::non_empty_env(env_get, "REFRESH_TOKEN") {
            self.broker_gateway.ctrader_refresh_token = Some(refresh_token);
        }

        if let Some(ctrader_account) = Self::non_empty_env(env_get, "CTRADER_ACCOUNT") {
            self.broker_gateway.ctrader_account = Some(ctrader_account);
        }

        if Self::first_non_empty_env(
            env_get,
            &[
                "APP_CLIENT_ID",
                "APP_CLIENT_SECRET",
                "ACCESS_TOKEN",
                "REFRESH_TOKEN",
                "CTRADER_ACCOUNT",
            ],
        )
        .is_some()
        {
            self.broker_gateway.set_ctrader_gateway_if_unset();
        }

        // Reward function configuration
        if let Some(reward_lambda_env) = Self::non_empty_env(env_get, "MODELENV_REWARD_LAMBDA") {
            if let Ok(value) = reward_lambda_env.parse::<f64>() {
                self.reward_lambda = value;
            }
        }

        if let Some(reward_action_penalty_env) =
            Self::non_empty_env(env_get, "MODELENV_REWARD_ACTION_PENALTY")
        {
            if let Ok(value) = reward_action_penalty_env.parse::<f64>() {
                self.reward_action_penalty = value;
            }
        }

        if let Some(reward_holding_penalty_env) =
            Self::non_empty_env(env_get, "MODELENV_REWARD_HOLDING_PENALTY")
        {
            if let Ok(value) = reward_holding_penalty_env.parse::<f64>() {
                self.reward_holding_penalty = value;
            }
        }

        if let Some(disable_hedging_env) =
            Self::non_empty_env(env_get, "MODELENV_DISABLE_HEDGING")
        {
            self.disable_hedging = matches!(
                disable_hedging_env.to_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            );
        }
    }

    fn non_empty_env<F>(env_get: &F, key: &str) -> Option<String>
    where
        F: Fn(&str) -> Option<String>,
    {
        env_get(key)
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty())
    }

    fn first_non_empty_env<F>(env_get: &F, keys: &[&str]) -> Option<String>
    where
        F: Fn(&str) -> Option<String>,
    {
        keys.iter()
            .find_map(|key| Self::non_empty_env(env_get, key))
    }

    pub fn validate(&self) -> Result<()> {
        if self.price_snapshot_ts.is_some_and(|value| value <= 0) {
            return Err(anyhow::anyhow!(
                "price snapshot timestamp must be > 0 when configured"
            ));
        }

        if self.local_cache_dir.trim().is_empty() {
            return Err(anyhow::anyhow!("local cache dir must not be empty"));
        }

        if self.mode != Mode::Live {
            return Ok(());
        }

        let gateway = self.broker_gateway.gateway_type().ok_or_else(|| {
            anyhow::anyhow!(
                "Production Mode requires broker gateway configuration. Please provide --broker-gateway <TYPE>."
            )
        })?;

        if self.broker_gateway.is_ctrader() {
            let mut missing = Vec::new();

            if self.broker_gateway.app_client_id().is_none() {
                missing.push("--ctrader-app-client-id or APP_CLIENT_ID");
            }
            if self.broker_gateway.app_client_secret().is_none() {
                missing.push("--ctrader-app-client-secret or APP_CLIENT_SECRET");
            }
            if self.broker_gateway.access_token().is_none() {
                missing.push("--ctrader-access-token or ACCESS_TOKEN");
            }
            if self.broker_gateway.account().is_none() {
                missing.push("--ctrader-account or CTRADER_ACCOUNT");
            }

            if missing.is_empty() {
                return Ok(());
            }

            return Err(anyhow::anyhow!(
                "cTrader configuration incomplete: missing {}",
                missing.join(", ")
            ));
        }

        if self.broker_gateway.broker_addr().is_none() {
            return Err(anyhow::anyhow!(
                "Broker gateway '{}' requires --broker-addr or MODELENV_BROKER_ADDR",
                gateway
            ));
        }

        Ok(())
    }

    /// Check if broker gateway is configured
    pub fn is_broker_gateway_configured(&self) -> bool {
        self.broker_gateway.gateway_type().is_some() && self.validate().is_ok()
    }

    /// Log the current configuration
    pub fn log(&self) {
        info!("Starting FX RL Model Environment");
        info!("Mode: {}", self.mode.as_str());
        info!("Address: {}", self.addr);
        info!("S3 Prefix: {}", self.s3_prefix);
        match self.price_snapshot_ts {
            Some(value) => info!("Price Snapshot Timestamp: {}", value),
            None => info!("Price Snapshot Timestamp: latest available"),
        }
        info!("Local Cache Dir: {}", self.local_cache_dir);
        info!("Symbol: {}", self.symbol);
        info!("Reward Lambda: {}", self.reward_lambda);
        info!("Reward Action Penalty: {}", self.reward_action_penalty);
        info!("Reward Holding Penalty: {}", self.reward_holding_penalty);
        info!("Disable Hedging: {}", self.disable_hedging);

        if let Some(ref broker_gateway) = self.broker_gateway.broker_gateway {
            info!("Broker Gateway: {}", broker_gateway);
            if let Some(ref broker_addr) = self.broker_gateway.broker_addr {
                info!("Broker Address: {}", broker_addr);
            } else if self.broker_gateway.is_ctrader() {
                info!("Broker Address: default cTrader endpoint");
            }
            if let Some(ref app_client_id) = self.broker_gateway.ctrader_app_client_id {
                info!("cTrader App Client ID: {}", app_client_id);
            }
            if self.broker_gateway.ctrader_app_client_secret.is_some() {
                info!("cTrader App Client Secret: [set]");
            }
            if self.broker_gateway.ctrader_access_token.is_some() {
                info!("cTrader Access Token: [set]");
            }
            if self.broker_gateway.refresh_token().is_some() {
                info!("cTrader Refresh Token: [set]");
            }
            if self.broker_gateway.ctrader_account.is_some() {
                info!("cTrader Account: [set]");
            }
        } else {
            info!("Broker Gateway: not configured");
        }
    }
}

fn print_help() {
    println!("FX RL Model Environment & Gateway");
    println!();
    println!("Usage: modelenv-server [OPTIONS]");
    println!();
    println!("Options:");
    println!(
        "  --mode <MODE>              Operating mode: 'training' or 'live' (default: training)"
    );
    println!("  --addr <ADDRESS>           gRPC server address (default: 0.0.0.0:50051)");
    println!("  --s3-prefix <PREFIX>       S3 bucket/prefix for training data (default: s3://prod-fintech-forex-sg-731833471586)");
    println!("  --price-snapshot-ts <TS>   Select the latest price snapshot at or before TS (nanoseconds)");
    println!(
        "  --local-cache-dir <PATH>   Local parquet cache directory (default: /tmp/modelenv-cache)"
    );
    println!("  --symbol <SYMBOL>          Trading symbol (default: USDJPY)");
    println!(
        "  --broker-gateway <TYPE>    Broker gateway type (e.g., 'ctrader', 'metatrader', 'ib')"
    );
    println!("  --broker-addr <ADDRESS>    Broker gateway address (host:port)");
    println!("  --ctrader-app-client-id <ID>      cTrader Open API app client ID");
    println!("  --ctrader-app-client-secret <SECRET> cTrader Open API app client secret");
    println!("  --ctrader-access-token <TOKEN>    cTrader Open API access token");
    println!("  --ctrader-refresh-token <TOKEN>   cTrader Open API refresh token");
    println!("  --ctrader-account <ACCOUNT>       cTrader trader account ID");
    println!("  --reward-lambda <LAMBDA>   Asymmetric drawdown penalty coefficient (default: 1.0)");
    println!("  --reward-action-penalty <C_A>    Action penalty coefficient (default: 0.001)");
    println!("  --reward-holding-penalty <C_H>   Holding penalty coefficient (default: 1e-6)");
    println!("  --help                     Display this help and exit");
    println!();
    println!("Environment Variables:");
    println!("  MODELENV_MODE              Same as --mode");
    println!("  MODELENV_ADDR              Same as --addr");
    println!("  MODELENV_S3_PREFIX         Same as --s3-prefix");
    println!("  MODELENV_PRICE_SNAPSHOT_TS Same as --price-snapshot-ts");
    println!("  MODELENV_LOCAL_CACHE_DIR   Same as --local-cache-dir");
    println!("  MODELENV_SYMBOL            Same as --symbol");
    println!("  MODELENV_BROKER_GATEWAY    Same as --broker-gateway");
    println!("  MODELENV_BROKER_ADDR       Same as --broker-addr");
    println!("  APP_CLIENT_ID              Same as --ctrader-app-client-id");
    println!("  APP_CLIENT_SECRET          Same as --ctrader-app-client-secret");
    println!("  ACCESS_TOKEN               Same as --ctrader-access-token");
    println!("  REFRESH_TOKEN              Same as --ctrader-refresh-token");
    println!("  CTRADER_ACCOUNT            Same as --ctrader-account");
    println!("  MODELENV_REWARD_LAMBDA     Same as --reward-lambda");
    println!("  MODELENV_REWARD_ACTION_PENALTY   Same as --reward-action-penalty");
    println!("  MODELENV_REWARD_HOLDING_PENALTY  Same as --reward-holding-penalty");
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn load_test_config(args: &[&str], env_pairs: &[(&str, &str)]) -> Result<Config> {
        let argv = args
            .iter()
            .map(|value| value.to_string())
            .collect::<Vec<_>>();
        let env_map = env_pairs
            .iter()
            .map(|(key, value)| ((*key).to_string(), (*value).to_string()))
            .collect::<HashMap<_, _>>();

        Config::load_from_sources(&argv, |key| env_map.get(key).cloned())
    }

    #[test]
    fn ctrader_env_defaults_are_loaded_for_live_mode() {
        let config = load_test_config(
            &["modelenv-server", "--mode", "live"],
            &[
                ("APP_CLIENT_ID", "env-client-id"),
                ("APP_CLIENT_SECRET", "env-client-secret"),
                ("ACCESS_TOKEN", "env-access-token"),
                ("REFRESH_TOKEN", "env-refresh-token"),
                ("CTRADER_ACCOUNT", "env-account"),
            ],
        )
        .unwrap();

        assert_eq!(
            config.broker_gateway.broker_gateway.as_deref(),
            Some("ctrader")
        );
        assert_eq!(
            config.broker_gateway.ctrader_app_client_id.as_deref(),
            Some("env-client-id")
        );
        assert_eq!(
            config.broker_gateway.ctrader_app_client_secret.as_deref(),
            Some("env-client-secret")
        );
        assert_eq!(
            config.broker_gateway.ctrader_access_token.as_deref(),
            Some("env-access-token")
        );
        assert_eq!(
            config.broker_gateway.ctrader_refresh_token.as_deref(),
            Some("env-refresh-token")
        );
        assert_eq!(
            config.broker_gateway.ctrader_account.as_deref(),
            Some("env-account")
        );
    }

    #[test]
    fn cli_ctrader_args_override_env_defaults() {
        let config = load_test_config(
            &[
                "modelenv-server",
                "--mode",
                "live",
                "--broker-gateway",
                "ctrader",
                "--ctrader-app-client-id",
                "cli-client-id",
                "--ctrader-app-client-secret",
                "cli-client-secret",
                "--ctrader-access-token",
                "cli-access-token",
                "--ctrader-refresh-token",
                "cli-refresh-token",
                "--ctrader-account",
                "cli-account",
            ],
            &[
                ("APP_CLIENT_ID", "env-client-id"),
                ("APP_CLIENT_SECRET", "env-client-secret"),
                ("ACCESS_TOKEN", "env-access-token"),
                ("REFRESH_TOKEN", "env-refresh-token"),
                ("CTRADER_ACCOUNT", "env-account"),
            ],
        )
        .unwrap();

        assert_eq!(
            config.broker_gateway.ctrader_app_client_id.as_deref(),
            Some("cli-client-id")
        );
        assert_eq!(
            config.broker_gateway.ctrader_app_client_secret.as_deref(),
            Some("cli-client-secret")
        );
        assert_eq!(
            config.broker_gateway.ctrader_access_token.as_deref(),
            Some("cli-access-token")
        );
        assert_eq!(
            config.broker_gateway.ctrader_refresh_token.as_deref(),
            Some("cli-refresh-token")
        );
        assert_eq!(
            config.broker_gateway.ctrader_account.as_deref(),
            Some("cli-account")
        );
    }

    #[test]
    fn live_ctrader_validation_requires_credentials() {
        let err = load_test_config(
            &[
                "modelenv-server",
                "--mode",
                "live",
                "--broker-gateway",
                "ctrader",
            ],
            &[],
        )
        .unwrap_err();

        assert!(err.to_string().contains("cTrader configuration incomplete"));
        assert!(err.to_string().contains("APP_CLIENT_ID"));
    }

    #[test]
    fn live_ctrader_configuration_does_not_require_broker_addr() {
        let config = load_test_config(
            &[
                "modelenv-server",
                "--mode",
                "live",
                "--broker-gateway",
                "ctrader",
                "--ctrader-app-client-id",
                "client-id",
                "--ctrader-app-client-secret",
                "client-secret",
                "--ctrader-access-token",
                "access-token",
                "--ctrader-account",
                "account",
            ],
            &[],
        )
        .unwrap();

        assert!(config.is_broker_gateway_configured());
    }

    #[test]
    fn cli_price_snapshot_ts_overrides_env_default() {
        let config = load_test_config(
            &["modelenv-server", "--price-snapshot-ts", "200"],
            &[("MODELENV_PRICE_SNAPSHOT_TS", "100")],
        )
        .unwrap();

        assert_eq!(config.price_snapshot_ts, Some(200));
    }

    #[test]
    fn cli_local_cache_dir_overrides_env_default() {
        let config = load_test_config(
            &["modelenv-server", "--local-cache-dir", "/cli/cache"],
            &[("MODELENV_LOCAL_CACHE_DIR", "/env/cache")],
        )
        .unwrap();

        assert_eq!(config.local_cache_dir, "/cli/cache");
    }

    #[test]
    fn disable_hedging_defaults_to_true() {
        let config = load_test_config(&["modelenv-server"], &[]).unwrap();
        assert!(config.disable_hedging);
    }

    #[test]
    fn disable_hedging_parses_true_variants() {
        for value in &["1", "true", "yes", "on", "TRUE", "Yes"] {
            let config = load_test_config(
                &["modelenv-server"],
                &[("MODELENV_DISABLE_HEDGING", value)],
            )
            .unwrap();
            assert!(config.disable_hedging, "value={value}");
        }
    }

    #[test]
    fn disable_hedging_ignores_unrecognised_values() {
        let config = load_test_config(
            &["modelenv-server"],
            &[("MODELENV_DISABLE_HEDGING", "maybe")],
        )
        .unwrap();
        assert!(!config.disable_hedging);
    }
}
