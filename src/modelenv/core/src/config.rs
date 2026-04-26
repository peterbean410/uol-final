// Configuration module for FX RL Model Environment
use anyhow::Result;
use log::info;
use std::env;

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
    /// Broker gateway username
    pub broker_username: Option<String>,
    /// Broker gateway password
    pub broker_password: Option<String>,
    /// Broker gateway account
    pub broker_account: Option<String>,
}

impl Default for BrokerGatewayConfig {
    fn default() -> Self {
        BrokerGatewayConfig {
            broker_gateway: None,
            broker_addr: None,
            broker_username: None,
            broker_password: None,
            broker_account: None,
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

    fn username(&self) -> Option<&str> {
        self.broker_username
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty())
    }

    fn password(&self) -> Option<&str> {
        self.broker_password
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty())
    }

    fn account(&self) -> Option<&str> {
        self.broker_account
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
    /// S3 bucket prefix for training data
    pub s3_prefix: String,
    /// Trading symbol
    pub symbol: String,
    /// Broker gateway configuration
    pub broker_gateway: BrokerGatewayConfig,
    /// Reward function configuration
    pub reward_lambda: f64,
    pub reward_action_penalty: f64,
    pub reward_holding_penalty: f64,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            mode: Mode::default(),
            addr: "0.0.0.0:50051".to_string(),
            s3_prefix: "s3://modelenv-data".to_string(),
            symbol: "USDJPY".to_string(),
            broker_gateway: BrokerGatewayConfig::default(),
            reward_lambda: 1.0,
            reward_action_penalty: 0.001,
            reward_holding_penalty: 1e-6,
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
                "--broker-username" => {
                    if i + 1 < args.len() {
                        self.broker_gateway.broker_username = Some(args[i + 1].clone());
                        i += 2;
                    } else {
                        return Err(anyhow::anyhow!("--broker-username requires a value"));
                    }
                }
                "--broker-password" => {
                    if i + 1 < args.len() {
                        self.broker_gateway.broker_password = Some(args[i + 1].clone());
                        i += 2;
                    } else {
                        return Err(anyhow::anyhow!("--broker-password requires a value"));
                    }
                }
                "--broker-account" => {
                    if i + 1 < args.len() {
                        self.broker_gateway.broker_account = Some(args[i + 1].clone());
                        i += 2;
                    } else {
                        return Err(anyhow::anyhow!("--broker-account requires a value"));
                    }
                }
                "--ctrader-username" => {
                    if i + 1 < args.len() {
                        self.broker_gateway.set_ctrader_gateway_if_unset();
                        self.broker_gateway.broker_username = Some(args[i + 1].clone());
                        i += 2;
                    } else {
                        return Err(anyhow::anyhow!("--ctrader-username requires a value"));
                    }
                }
                "--ctrader-password" => {
                    if i + 1 < args.len() {
                        self.broker_gateway.set_ctrader_gateway_if_unset();
                        self.broker_gateway.broker_password = Some(args[i + 1].clone());
                        i += 2;
                    } else {
                        return Err(anyhow::anyhow!("--ctrader-password requires a value"));
                    }
                }
                "--ctrader-account" => {
                    if i + 1 < args.len() {
                        self.broker_gateway.set_ctrader_gateway_if_unset();
                        self.broker_gateway.broker_account = Some(args[i + 1].clone());
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

        if let Some(ctrader_username) =
            Self::first_non_empty_env(env_get, &["CTRADER_USERNAME", "MODELENV_BROKER_USERNAME"])
        {
            self.broker_gateway.broker_username = Some(ctrader_username);
        }

        if let Some(ctrader_password) =
            Self::first_non_empty_env(env_get, &["CTRADER_PASSWORD", "MODELENV_BROKER_PASSWORD"])
        {
            self.broker_gateway.broker_password = Some(ctrader_password);
        }

        if let Some(ctrader_account) =
            Self::first_non_empty_env(env_get, &["CTRADER_ACCOUNT", "MODELENV_BROKER_ACCOUNT"])
        {
            self.broker_gateway.broker_account = Some(ctrader_account);
        }

        if Self::first_non_empty_env(
            env_get,
            &["CTRADER_USERNAME", "CTRADER_PASSWORD", "CTRADER_ACCOUNT"],
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

            if self.broker_gateway.username().is_none() {
                missing.push("--ctrader-username or CTRADER_USERNAME");
            }
            if self.broker_gateway.password().is_none() {
                missing.push("--ctrader-password or CTRADER_PASSWORD");
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
        info!("Symbol: {}", self.symbol);
        info!("Reward Lambda: {}", self.reward_lambda);
        info!("Reward Action Penalty: {}", self.reward_action_penalty);
        info!("Reward Holding Penalty: {}", self.reward_holding_penalty);

        if let Some(ref broker_gateway) = self.broker_gateway.broker_gateway {
            info!("Broker Gateway: {}", broker_gateway);
            if let Some(ref broker_addr) = self.broker_gateway.broker_addr {
                info!("Broker Address: {}", broker_addr);
            } else if self.broker_gateway.is_ctrader() {
                info!("Broker Address: default cTrader endpoint");
            }
            if let Some(ref broker_username) = self.broker_gateway.broker_username {
                if self.broker_gateway.is_ctrader() {
                    info!("cTrader Username: {}", broker_username);
                } else {
                    info!("Broker Username: {}", broker_username);
                }
            }
            if self.broker_gateway.broker_account.is_some() {
                if self.broker_gateway.is_ctrader() {
                    info!("cTrader Account: [set]");
                } else {
                    info!("Broker Account: [set]");
                }
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
    println!("  --s3-prefix <PREFIX>       S3 bucket prefix for training data (default: s3://modelenv-data)");
    println!("  --symbol <SYMBOL>          Trading symbol (default: USDJPY)");
    println!(
        "  --broker-gateway <TYPE>    Broker gateway type (e.g., 'ctrader', 'metatrader', 'ib')"
    );
    println!("  --broker-addr <ADDRESS>    Broker gateway address (host:port)");
    println!("  --broker-username <USER>   Broker gateway username");
    println!("  --broker-password <PASS>   Broker gateway password");
    println!("  --broker-account <ACCOUNT> Broker gateway account");
    println!("  --ctrader-username <USER>  cTrader API username");
    println!("  --ctrader-password <PASS>  cTrader API password");
    println!("  --ctrader-account <ACCOUNT> cTrader API account");
    println!("  --reward-lambda <LAMBDA>   Asymmetric drawdown penalty coefficient (default: 1.0)");
    println!("  --reward-action-penalty <C_A>    Action penalty coefficient (default: 0.001)");
    println!("  --reward-holding-penalty <C_H>   Holding penalty coefficient (default: 1e-6)");
    println!("  --help                     Display this help and exit");
    println!();
    println!("Environment Variables:");
    println!("  MODELENV_MODE              Same as --mode");
    println!("  MODELENV_ADDR              Same as --addr");
    println!("  MODELENV_S3_PREFIX         Same as --s3-prefix");
    println!("  MODELENV_SYMBOL            Same as --symbol");
    println!("  MODELENV_BROKER_GATEWAY    Same as --broker-gateway");
    println!("  MODELENV_BROKER_ADDR       Same as --broker-addr");
    println!("  MODELENV_BROKER_USERNAME   Same as --broker-username");
    println!("  MODELENV_BROKER_PASSWORD   Same as --broker-password");
    println!("  MODELENV_BROKER_ACCOUNT    Same as --broker-account");
    println!("  CTRADER_USERNAME           Same as --ctrader-username");
    println!("  CTRADER_PASSWORD           Same as --ctrader-password");
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
                ("CTRADER_USERNAME", "env-user"),
                ("CTRADER_PASSWORD", "env-pass"),
                ("CTRADER_ACCOUNT", "env-account"),
            ],
        )
        .unwrap();

        assert_eq!(
            config.broker_gateway.broker_gateway.as_deref(),
            Some("ctrader")
        );
        assert_eq!(
            config.broker_gateway.broker_username.as_deref(),
            Some("env-user")
        );
        assert_eq!(
            config.broker_gateway.broker_password.as_deref(),
            Some("env-pass")
        );
        assert_eq!(
            config.broker_gateway.broker_account.as_deref(),
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
                "--ctrader-username",
                "cli-user",
                "--ctrader-password",
                "cli-pass",
                "--ctrader-account",
                "cli-account",
            ],
            &[
                ("CTRADER_USERNAME", "env-user"),
                ("CTRADER_PASSWORD", "env-pass"),
                ("CTRADER_ACCOUNT", "env-account"),
            ],
        )
        .unwrap();

        assert_eq!(
            config.broker_gateway.broker_username.as_deref(),
            Some("cli-user")
        );
        assert_eq!(
            config.broker_gateway.broker_password.as_deref(),
            Some("cli-pass")
        );
        assert_eq!(
            config.broker_gateway.broker_account.as_deref(),
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
        assert!(err.to_string().contains("CTRADER_USERNAME"));
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
                "--ctrader-username",
                "user",
                "--ctrader-password",
                "pass",
                "--ctrader-account",
                "account",
            ],
            &[],
        )
        .unwrap();

        assert!(config.is_broker_gateway_configured());
    }
}
