// Configuration module for FX RL Model Environment
use anyhow::Result;
use std::env;

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
        let mut config = Config::new();

        // Parse command-line arguments
        let args: Vec<String> = env::args().collect();

        // Parse arguments manually (no external crate dependency)
        let mut i = 1; // Skip program name
        while i < args.len() {
            match args[i].as_str() {
                "--mode" => {
                    if i + 1 < args.len() {
                        config.mode = Mode::from_str(&args[i + 1])?;
                        i += 2;
                    } else {
                        return Err(anyhow::anyhow!("--mode requires a value"));
                    }
                }
                "--addr" => {
                    if i + 1 < args.len() {
                        config.addr = args[i + 1].clone();
                        i += 2;
                    } else {
                        return Err(anyhow::anyhow!("--addr requires a value"));
                    }
                }
                "--s3-prefix" => {
                    if i + 1 < args.len() {
                        config.s3_prefix = args[i + 1].clone();
                        i += 2;
                    } else {
                        return Err(anyhow::anyhow!("--s3-prefix requires a value"));
                    }
                }
                "--symbol" => {
                    if i + 1 < args.len() {
                        config.symbol = args[i + 1].clone();
                        i += 2;
                    } else {
                        return Err(anyhow::anyhow!("--symbol requires a value"));
                    }
                }
                "--broker-gateway" => {
                    if i + 1 < args.len() {
                        config.broker_gateway.broker_gateway = Some(args[i + 1].clone());
                        i += 2;
                    } else {
                        return Err(anyhow::anyhow!("--broker-gateway requires a value"));
                    }
                }
                "--broker-addr" => {
                    if i + 1 < args.len() {
                        config.broker_gateway.broker_addr = Some(args[i + 1].clone());
                        i += 2;
                    } else {
                        return Err(anyhow::anyhow!("--broker-addr requires a value"));
                    }
                }
                "--broker-username" => {
                    if i + 1 < args.len() {
                        config.broker_gateway.broker_username = Some(args[i + 1].clone());
                        i += 2;
                    } else {
                        return Err(anyhow::anyhow!("--broker-username requires a value"));
                    }
                }
                "--broker-password" => {
                    if i + 1 < args.len() {
                        config.broker_gateway.broker_password = Some(args[i + 1].clone());
                        i += 2;
                    } else {
                        return Err(anyhow::anyhow!("--broker-password requires a value"));
                    }
                }
                "--broker-account" => {
                    if i + 1 < args.len() {
                        config.broker_gateway.broker_account = Some(args[i + 1].clone());
                        i += 2;
                    } else {
                        return Err(anyhow::anyhow!("--broker-account requires a value"));
                    }
                }
                "--reward-lambda" => {
                    if i + 1 < args.len() {
                        config.reward_lambda = args[i + 1].parse()?;
                        i += 2;
                    } else {
                        return Err(anyhow::anyhow!("--reward-lambda requires a value"));
                    }
                }
                "--reward-action-penalty" => {
                    if i + 1 < args.len() {
                        config.reward_action_penalty = args[i + 1].parse()?;
                        i += 2;
                    } else {
                        return Err(anyhow::anyhow!("--reward-action-penalty requires a value"));
                    }
                }
                "--reward-holding-penalty" => {
                    if i + 1 < args.len() {
                        config.reward_holding_penalty = args[i + 1].parse()?;
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

        // Apply environment variable defaults
        config.apply_env_defaults();

        Ok(config)
    }

    /// Apply environment variable defaults for configuration
    fn apply_env_defaults(&mut self) {
        // Mode
        if let Ok(mode_env) = env::var("MODELENV_MODE") {
            if !mode_env.is_empty() {
                if let Ok(mode) = Mode::from_str(&mode_env) {
                    self.mode = mode;
                }
            }
        }

        // Address
        if let Ok(addr_env) = env::var("MODELENV_ADDR") {
            if !addr_env.is_empty() {
                self.addr = addr_env;
            }
        }

        // S3 prefix
        if let Ok(s3_prefix_env) = env::var("MODELENV_S3_PREFIX") {
            if !s3_prefix_env.is_empty() {
                self.s3_prefix = s3_prefix_env;
            }
        }

        // Symbol
        if let Ok(symbol_env) = env::var("MODELENV_SYMBOL") {
            if !symbol_env.is_empty() {
                self.symbol = symbol_env;
            }
        }

        // Broker gateway configuration
        if let Ok(broker_gateway_env) = env::var("MODELENV_BROKER_GATEWAY") {
            if !broker_gateway_env.is_empty() {
                self.broker_gateway.broker_gateway = Some(broker_gateway_env);
            }
        }

        if let Ok(broker_addr_env) = env::var("MODELENV_BROKER_ADDR") {
            if !broker_addr_env.is_empty() {
                self.broker_gateway.broker_addr = Some(broker_addr_env);
            }
        }

        if let Ok(broker_username_env) = env::var("MODELENV_BROKER_USERNAME") {
            if !broker_username_env.is_empty() {
                self.broker_gateway.broker_username = Some(broker_username_env);
            }
        }

        if let Ok(broker_password_env) = env::var("MODELENV_BROKER_PASSWORD") {
            if !broker_password_env.is_empty() {
                self.broker_gateway.broker_password = Some(broker_password_env);
            }
        }

        if let Ok(broker_account_env) = env::var("MODELENV_BROKER_ACCOUNT") {
            if !broker_account_env.is_empty() {
                self.broker_gateway.broker_account = Some(broker_account_env);
            }
        }

        // Reward function configuration
        if let Ok(reward_lambda_env) = env::var("MODELENV_REWARD_LAMBDA") {
            if !reward_lambda_env.is_empty() {
                if let Ok(value) = reward_lambda_env.parse::<f64>() {
                    self.reward_lambda = value;
                }
            }
        }

        if let Ok(reward_action_penalty_env) = env::var("MODELENV_REWARD_ACTION_PENALTY") {
            if !reward_action_penalty_env.is_empty() {
                if let Ok(value) = reward_action_penalty_env.parse::<f64>() {
                    self.reward_action_penalty = value;
                }
            }
        }

        if let Ok(reward_holding_penalty_env) = env::var("MODELENV_REWARD_HOLDING_PENALTY") {
            if !reward_holding_penalty_env.is_empty() {
                if let Ok(value) = reward_holding_penalty_env.parse::<f64>() {
                    self.reward_holding_penalty = value;
                }
            }
        }
    }

    /// Check if broker gateway is configured
    pub fn is_broker_gateway_configured(&self) -> bool {
        self.broker_gateway.broker_gateway.is_some()
            && self.broker_gateway.broker_addr.is_some()
    }

    /// Log the current configuration
    pub fn log(&self) {
        println!("Starting FX RL Model Environment");
        println!("Mode: {}", self.mode.as_str());
        println!("Address: {}", self.addr);
        println!("S3 Prefix: {}", self.s3_prefix);
        println!("Symbol: {}", self.symbol);
        println!("Reward Lambda: {}", self.reward_lambda);
        println!("Reward Action Penalty: {}", self.reward_action_penalty);
        println!("Reward Holding Penalty: {}", self.reward_holding_penalty);

        if let Some(ref broker_gateway) = self.broker_gateway.broker_gateway {
            println!("Broker Gateway: {}", broker_gateway);
            if let Some(ref broker_addr) = self.broker_gateway.broker_addr {
                println!("Broker Address: {}", broker_addr);
            }
            if let Some(ref broker_username) = self.broker_gateway.broker_username {
                println!("Broker Username: {}", broker_username);
            }
            if self.broker_gateway.broker_account.is_some() {
                println!("Broker Account: [set]");
            }
        } else {
            println!("Broker Gateway: not configured");
        }
    }
}

fn print_help() {
    println!("FX RL Model Environment & Gateway");
    println!();
    println!("Usage: modelenv-server [OPTIONS]");
    println!();
    println!("Options:");
    println!("  --mode <MODE>              Operating mode: 'training' or 'live' (default: training)");
    println!("  --addr <ADDRESS>           gRPC server address (default: 0.0.0.0:50051)");
    println!("  --s3-prefix <PREFIX>       S3 bucket prefix for training data (default: s3://modelenv-data)");
    println!("  --symbol <SYMBOL>          Trading symbol (default: USDJPY)");
    println!("  --broker-gateway <TYPE>    Broker gateway type (e.g., 'ctrader', 'metatrader', 'ib')");
    println!("  --broker-addr <ADDRESS>    Broker gateway address (host:port)");
    println!("  --broker-username <USER>   Broker gateway username");
    println!("  --broker-password <PASS>   Broker gateway password");
    println!("  --broker-account <ACCOUNT> Broker gateway account");
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
    println!("  MODELENV_REWARD_LAMBDA     Same as --reward-lambda");
    println!("  MODELENV_REWARD_ACTION_PENALTY   Same as --reward-action-penalty");
    println!("  MODELENV_REWARD_HOLDING_PENALTY  Same as --reward-holding-penalty");
}
