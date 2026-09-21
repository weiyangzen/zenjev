use serde::Deserialize;
use std::path::PathBuf;

/// Transport-neutral bridge configuration. Python renders this JSON from the
/// authoritative YAML `mq` section plus the active schema/policy digests so the
/// bridge never parses a broker-specific or user-authored document itself.
#[derive(Debug, Clone, Deserialize)]
pub struct BridgeConfig {
    pub adapter: String,
    #[serde(default)]
    pub endpoints: Vec<String>,
    #[serde(default)]
    pub stream: String,
    #[serde(default)]
    pub subject: String,
    #[serde(default)]
    pub durable_name: String,
    #[serde(default)]
    pub consumer_group: Option<String>,
    #[serde(default = "default_start_position")]
    pub start_position: String,
    #[serde(default)]
    pub start_offset: Option<u64>,
    #[serde(default)]
    pub start_timestamp: Option<i64>,
    #[serde(default = "default_batch")]
    pub batch: usize,
    #[serde(default = "default_max_bytes")]
    pub max_bytes: usize,
    #[serde(default = "default_ack_wait")]
    pub ack_wait_seconds: u64,
    #[serde(default = "default_max_ack_pending")]
    pub max_ack_pending: usize,
    #[serde(default = "default_prefetch")]
    pub prefetch: usize,
    #[serde(default = "default_max_deliver")]
    pub max_deliver: u32,
    #[serde(default)]
    pub dlq_subject: String,
    #[serde(default)]
    pub dlq_path: Option<PathBuf>,
    #[serde(default)]
    pub quarantine_path: PathBuf,
    #[serde(default)]
    pub wal_path: PathBuf,
    #[serde(default)]
    pub socket_path: PathBuf,
    #[serde(default = "default_dedup_window")]
    pub dedup_window: usize,
    /// Mock-only infinite testing mode: after the last record, wrap to record 0
    /// and replay the same dataset with a distinct `loop_cycle` per pass.
    #[serde(default, rename = "loop")]
    pub loop_enabled: bool,
    /// Stop the mock loop after this many cycles; `0` means unbounded.
    #[serde(default)]
    pub max_cycles: u64,
    #[serde(default)]
    pub tls_required: bool,
    #[serde(default)]
    pub ca_cert: Option<PathBuf>,
    #[serde(default)]
    pub client_cert: Option<PathBuf>,
    #[serde(default)]
    pub client_key: Option<PathBuf>,
    /// Secret reference (for example `env:JEV_MQ_PASSWORD`). Never a value.
    #[serde(default)]
    pub secret_ref: Option<String>,
    pub schema_id: String,
    pub schema_version: u32,
    pub schema_digest: String,
    #[serde(default)]
    pub allowed_models: Vec<String>,
    #[serde(default)]
    pub metrics_path: Option<PathBuf>,
    #[serde(default)]
    pub state_path: Option<PathBuf>,
    #[serde(default)]
    pub source_path: Option<PathBuf>,
    #[serde(default)]
    pub faults: FaultConfig,
    /// Exit after the client drains and disconnects. Used by smoke/tests.
    #[serde(default)]
    pub exit_after_drain: bool,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct FaultConfig {
    #[serde(default)]
    pub duplicate_every: u64,
    #[serde(default)]
    pub poison_every: u64,
    #[serde(default)]
    pub oversize_every: u64,
}

fn default_start_position() -> String {
    "new".to_string()
}
fn default_batch() -> usize {
    64
}
fn default_max_bytes() -> usize {
    32 * 1024 * 1024
}
fn default_ack_wait() -> u64 {
    30
}
fn default_max_ack_pending() -> usize {
    16
}
fn default_prefetch() -> usize {
    32
}
fn default_max_deliver() -> u32 {
    5
}
fn default_dedup_window() -> usize {
    10_000
}

pub const KNOWN_ADAPTERS: [&str; 4] = ["mock", "nats-jetstream", "kafka", "iggy"];

impl BridgeConfig {
    pub fn from_path(path: &std::path::Path) -> Result<Self, String> {
        let raw = std::fs::read_to_string(path)
            .map_err(|error| format!("cannot read bridge config {}: {error}", path.display()))?;
        let config: BridgeConfig = serde_json::from_str(&raw)
            .map_err(|error| format!("invalid bridge config JSON: {error}"))?;
        config.validate()?;
        Ok(config)
    }

    /// Fail closed before any connection or launch side effect.
    pub fn validate(&self) -> Result<(), String> {
        if !KNOWN_ADAPTERS.contains(&self.adapter.as_str()) {
            return Err(format!("unknown adapter kind: {}", self.adapter));
        }
        if self.schema_id.is_empty() || self.schema_version == 0 || !is_sha256(&self.schema_digest)
        {
            return Err("schema_id/schema_version/schema_digest are required".to_string());
        }
        if self.batch == 0 || self.max_bytes == 0 || self.max_ack_pending == 0 || self.prefetch == 0
        {
            return Err("batch/max_bytes/max_ack_pending/prefetch must be positive".to_string());
        }
        if self.max_deliver == 0 {
            return Err("max_deliver must be at least 1".to_string());
        }
        if let Some(reference) = &self.secret_ref {
            if !(reference.starts_with("env:") || reference.starts_with("file:")) {
                return Err(
                    "secret_ref must be an env:/file: reference, never a secret value".to_string(),
                );
            }
        }
        if self.socket_path.as_os_str().is_empty() || self.wal_path.as_os_str().is_empty() {
            return Err("socket_path and wal_path are required".to_string());
        }
        if self.tls_required {
            if let (Some(cert), Some(key)) = (&self.client_cert, &self.client_key) {
                if cert.as_os_str().is_empty() || key.as_os_str().is_empty() {
                    return Err("client_cert/client_key must be real paths when set".to_string());
                }
            }
        }
        if self.adapter == "nats-jetstream" {
            if self.endpoints.is_empty() {
                return Err("nats-jetstream requires at least one endpoint".to_string());
            }
            if self.stream.is_empty() || self.durable_name.is_empty() {
                return Err("nats-jetstream requires stream and durable_name".to_string());
            }
            for endpoint in &self.endpoints {
                if is_remote(endpoint) && !self.tls_required {
                    return Err(format!("non-TLS remote endpoint fails closed: {endpoint}"));
                }
            }
        }
        if self.adapter == "kafka" && self.consumer_group.is_none() {
            return Err("kafka requires consumer_group".to_string());
        }
        Ok(())
    }

    pub fn endpoint_identity(&self) -> String {
        if self.endpoints.is_empty() {
            "local".to_string()
        } else {
            self.endpoints.join(",")
        }
    }
}

pub fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn is_remote(endpoint: &str) -> bool {
    let host = endpoint
        .split("://")
        .nth(1)
        .unwrap_or(endpoint)
        .split([':', '/'])
        .next()
        .unwrap_or("")
        .to_ascii_lowercase();
    !(host == "localhost"
        || host == "127.0.0.1"
        || host == "::1"
        || host == "[::1]"
        || host.is_empty())
}
