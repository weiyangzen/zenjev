use serde::Serialize;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Default, Serialize, Clone)]
pub struct Metrics {
    pub adapter: String,
    pub bridge_build: String,
    pub endpoint_identity: String,
    pub stream: String,
    pub subject: String,
    pub durable_name: String,
    pub schema_digest: String,
    pub pulled: u64,
    pub delivered: u64,
    pub acked: u64,
    pub duplicates_suppressed: u64,
    pub dlq: u64,
    pub quarantined: u64,
    pub rejected_by_client: u64,
    pub inflight: u64,
    pub credits: i64,
    pub wal_records: u64,
    pub wal_bytes: u64,
    pub consumer_lag: i64,
    /// Highest mock loop cycle index delivered so far (0 when not looping).
    pub loop_cycles: u64,
    pub ack_latency_ms: Latency,
    pub offset_checkpoint: Option<u64>,
    pub last_error_class: Option<String>,
    pub started_at_unix_ms: u128,
    pub updated_at_unix_ms: u128,
}

#[derive(Debug, Default, Serialize, Clone)]
pub struct Latency {
    pub count: u64,
    pub p50: f64,
    pub p95: f64,
    pub max: f64,
}

#[derive(Debug, Default)]
pub struct LatencyTracker {
    samples: Vec<f64>,
}

impl LatencyTracker {
    pub fn observe(&mut self, value_ms: f64) {
        self.samples.push(value_ms.max(0.0));
        if self.samples.len() > 4096 {
            self.samples.drain(0..2048);
        }
    }

    pub fn snapshot(&self) -> Latency {
        if self.samples.is_empty() {
            return Latency::default();
        }
        let mut sorted = self.samples.clone();
        sorted.sort_by(f64::total_cmp);
        let index = |fraction: f64| sorted[((sorted.len() - 1) as f64 * fraction) as usize];
        Latency {
            count: sorted.len() as u64,
            p50: index(0.50),
            p95: index(0.95),
            max: *sorted.last().expect("non-empty"),
        }
    }
}

pub fn unix_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis())
        .unwrap_or_default()
}

impl Metrics {
    pub fn new(
        adapter: &str,
        endpoint: &str,
        stream: &str,
        subject: &str,
        durable: &str,
        schema_digest: &str,
    ) -> Self {
        Self {
            adapter: adapter.to_string(),
            bridge_build: format!("jev-mq-bridge/{}", env!("CARGO_PKG_VERSION")),
            endpoint_identity: endpoint.to_string(),
            stream: stream.to_string(),
            subject: subject.to_string(),
            durable_name: durable.to_string(),
            schema_digest: schema_digest.to_string(),
            started_at_unix_ms: unix_ms(),
            updated_at_unix_ms: unix_ms(),
            ..Default::default()
        }
    }

    pub fn write(&self, path: Option<&PathBuf>) {
        let Some(path) = path else { return };
        let mut snapshot = self.clone();
        snapshot.updated_at_unix_ms = unix_ms();
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let temporary = path.with_extension("tmp");
        if let Ok(body) = serde_json::to_string_pretty(&snapshot) {
            if std::fs::write(&temporary, body + "\n").is_ok() {
                let _ = std::fs::rename(&temporary, path);
            }
        }
    }
}

pub fn append_jsonl(path: &Path, value: &serde_json::Value) {
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let Ok(mut body) = serde_json::to_string(value) else {
        return;
    };
    body.push('\n');
    use std::io::Write;
    if let Ok(mut file) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
    {
        let _ = file.write_all(body.as_bytes());
        let _ = file.sync_data();
    }
}
