use crate::config::BridgeConfig;

pub mod mock;
#[cfg(feature = "nats")]
pub mod nats;

#[derive(Debug, Clone)]
pub struct Delivery {
    pub id: String,
    pub offset: u64,
    pub attempts: u32,
    pub raw: Vec<u8>,
}

#[derive(Debug, Clone, Default)]
pub struct SourceStatus {
    pub lag: i64,
    pub checkpoint: Option<u64>,
    /// Highest mock loop cycle index delivered so far (0 when not looping).
    pub loop_cycles: u64,
}

/// One pluggable broker protocol. The bridge exposes the same envelope/ack
/// contract to the trainer regardless of the configured adapter.
pub trait Source: Send {
    fn adapter(&self) -> &'static str;
    fn next(&mut self, max: usize) -> Result<Vec<Delivery>, String>;
    /// Ack/commit only after the downstream trainer confirmed durable
    /// acceptance. Never call this when records are merely sent.
    fn ack(&mut self, delivery: &Delivery) -> Result<(), String>;
    fn dead_letter(&mut self, raw: &[u8], reason: &str, attempts: u32) -> Result<(), String>;
    fn quarantine(&mut self, raw: &[u8], reason: &str, attempts: u32) -> Result<(), String>;
    fn status(&self) -> SourceStatus;
    fn shutdown(&mut self);
}

pub fn open_source(config: &BridgeConfig) -> Result<Box<dyn Source>, String> {
    match config.adapter.as_str() {
        "mock" => Ok(Box::new(mock::MockSource::open(config)?)),
        "nats-jetstream" => open_nats(config),
        "kafka" => Err(
            "adapter kafka requires a build with --features kafka and system librdkafka/cmake; \
             this binary was built without it and fails closed"
                .to_string(),
        ),
        "iggy" => Err(
            "adapter iggy requires a build with --features iggy; this binary was built without it \
             and fails closed"
                .to_string(),
        ),
        other => Err(format!("unknown adapter kind: {other}")),
    }
}

#[cfg(feature = "nats")]
fn open_nats(config: &BridgeConfig) -> Result<Box<dyn Source>, String> {
    Ok(Box::new(nats::NatsSource::open(config)?))
}

#[cfg(not(feature = "nats"))]
fn open_nats(_config: &BridgeConfig) -> Result<Box<dyn Source>, String> {
    Err("adapter nats-jetstream requires a build with --features nats".to_string())
}
