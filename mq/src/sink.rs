use serde_json::{json, Value};
use std::collections::HashMap;
use std::io::{Read, Write};
use std::os::unix::fs::PermissionsExt;
use std::os::unix::net::{UnixListener, UnixStream};
use std::sync::mpsc::{channel, RecvTimeoutError, Sender};
use std::time::{Duration, Instant};

use crate::config::BridgeConfig;
use crate::dedup::DedupLedger;
use crate::envelope::{self, Envelope, Rejection};
use crate::metrics::{unix_ms, LatencyTracker, Metrics};
use crate::source::{Delivery, Source};
use crate::wal::{Wal, WalFull};

const MAX_FRAME: usize = 64 * 1024 * 1024;
const PROTOCOL_VERSION: u64 = 1;

struct Inflight {
    delivery: Delivery,
    key: String,
    raw: Vec<u8>,
    sent_at: Instant,
}

pub struct Sink {
    config: BridgeConfig,
    source: Box<dyn Source>,
    wal: Wal,
    dedup: DedupLedger,
    metrics: Metrics,
    latency: LatencyTracker,
}

impl Sink {
    pub fn open(config: BridgeConfig) -> Result<Self, String> {
        config.validate()?;
        let source = crate::source::open_source(&config)?;
        let mut wal = Wal::open(&config.wal_path, config.max_bytes)?;
        let ledger_path = config.wal_path.with_extension("dedup");
        let dedup = DedupLedger::open(&ledger_path, config.dedup_window)?;
        let replayed = wal
            .is_empty()
            .then(Vec::new)
            .unwrap_or_else(|| wal.pending().to_vec());
        for record in replayed {
            if let Some(key) = raw_idempotency_key(&record) {
                if dedup.contains(&key) {
                    // Broker redelivered a record the trainer already durably
                    // accepted before a previous shutdown.
                    wal.acknowledge(&record);
                }
            }
        }
        let metrics = Metrics::new(
            source.adapter(),
            &config.endpoint_identity(),
            &config.stream,
            &config.subject,
            &config.durable_name,
            &config.schema_digest,
        );
        Ok(Self {
            config,
            source,
            wal,
            dedup,
            metrics,
            latency: LatencyTracker::default(),
        })
    }

    pub fn metrics(&self) -> &Metrics {
        &self.metrics
    }

    pub fn run(&mut self) -> Result<(), String> {
        let socket_path = self.config.socket_path.clone();
        if let Some(parent) = socket_path.parent() {
            std::fs::create_dir_all(parent).map_err(|error| error.to_string())?;
        }
        if socket_path.exists() {
            std::fs::remove_file(&socket_path).map_err(|error| error.to_string())?;
        }
        let listener = UnixListener::bind(&socket_path)
            .map_err(|error| format!("cannot bind bridge socket: {error}"))?;
        std::fs::set_permissions(&socket_path, std::fs::Permissions::from_mode(0o600))
            .map_err(|error| format!("cannot restrict bridge socket: {error}"))?;
        println!(
            "{}",
            json!({
                "event": "listening",
                "socket": socket_path,
                "adapter": self.source.adapter(),
                "schema_digest": self.config.schema_digest,
                "protocol": PROTOCOL_VERSION,
            })
        );
        let _ = std::io::stdout().flush();
        loop {
            let (stream, _) = listener
                .accept()
                .map_err(|error| format!("bridge accept failed: {error}"))?;
            let drained = self.handle_connection(stream)?;
            self.refresh_metrics();
            self.metrics.write(self.config.metrics_path.as_ref());
            if drained && self.config.exit_after_drain {
                break;
            }
        }
        self.source.shutdown();
        let _ = std::fs::remove_file(&socket_path);
        Ok(())
    }

    fn handle_connection(&mut self, mut stream: UnixStream) -> Result<bool, String> {
        stream
            .set_read_timeout(None)
            .map_err(|error| error.to_string())?;
        let (messages_tx, messages_rx) = channel::<Option<Value>>();
        let mut reader = stream
            .try_clone()
            .map_err(|error| format!("cannot clone bridge socket: {error}"))?;
        let reader_thread = std::thread::Builder::new()
            .name("jev-mq-client-reader".to_string())
            .spawn(move || read_loop(&mut reader, messages_tx))
            .map_err(|error| error.to_string())?;

        let mut credits: i64 = 0;
        let mut inflight: HashMap<String, Inflight> = HashMap::new();
        let mut paused = false;
        let mut draining = false;
        let mut drain_sent = false;
        let mut handshake = false;
        let mut done = false;
        let mut result = Ok(false);

        while !done {
            self.metrics.inflight = inflight.len() as u64;
            self.metrics.credits = credits;
            self.metrics.wal_records = self.wal.len() as u64;
            self.metrics.wal_bytes = self.wal.bytes() as u64;
            self.metrics.ack_latency_ms = self.latency.snapshot();

            if handshake && !paused && !draining {
                let capacity = (self.config.max_ack_pending - inflight.len()) as i64;
                let wanted = credits.min(capacity).min(self.config.batch as i64);
                if wanted > 0 && self.wal.bytes() * 2 < self.config.max_bytes {
                    match self.source.next(wanted as usize) {
                        Ok(batch) => {
                            self.metrics.pulled += batch.len() as u64;
                            for delivery in batch {
                                if let Some(key) = raw_idempotency_key(&delivery.raw) {
                                    if self.dedup.contains(&key) {
                                        self.metrics.duplicates_suppressed += 1;
                                        let _ = self.source.ack(&delivery);
                                        continue;
                                    }
                                }
                                if delivery.attempts > self.config.max_deliver {
                                    self.metrics.dlq += 1;
                                    self.metrics.last_error_class =
                                        Some("retry_ceiling".to_string());
                                    let _ = self.source.dead_letter(
                                        &delivery.raw,
                                        "retry_ceiling",
                                        delivery.attempts,
                                    );
                                    let _ = self.source.ack(&delivery);
                                    continue;
                                }
                                match envelope::validate(&delivery.raw, &self.config) {
                                    Err(rejection @ Rejection::DeadLetter(_)) => {
                                        self.metrics.dlq += 1;
                                        self.metrics.last_error_class = Some(format!(
                                            "{}:{}",
                                            rejection.class(),
                                            rejection.reason()
                                        ));
                                        let _ = self.source.dead_letter(
                                            &delivery.raw,
                                            rejection.reason(),
                                            delivery.attempts,
                                        );
                                        let _ = self.source.ack(&delivery);
                                    }
                                    Err(rejection @ Rejection::Quarantine(_)) => {
                                        self.metrics.quarantined += 1;
                                        let _ = self.source.quarantine(
                                            &delivery.raw,
                                            rejection.reason(),
                                            delivery.attempts,
                                        );
                                        let _ = self.source.ack(&delivery);
                                    }
                                    Ok(parsed) => {
                                        let key = parsed.idempotency_key();
                                        match self.wal.append(&delivery.raw) {
                                            Ok(()) => {}
                                            Err(WalFull) => {
                                                // Bounded spool backpressure: the
                                                // unacked record stays at the broker.
                                                credits = 0;
                                                self.metrics.last_error_class =
                                                    Some("wal_full".to_string());
                                                break;
                                            }
                                        }
                                        let _ = send_record(&mut stream, &delivery, &parsed);
                                        self.metrics.delivered += 1;
                                        credits -= 1;
                                        inflight.insert(
                                            delivery.id.clone(),
                                            Inflight {
                                                key: key.clone(),
                                                raw: delivery.raw.clone(),
                                                sent_at: Instant::now(),
                                                delivery,
                                            },
                                        );
                                    }
                                }
                            }
                        }
                        Err(error) => {
                            self.metrics.last_error_class = Some(error);
                        }
                    }
                }
            }

            match messages_rx.recv_timeout(Duration::from_millis(20)) {
                Ok(Some(message)) => {
                    let message_type = message
                        .get("type")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_string();
                    match message_type.as_str() {
                        "hello" => {
                            if message.get("protocol").and_then(Value::as_u64)
                                != Some(PROTOCOL_VERSION)
                            {
                                let _ = send_error(&mut stream, "protocol_mismatch");
                                done = true;
                                result = Err("client protocol mismatch".to_string());
                                continue;
                            }
                            credits = message
                                .get("credits")
                                .and_then(Value::as_i64)
                                .unwrap_or(0)
                                .clamp(0, self.config.max_ack_pending as i64);
                            handshake = true;
                            let _ = send_frame(
                                &mut stream,
                                &json!({
                                    "type": "ready",
                                    "protocol": PROTOCOL_VERSION,
                                    "bridge": format!("jev-mq-bridge/{}", env!("CARGO_PKG_VERSION")),
                                    "adapter": self.source.adapter(),
                                    "wal_records": self.wal.len(),
                                    "schema_digest": self.config.schema_digest,
                                }),
                            );
                        }
                        "credit" => {
                            let mut acknowledged = 0u64;
                            if let Some(acks) = message.get("acks").and_then(Value::as_array) {
                                for ack in acks.iter().filter_map(Value::as_str) {
                                    if let Some(item) = inflight.remove(ack) {
                                        self.wal.acknowledge(&item.raw);
                                        if let Err(error) = self.dedup.record(&item.key) {
                                            self.metrics.last_error_class = Some(error);
                                        }
                                        if let Err(error) = self.source.ack(&item.delivery) {
                                            self.metrics.last_error_class = Some(error);
                                        }
                                        let elapsed = item.sent_at.elapsed().as_secs_f64() * 1000.0;
                                        self.latency.observe(elapsed);
                                        acknowledged += 1;
                                    }
                                }
                            }
                            self.metrics.acked += acknowledged;
                            if let Some(quarantine) =
                                message.get("quarantine").and_then(Value::as_array)
                            {
                                for item in quarantine {
                                    let Some(id) = item.get("delivery_id").and_then(Value::as_str)
                                    else {
                                        continue;
                                    };
                                    let reason = item
                                        .get("reason")
                                        .and_then(Value::as_str)
                                        .unwrap_or("client_quarantine");
                                    if let Some(entry) = inflight.remove(id) {
                                        let _ = self.source.quarantine(
                                            &entry.raw,
                                            reason,
                                            entry.delivery.attempts,
                                        );
                                        let _ = self.source.ack(&entry.delivery);
                                        self.metrics.quarantined += 1;
                                        self.metrics.rejected_by_client += 1;
                                    }
                                }
                            }
                            let granted = message
                                .get("count")
                                .and_then(Value::as_i64)
                                .unwrap_or(0)
                                .max(0);
                            credits = (credits + granted).min(self.config.max_ack_pending as i64);
                        }
                        "pause" => paused = true,
                        "resume" => paused = false,
                        "drain" => {
                            draining = true;
                            if inflight.is_empty() && !drain_sent {
                                drain_sent = true;
                                let _ = send_frame(
                                    &mut stream,
                                    &json!({"type": "drained", "inflight": 0}),
                                );
                            }
                        }
                        "status" => {
                            let status = self.source.status();
                            self.metrics.consumer_lag = status.lag;
                            self.metrics.loop_cycles = status.loop_cycles;
                            self.metrics.offset_checkpoint = status.checkpoint;
                            let _ = send_frame(
                                &mut stream,
                                &json!({"type": "status", "metrics": serde_json::to_value(&self.metrics).unwrap_or_default()}),
                            );
                        }
                        "goodbye" => {
                            if draining || inflight.is_empty() {
                                done = true;
                                result = Ok(drain_sent || draining);
                            } else {
                                let _ = send_error(&mut stream, "goodbye_with_inflight");
                            }
                        }
                        other => {
                            let _ = send_error(&mut stream, &format!("unknown_message:{other}"));
                        }
                    }
                }
                Ok(None) => {
                    // Client disconnected. In-flight records stay unacked; the
                    // broker will redeliver them and the WAL replays at restart.
                    done = true;
                    result = Ok(drain_sent);
                }
                Err(RecvTimeoutError::Timeout) => {}
                Err(RecvTimeoutError::Disconnected) => {
                    done = true;
                    result = Ok(drain_sent);
                }
            }

            self.refresh_metrics();
            if draining && inflight.is_empty() && !drain_sent {
                drain_sent = true;
                let _ = send_frame(&mut stream, &json!({"type": "drained", "inflight": 0}));
                if self.config.exit_after_drain {
                    done = true;
                    result = Ok(true);
                }
            }
        }

        // Unblock the reader thread; the server side closes the connection once
        // the client said goodbye or drained.
        let _ = stream.shutdown(std::net::Shutdown::Both);
        drop(messages_rx);
        let _ = reader_thread.join();
        self.metrics.write(self.config.metrics_path.as_ref());
        result
    }

    fn refresh_metrics(&mut self) {
        self.metrics.updated_at_unix_ms = unix_ms();
        let status = self.source.status();
        self.metrics.consumer_lag = status.lag;
        self.metrics.loop_cycles = status.loop_cycles;
        if status.checkpoint.is_some() {
            self.metrics.offset_checkpoint = status.checkpoint;
        }
        self.metrics.ack_latency_ms = self.latency.snapshot();
    }
}

fn raw_idempotency_key(raw: &[u8]) -> Option<String> {
    let value: Value = serde_json::from_slice(raw).ok()?;
    let record_id = value.get("record_id")?.as_str()?;
    let content = value.get("content_sha256")?.as_str()?;
    Some(format!("{record_id}|{content}"))
}

fn send_record(
    stream: &mut UnixStream,
    delivery: &Delivery,
    parsed: &Envelope,
) -> std::io::Result<()> {
    send_frame(
        stream,
        &json!({
            "type": "record",
            "delivery_id": delivery.id,
            "offset": delivery.offset,
            "attempt": delivery.attempts,
            "envelope_version": parsed.envelope_version,
            "kind": parsed.kind,
            "observed_model": parsed.observed_model_id(),
            "observed_model_provider": parsed.observed_model_provider(),
            "schema_id": parsed.schema.id,
            "schema_version": parsed.schema.version,
            "source_uri": parsed.source.uri,
            "source_license": parsed.source.license,
            "loop_cycle": parsed.value.get("loop_cycle").and_then(Value::as_u64).unwrap_or(0),
            "envelope": parsed.value,
        }),
    )
}

fn send_error(stream: &mut UnixStream, reason: &str) -> std::io::Result<()> {
    send_frame(stream, &json!({"type": "error", "reason": reason}))
}

fn send_frame(stream: &mut UnixStream, value: &Value) -> std::io::Result<()> {
    let body = serde_json::to_vec(value)?;
    stream.write_all(&(body.len() as u32).to_be_bytes())?;
    stream.write_all(&body)?;
    stream.flush()
}

fn read_loop(stream: &mut UnixStream, sender: Sender<Option<Value>>) {
    loop {
        match read_frame(stream) {
            Ok(Some(value)) => {
                if sender.send(Some(value)).is_err() {
                    return;
                }
            }
            _ => {
                let _ = sender.send(None);
                return;
            }
        }
    }
}

fn read_frame(stream: &mut UnixStream) -> std::io::Result<Option<Value>> {
    let mut length_bytes = [0u8; 4];
    if let Err(error) = stream.read_exact(&mut length_bytes) {
        return if error.kind() == std::io::ErrorKind::UnexpectedEof {
            Ok(None)
        } else {
            Err(error)
        };
    }
    let length = u32::from_be_bytes(length_bytes) as usize;
    if length > MAX_FRAME {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "frame too large",
        ));
    }
    let mut body = vec![0u8; length];
    stream.read_exact(&mut body)?;
    let value = serde_json::from_slice(&body)
        .map_err(|error| std::io::Error::new(std::io::ErrorKind::InvalidData, error))?;
    Ok(Some(value))
}
