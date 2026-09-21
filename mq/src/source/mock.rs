use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

use super::{Delivery, Source, SourceStatus};
use crate::canonical::{envelope_content_hash, sha256_hex};
use crate::config::BridgeConfig;
use crate::metrics::append_jsonl;

/// File-backed deterministic source used by smoke tests and fault injection.
///
/// Committed offset is persisted, so restarting the bridge resumes exactly
/// where the last downstream durable ack happened. Records sent but not acked
/// are redelivered, which is how the at-least-once replay path is exercised
/// without a live broker.
///
/// With `loop: true` the same dataset replays head-to-tail forever (bounded by
/// `max_cycles` when positive). Every cycle after the first injects a
/// `loop_cycle` field into the envelope and recomputes the canonical
/// `content_sha256`, so each pass produces a distinct idempotency key while
/// cycle 0 stays byte-identical to the authored fixture.
pub struct MockSource {
    records: Vec<Vec<u8>>,
    committed: u64,
    cursor: u64,
    cycle: u64,
    loop_enabled: bool,
    max_cycles: u64,
    state_path: PathBuf,
    dlq_path: PathBuf,
    quarantine_path: PathBuf,
    max_bytes: usize,
    faults: crate::config::FaultConfig,
    attempts: HashMap<String, u32>,
    duplicated: HashMap<(u64, u64), bool>,
}

#[derive(Debug, Serialize, Deserialize)]
struct MockState {
    committed: u64,
    #[serde(default)]
    cycle: u64,
    #[serde(default)]
    cursor: Option<u64>,
}

impl MockSource {
    pub fn open(config: &BridgeConfig) -> Result<Self, String> {
        let source_path = config
            .source_path
            .as_ref()
            .ok_or_else(|| "mock adapter requires source_path".to_string())?;
        let records = read_records(source_path)?;
        let state_path = config
            .state_path
            .clone()
            .unwrap_or_else(|| PathBuf::from("runs/jev/mq/mock-state.json"));
        let mut committed = 0u64;
        let mut cycle = 0u64;
        let mut resumed_cursor: Option<u64> = None;
        if config.start_position != "first" {
            if let Ok(raw) = std::fs::read_to_string(&state_path) {
                if let Ok(state) = serde_json::from_str::<MockState>(&raw) {
                    committed = state.committed;
                    if config.loop_enabled {
                        cycle = state.cycle;
                        resumed_cursor = state.cursor;
                    }
                }
            }
            if let Some(offset) = config.start_offset {
                committed = offset;
                resumed_cursor = Some(offset);
            }
        }
        committed = committed.min(records.len() as u64);
        let cursor = if config.loop_enabled {
            resumed_cursor
                .unwrap_or(committed)
                .min(records.len() as u64)
        } else {
            committed
        };
        Ok(Self {
            records,
            committed,
            cursor,
            cycle,
            loop_enabled: config.loop_enabled,
            max_cycles: config.max_cycles,
            state_path,
            dlq_path: config
                .dlq_path
                .clone()
                .unwrap_or_else(|| PathBuf::from("runs/jev/mq/dlq.jsonl")),
            quarantine_path: config.quarantine_path.clone(),
            max_bytes: config.max_bytes,
            faults: config.faults.clone(),
            attempts: HashMap::new(),
            duplicated: HashMap::new(),
        })
    }

    fn persist_state(&self) -> Result<(), String> {
        if let Some(parent) = self.state_path.parent() {
            std::fs::create_dir_all(parent).map_err(|error| error.to_string())?;
        }
        let temporary = self.state_path.with_extension("tmp");
        let body = serde_json::to_string(&MockState {
            committed: self.committed,
            cycle: self.cycle,
            cursor: Some(self.cursor),
        })
        .map_err(|error| error.to_string())?;
        std::fs::write(&temporary, body).map_err(|error| error.to_string())?;
        std::fs::rename(&temporary, &self.state_path).map_err(|error| error.to_string())
    }
}

impl Source for MockSource {
    fn adapter(&self) -> &'static str {
        "mock"
    }

    fn next(&mut self, max: usize) -> Result<Vec<Delivery>, String> {
        let mut batch = Vec::new();
        while batch.len() < max {
            if self.records.is_empty() {
                break;
            }
            if self.cursor as usize >= self.records.len() {
                if !self.loop_enabled {
                    break;
                }
                if self.max_cycles > 0 && self.cycle + 1 >= self.max_cycles {
                    break;
                }
                self.cycle += 1;
                self.cursor = 0;
                self.persist_state()?;
            }
            let offset = self.cursor;
            let line_number = offset + 1;
            let raw = self.records[offset as usize].clone();
            let mut send_raw = raw.clone();
            if self.cycle > 0 {
                // Injection is best effort: a malformed fixture still reaches
                // the sink unchanged and is routed to the DLQ as before.
                if let Some(looped) = inject_loop_cycle(&raw, self.cycle) {
                    send_raw = looped;
                }
            }
            if self.faults.oversize_every > 0 && line_number % self.faults.oversize_every == 0 {
                send_raw = vec![b'x'; self.max_bytes + 1];
            } else if self.faults.poison_every > 0 && line_number % self.faults.poison_every == 0 {
                send_raw = b"{\"envelope_version\": 1, \"record_id\":".to_vec();
            }
            let id = format!("mock-{offset}-{}", &sha256_hex(&send_raw)[..12]);
            let attempts = self.attempts.entry(id.clone()).or_insert(0);
            let delivery = Delivery {
                id: id.clone(),
                offset,
                attempts: *attempts + 1,
                raw: send_raw.clone(),
            };
            batch.push(delivery);
            if self.faults.duplicate_every > 0
                && line_number % self.faults.duplicate_every == 0
                && !self
                    .duplicated
                    .get(&(self.cycle, offset))
                    .copied()
                    .unwrap_or(false)
            {
                self.duplicated.insert((self.cycle, offset), true);
                batch.push(Delivery {
                    id: format!("{id}-dup"),
                    offset,
                    attempts: 1,
                    raw: send_raw,
                });
            }
            self.cursor += 1;
        }
        Ok(batch)
    }

    fn ack(&mut self, delivery: &Delivery) -> Result<(), String> {
        let mut dirty = false;
        if delivery.offset + 1 > self.committed {
            self.committed = delivery.offset + 1;
            dirty = true;
        }
        if self.loop_enabled {
            // Best-effort cycle/cursor progress so a restart resumes near the
            // wrap point instead of replaying from the first cycle.
            dirty = true;
        }
        if dirty {
            self.persist_state()?;
        }
        self.attempts.remove(&delivery.id);
        Ok(())
    }

    fn dead_letter(&mut self, raw: &[u8], reason: &str, attempts: u32) -> Result<(), String> {
        append_jsonl(
            &self.dlq_path,
            &serde_json::json!({
                "class": "dlq",
                "reason": reason,
                "attempts": attempts,
                "raw_sha256": sha256_hex(raw),
                "raw_bytes": raw.len(),
                "raw_preview": String::from_utf8_lossy(&raw[..raw.len().min(256)]),
            }),
        );
        Ok(())
    }

    fn quarantine(&mut self, raw: &[u8], reason: &str, attempts: u32) -> Result<(), String> {
        append_jsonl(
            &self.quarantine_path,
            &serde_json::json!({
                "class": "quarantine",
                "reason": reason,
                "attempts": attempts,
                "raw_sha256": sha256_hex(raw),
                "raw_bytes": raw.len(),
            }),
        );
        Ok(())
    }

    fn status(&self) -> SourceStatus {
        let lag = if self.loop_enabled {
            (self.records.len() as i64 - self.cursor as i64).max(0)
        } else {
            self.records.len() as i64 - self.committed as i64
        };
        SourceStatus {
            lag,
            checkpoint: Some(self.committed),
            loop_cycles: self.cycle,
        }
    }

    fn shutdown(&mut self) {}
}

/// Add the cycle marker and recompute the canonical content hash so the
/// idempotency key differs per loop pass.
fn inject_loop_cycle(raw: &[u8], cycle: u64) -> Option<Vec<u8>> {
    let mut value: Value = serde_json::from_slice(raw).ok()?;
    value
        .as_object_mut()?
        .insert("loop_cycle".to_string(), Value::from(cycle));
    let hash = envelope_content_hash(&value).ok()?;
    value
        .as_object_mut()?
        .insert("content_sha256".to_string(), Value::String(hash));
    serde_json::to_vec(&value).ok()
}

pub fn read_records(path: &Path) -> Result<Vec<Vec<u8>>, String> {
    let raw = std::fs::read_to_string(path)
        .map_err(|error| format!("cannot read mock source {}: {error}", path.display()))?;
    Ok(raw
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| line.as_bytes().to_vec())
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::BridgeConfig;

    fn config_for(path: &Path, state: &Path) -> BridgeConfig {
        serde_json::from_value(serde_json::json!({
            "adapter": "mock",
            "source_path": path,
            "state_path": state,
            "socket_path": state.with_extension("sock"),
            "wal_path": state.with_extension("wal"),
            "quarantine_path": state.with_extension("quarantine"),
            "schema_id": "s",
            "schema_version": 1,
            "schema_digest": "a".repeat(64),
        }))
        .unwrap()
    }

    fn loop_config_for(path: &Path, state: &Path, max_cycles: u64) -> BridgeConfig {
        serde_json::from_value(serde_json::json!({
            "adapter": "mock",
            "source_path": path,
            "state_path": state,
            "socket_path": state.with_extension("sock"),
            "wal_path": state.with_extension("wal"),
            "quarantine_path": state.with_extension("quarantine"),
            "schema_id": "s",
            "schema_version": 1,
            "schema_digest": "a".repeat(64),
            "loop": true,
            "max_cycles": max_cycles,
        }))
        .unwrap()
    }

    fn fixture_envelope(record_id: &str) -> Vec<u8> {
        let mut value = serde_json::json!({
            "envelope_version": 1,
            "record_id": record_id,
            "content_sha256": "",
            "kind": "raw_document",
            "document": {"text": record_id},
        });
        value["content_sha256"] =
            Value::String(envelope_content_hash(&value).expect("fixture hash"));
        serde_json::to_vec(&value).unwrap()
    }

    #[test]
    fn resumes_from_persisted_commit_and_redelivers_unacked() {
        let directory = std::env::temp_dir().join(format!("jev-mock-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&directory);
        std::fs::create_dir_all(&directory).unwrap();
        let source_path = directory.join("records.jsonl");
        std::fs::write(&source_path, "{\"n\":1}\n{\"n\":2}\n{\"n\":3}\n").unwrap();
        let mut source =
            MockSource::open(&config_for(&source_path, &directory.join("state.json"))).unwrap();
        let batch = source.next(10).unwrap();
        assert_eq!(batch.len(), 3);
        source.ack(&batch[0]).unwrap();
        let mut reopened =
            MockSource::open(&config_for(&source_path, &directory.join("state.json"))).unwrap();
        let replay = reopened.next(10).unwrap();
        assert_eq!(replay.len(), 2);
        assert_eq!(replay[0].offset, 1);
        let _ = std::fs::remove_dir_all(&directory);
    }

    #[test]
    fn loop_wraps_with_distinct_hashes_and_stops_at_max_cycles() {
        let directory =
            std::env::temp_dir().join(format!("jev-mock-loop-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&directory);
        std::fs::create_dir_all(&directory).unwrap();
        let source_path = directory.join("records.jsonl");
        let first = fixture_envelope("r1");
        let second = fixture_envelope("r2");
        std::fs::write(
            &source_path,
            format!(
                "{}\n{}\n",
                String::from_utf8_lossy(&first),
                String::from_utf8_lossy(&second)
            ),
        )
        .unwrap();
        let state = directory.join("state.json");
        let mut source = MockSource::open(&loop_config_for(&source_path, &state, 3)).unwrap();
        let mut cycles = Vec::new();
        let mut hashes = Vec::new();
        loop {
            let batch = source.next(8).unwrap();
            if batch.is_empty() {
                break;
            }
            for delivery in &batch {
                let value: Value = serde_json::from_slice(&delivery.raw).unwrap();
                cycles.push(value.get("loop_cycle").and_then(Value::as_u64));
                hashes.push(value["content_sha256"].as_str().unwrap().to_string());
                source.ack(delivery).unwrap();
            }
        }
        assert_eq!(cycles, vec![None, None, Some(1), Some(1), Some(2), Some(2)]);
        assert_eq!(hashes.len(), 6);
        assert_ne!(hashes[0], hashes[2]);
        assert_ne!(hashes[2], hashes[4]);
        assert_eq!(source.status().loop_cycles, 2);
        // Restart resumes near the wrap point rather than the first cycle.
        let mut reopened = MockSource::open(&loop_config_for(&source_path, &state, 3)).unwrap();
        assert_eq!(reopened.status().loop_cycles, 2);
        assert!(reopened.next(8).unwrap().is_empty());
        let _ = std::fs::remove_dir_all(&directory);
    }
}
