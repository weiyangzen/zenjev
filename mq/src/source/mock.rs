use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

use super::{Delivery, Source, SourceStatus};
use crate::canonical::sha256_hex;
use crate::config::BridgeConfig;
use crate::metrics::append_jsonl;

/// File-backed deterministic source used by smoke tests and fault injection.
///
/// Committed offset is persisted, so restarting the bridge resumes exactly
/// where the last downstream durable ack happened. Records sent but not acked
/// are redelivered, which is how the at-least-once replay path is exercised
/// without a live broker.
pub struct MockSource {
    records: Vec<Vec<u8>>,
    committed: u64,
    cursor: u64,
    state_path: PathBuf,
    dlq_path: PathBuf,
    quarantine_path: PathBuf,
    max_bytes: usize,
    faults: crate::config::FaultConfig,
    attempts: HashMap<String, u32>,
    duplicated: HashMap<u64, bool>,
}

#[derive(Debug, Serialize, Deserialize)]
struct MockState {
    committed: u64,
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
        if config.start_position != "first" {
            if let Ok(raw) = std::fs::read_to_string(&state_path) {
                if let Ok(state) = serde_json::from_str::<MockState>(&raw) {
                    committed = state.committed;
                }
            }
            if let Some(offset) = config.start_offset {
                committed = offset;
            }
        }
        committed = committed.min(records.len() as u64);
        Ok(Self {
            records,
            committed,
            cursor: committed,
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
        while batch.len() < max && (self.cursor as usize) < self.records.len() {
            let offset = self.cursor;
            let line_number = offset + 1;
            let raw = self.records[offset as usize].clone();
            let mut send_raw = raw.clone();
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
                && !self.duplicated.get(&offset).copied().unwrap_or(false)
            {
                self.duplicated.insert(offset, true);
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
        if delivery.offset + 1 > self.committed {
            self.committed = delivery.offset + 1;
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
        SourceStatus {
            lag: self.records.len() as i64 - self.committed as i64,
            checkpoint: Some(self.committed),
        }
    }

    fn shutdown(&mut self) {}
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
}
