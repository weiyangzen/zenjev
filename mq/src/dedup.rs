use std::collections::{HashSet, VecDeque};
use std::path::{Path, PathBuf};

/// Idempotency ledger keyed by `record_id|content_sha256`.
///
/// Redelivery is expected under the at-least-once contract. Accepted keys are
/// persisted so a bridge restart can still suppress a duplicate that the broker
/// redelivers.
pub struct DedupLedger {
    window: usize,
    order: VecDeque<String>,
    seen: HashSet<String>,
    path: PathBuf,
}

impl DedupLedger {
    pub fn open(path: &Path, window: usize) -> Result<Self, String> {
        if window == 0 {
            return Err("dedup window must be positive".to_string());
        }
        let mut order: VecDeque<String> = VecDeque::new();
        if path.exists() {
            let raw = std::fs::read_to_string(path)
                .map_err(|error| format!("cannot read dedup ledger: {error}"))?;
            let lines: Vec<&str> = raw.lines().filter(|line| !line.is_empty()).collect();
            for line in lines.iter().skip(lines.len().saturating_sub(window)) {
                order.push_back((*line).to_string());
            }
        }
        let seen = order.iter().cloned().collect();
        Ok(Self {
            window,
            order,
            seen,
            path: path.to_path_buf(),
        })
    }

    pub fn contains(&self, key: &str) -> bool {
        self.seen.contains(key)
    }

    pub fn record(&mut self, key: &str) -> Result<(), String> {
        if self.seen.insert(key.to_string()) {
            self.order.push_back(key.to_string());
            while self.order.len() > self.window {
                if let Some(evicted) = self.order.pop_front() {
                    self.seen.remove(&evicted);
                }
            }
            self.persist()?;
        }
        Ok(())
    }

    fn persist(&self) -> Result<(), String> {
        if let Some(parent) = self.path.parent() {
            std::fs::create_dir_all(parent).map_err(|error| error.to_string())?;
        }
        let temporary = self.path.with_extension("tmp");
        let body = self.order.iter().cloned().collect::<Vec<_>>().join("\n");
        std::fs::write(&temporary, body).map_err(|error| error.to_string())?;
        std::fs::rename(&temporary, &self.path).map_err(|error| error.to_string())?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn window_evicts_and_persists() {
        let directory = std::env::temp_dir().join(format!("jev-dedup-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&directory);
        std::fs::create_dir_all(&directory).unwrap();
        let path = directory.join("dedup.log");
        let mut ledger = DedupLedger::open(&path, 2).unwrap();
        ledger.record("a").unwrap();
        ledger.record("b").unwrap();
        ledger.record("c").unwrap();
        assert!(!ledger.contains("a"));
        assert!(ledger.contains("c"));
        let mut reopened = DedupLedger::open(&path, 2).unwrap();
        assert!(reopened.contains("b") && reopened.contains("c"));
        reopened.record("d").unwrap();
        assert!(!reopened.contains("b"));
        let _ = std::fs::remove_dir_all(&directory);
    }
}
