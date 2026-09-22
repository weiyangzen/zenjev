use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::{BufRead, BufReader, Seek, SeekFrom};
use std::path::PathBuf;
use std::time::{Duration, Instant};

use super::{Delivery, Source, SourceStatus};
use crate::canonical::sha256_hex;
use crate::config::BridgeConfig;
use crate::metrics::append_jsonl;

/// Perpetual directory-spool source (§1.7).
///
/// Consumes NDJSON files under `source_root` in deterministic relative-path
/// order. Two cursors are kept per file: an in-memory `read` cursor that this
/// process has already handed out, and the durable `committed` watermark that
/// only `ack` (or a delivery-free skip) advances. A crash before the
/// downstream durable acceptance therefore redelivers exactly the records that
/// were read but never committed, and the idempotency key suppresses the
/// duplicates downstream.
///
/// The loop never exits on an exhausted directory: an idle pass returns an
/// empty batch and the sink polls again after its 20ms wait. Partial lines at
/// EOF are never consumed, and oversized lines are dead-lettered and skipped
/// with the watermark advanced past them.
pub struct DirSpoolSource {
    root: PathBuf,
    patterns: Vec<String>,
    scan_every: Duration,
    state_path: PathBuf,
    dlq_path: PathBuf,
    quarantine_path: PathBuf,
    max_bytes: usize,
    entries: Vec<FileEntry>,
    cursor: usize,
    last_scan: Option<Instant>,
    endings: HashMap<String, (String, u64)>,
    attempts: HashMap<String, u32>,
}

struct FileEntry {
    path: PathBuf,
    rel: String,
    /// Committed watermark. Only `ack` and delivery-free skips advance it.
    committed: u64,
    /// In-memory read cursor for this process. Never persisted.
    read: u64,
    size: u64,
}

#[derive(Debug, Default, Serialize, Deserialize)]
struct SpoolState {
    #[serde(default)]
    files: HashMap<String, FileCursor>,
}

#[derive(Debug, Default, Serialize, Deserialize, Clone)]
struct FileCursor {
    #[serde(default)]
    offset: u64,
    #[serde(default)]
    size: u64,
}

enum LineOutcome {
    Complete(Vec<u8>),
    /// A line without a terminating newline. Not consumed; retried next pass.
    Partial,
    /// A line beyond `max_bytes`; its byte length is reported so the watermark
    /// can advance past it without ever buffering it.
    Oversize(u64),
    Eof,
}

impl DirSpoolSource {
    pub fn open(config: &BridgeConfig) -> Result<Self, String> {
        let root = config
            .source_root
            .clone()
            .ok_or_else(|| "dir-spool adapter requires source_root".to_string())?;
        if !root.is_dir() {
            return Err(format!(
                "dir-spool source_root is not a readable directory: {}",
                root.display()
            ));
        }
        let patterns = if config.patterns.is_empty() {
            vec!["*.ndjson".to_string()]
        } else {
            config.patterns.clone()
        };
        let poll_ms = config.poll_interval_ms.max(20);
        let state_path = config
            .state_path
            .clone()
            .unwrap_or_else(|| PathBuf::from("runs/jev/mq/dir-spool-state.json"));
        let mut source = Self {
            root,
            patterns,
            scan_every: Duration::from_millis(poll_ms),
            state_path,
            dlq_path: config
                .dlq_path
                .clone()
                .unwrap_or_else(|| PathBuf::from("runs/jev/mq/dlq.jsonl")),
            quarantine_path: config.quarantine_path.clone(),
            max_bytes: config.max_bytes,
            entries: Vec::new(),
            cursor: 0,
            last_scan: None,
            endings: HashMap::new(),
            attempts: HashMap::new(),
        };
        source.scan()?;
        Ok(source)
    }

    fn load_state(&self) -> SpoolState {
        std::fs::read_to_string(&self.state_path)
            .ok()
            .and_then(|raw| serde_json::from_str::<SpoolState>(&raw).ok())
            .unwrap_or_default()
    }

    fn persist_state(&self) -> Result<(), String> {
        if let Some(parent) = self.state_path.parent() {
            std::fs::create_dir_all(parent).map_err(|error| error.to_string())?;
        }
        let mut state = SpoolState::default();
        for entry in &self.entries {
            state.files.insert(
                entry.rel.clone(),
                FileCursor {
                    offset: entry.committed,
                    size: entry.size,
                },
            );
        }
        let body = serde_json::to_string(&state).map_err(|error| error.to_string())?;
        let temporary = self.state_path.with_extension("tmp");
        std::fs::write(&temporary, body).map_err(|error| error.to_string())?;
        std::fs::rename(&temporary, &self.state_path).map_err(|error| error.to_string())
    }

    /// Full rescan. Files are ordered by relative path; a file that shrank
    /// below its committed watermark (truncation/rotation) restarts at zero and
    /// duplicates are suppressed downstream by the idempotency key.
    fn scan(&mut self) -> Result<(), String> {
        let state = self.load_state();
        let previous: HashMap<String, FileEntry> = self
            .entries
            .drain(..)
            .map(|entry| (entry.rel.clone(), entry))
            .collect();
        let mut found: Vec<FileEntry> = Vec::new();
        let mut stack = vec![self.root.clone()];
        while let Some(directory) = stack.pop() {
            let listing = std::fs::read_dir(&directory)
                .map_err(|error| format!("cannot read {}: {error}", directory.display()))?;
            for item in listing {
                let item = item.map_err(|error| error.to_string())?;
                let path = item.path();
                let file_type = item.file_type().map_err(|error| error.to_string())?;
                if file_type.is_dir() {
                    stack.push(path);
                    continue;
                }
                let name = match path.file_name().and_then(|value| value.to_str()) {
                    Some(value) => value.to_string(),
                    None => continue,
                };
                if !matches_any(&name, &self.patterns) {
                    continue;
                }
                let rel = path
                    .strip_prefix(&self.root)
                    .map(|value| value.to_string_lossy().replace('\\', "/"))
                    .unwrap_or_else(|_| name.clone());
                let size = std::fs::metadata(&path)
                    .map(|metadata| metadata.len())
                    .unwrap_or(0);
                let mut committed = state
                    .files
                    .get(&rel)
                    .map(|cursor| cursor.offset)
                    .unwrap_or(0);
                if committed > size {
                    committed = 0;
                }
                let read = previous
                    .get(&rel)
                    .map(|entry| entry.read.max(committed))
                    .unwrap_or(committed)
                    .min(size);
                found.push(FileEntry {
                    path,
                    rel,
                    committed,
                    read,
                    size,
                });
            }
        }
        found.sort_by(|left, right| left.rel.cmp(&right.rel));
        self.entries = found;
        self.cursor = 0;
        self.last_scan = Some(Instant::now());
        Ok(())
    }

    fn note(&self, class: &str, reason: &str, extra: serde_json::Value) {
        let path = if class == "quarantine" {
            &self.quarantine_path
        } else {
            &self.dlq_path
        };
        let mut body = serde_json::json!({
            "class": class,
            "reason": reason,
            "adapter": "dir-spool",
        });
        if let (Some(object), Some(extra)) = (body.as_object_mut(), extra.as_object()) {
            for (key, value) in extra {
                object.insert(key.clone(), value.clone());
            }
        }
        append_jsonl(path, &body);
    }

    fn lag_bytes(&self) -> i64 {
        self.entries
            .iter()
            .map(|entry| entry.size.saturating_sub(entry.committed) as i64)
            .sum()
    }
}

impl Source for DirSpoolSource {
    fn adapter(&self) -> &'static str {
        "dir-spool"
    }

    fn next(&mut self, max: usize) -> Result<Vec<Delivery>, String> {
        if max == 0 {
            return Ok(Vec::new());
        }
        let idle = self.cursor >= self.entries.len()
            && self
                .last_scan
                .map(|last| last.elapsed() >= self.scan_every)
                .unwrap_or(true);
        if self.entries.is_empty() || idle {
            self.scan()?;
        }
        let mut batch: Vec<Delivery> = Vec::new();
        let mut moved_without_delivery = false;
        'files: while batch.len() < max && self.cursor < self.entries.len() {
            let index = self.cursor;
            let (path, rel, mut read, size) = {
                let entry = &self.entries[index];
                (
                    entry.path.clone(),
                    entry.rel.clone(),
                    entry.read,
                    entry.size,
                )
            };
            if read >= size {
                self.cursor += 1;
                continue;
            }
            let metadata = std::fs::metadata(&path)
                .map_err(|error| format!("cannot stat {}: {error}", path.display()))?;
            let size = metadata.len();
            let file = match std::fs::File::open(&path) {
                Ok(file) => file,
                Err(error) => {
                    self.note(
                        "dlq",
                        "open_failed",
                        serde_json::json!({"path": rel, "error": error.to_string()}),
                    );
                    self.cursor += 1;
                    continue;
                }
            };
            let mut reader = BufReader::new(file);
            reader
                .seek(SeekFrom::Start(read))
                .map_err(|error| format!("cannot seek {}: {error}", path.display()))?;
            let mut saturated = false;
            let mut skipped_without_delivery = false;
            loop {
                if batch.len() >= max {
                    saturated = true;
                    break;
                }
                match read_line_bounded(&mut reader, self.max_bytes)
                    .map_err(|error| format!("cannot read {}: {error}", path.display()))?
                {
                    LineOutcome::Complete(line) => {
                        let line_start = read;
                        read += line.len() as u64;
                        if line.iter().all(|byte| byte.is_ascii_whitespace()) {
                            skipped_without_delivery = true;
                            continue;
                        }
                        let id = format!(
                            "dirspool-{}-{}",
                            &sha256_hex(rel.as_bytes())[..12],
                            line_start
                        );
                        let attempts = self.attempts.entry(id.clone()).or_insert(0);
                        *attempts += 1;
                        self.endings.insert(id.clone(), (rel.clone(), read));
                        batch.push(Delivery {
                            id,
                            offset: line_start,
                            attempts: *attempts,
                            raw: line,
                        });
                    }
                    LineOutcome::Partial => break,
                    LineOutcome::Oversize(skipped) => {
                        self.note(
                            "dlq",
                            "oversized_line",
                            serde_json::json!({"path": rel, "offset": read, "bytes": skipped}),
                        );
                        read += skipped;
                        skipped_without_delivery = true;
                    }
                    LineOutcome::Eof => break,
                }
            }
            {
                let entry = &mut self.entries[index];
                entry.read = read.max(entry.read);
                entry.size = size;
                if skipped_without_delivery {
                    entry.committed = entry.committed.max(read);
                    moved_without_delivery = true;
                }
            }
            if saturated {
                self.persist_state()?;
                // Round-robin: continue from the next file on the following
                // call so a fast-growing file cannot starve its neighbours.
                self.cursor = if self.entries.is_empty() {
                    0
                } else {
                    (index + 1) % self.entries.len()
                };
                break 'files;
            }
            // Partial line, EOF, or a fully consumed file: move to the next
            // file for fairness, and stop this call after a full pass. The next
            // call rescans once `scan_every` has elapsed, which is what retries
            // a partial line and discovers new files. Wrapping the cursor to
            // zero here would spin on the same partial line forever.
            let next = index + 1;
            if next >= self.entries.len() {
                self.cursor = self.entries.len();
            } else {
                self.cursor = next;
            }
        }
        if moved_without_delivery {
            self.persist_state()?;
        }
        Ok(batch)
    }

    fn ack(&mut self, delivery: &Delivery) -> Result<(), String> {
        if let Some((rel, end)) = self.endings.remove(&delivery.id) {
            if let Some(entry) = self.entries.iter_mut().find(|entry| entry.rel == rel) {
                if end > entry.committed {
                    entry.committed = end;
                    self.persist_state()?;
                }
            }
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
                "adapter": "dir-spool",
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
                "adapter": "dir-spool",
                "attempts": attempts,
                "raw_sha256": sha256_hex(raw),
                "raw_bytes": raw.len(),
            }),
        );
        Ok(())
    }

    fn status(&self) -> SourceStatus {
        SourceStatus {
            lag: self.lag_bytes(),
            checkpoint: Some(
                self.entries
                    .iter()
                    .map(|entry| entry.committed)
                    .sum::<u64>(),
            ),
            loop_cycles: 0,
        }
    }

    fn shutdown(&mut self) {
        let _ = self.persist_state();
    }
}

fn matches_any(name: &str, patterns: &[String]) -> bool {
    patterns.iter().any(|pattern| {
        if let Some(suffix) = pattern.strip_prefix('*') {
            name.ends_with(suffix)
        } else {
            name == pattern
        }
    })
}

fn read_line_bounded<R: BufRead>(reader: &mut R, cap: usize) -> std::io::Result<LineOutcome> {
    let mut accumulated: Vec<u8> = Vec::new();
    let mut oversize = false;
    let mut skipped: u64 = 0;
    loop {
        let available = reader.fill_buf()?;
        if available.is_empty() {
            if oversize {
                return Ok(LineOutcome::Oversize(skipped));
            }
            if accumulated.is_empty() {
                return Ok(LineOutcome::Eof);
            }
            return Ok(LineOutcome::Partial);
        }
        if let Some(position) = available.iter().position(|byte| *byte == b'\n') {
            let take = position + 1;
            if oversize {
                skipped += take as u64;
                reader.consume(take);
                return Ok(LineOutcome::Oversize(skipped));
            }
            if accumulated.len() + take > cap {
                reader.consume(take);
                return Ok(LineOutcome::Oversize(take as u64));
            }
            accumulated.extend_from_slice(&available[..take]);
            reader.consume(take);
            return Ok(LineOutcome::Complete(accumulated));
        }
        let length = available.len();
        if oversize {
            skipped += length as u64;
        } else if accumulated.len() + length > cap {
            oversize = true;
            accumulated.clear();
            skipped = length as u64;
        } else {
            accumulated.extend_from_slice(available);
        }
        reader.consume(length);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::BridgeConfig;
    use std::io::Write;
    use std::path::Path;

    fn config_for(root: &Path, state: &Path) -> BridgeConfig {
        serde_json::from_value(serde_json::json!({
            "adapter": "dir-spool",
            "source_root": root,
            "state_path": state,
            "socket_path": state.with_extension("sock"),
            "wal_path": state.with_extension("wal"),
            "dlq_path": state.with_extension("dlq"),
            "quarantine_path": state.with_extension("quarantine"),
            "max_bytes": 4096,
            "poll_interval_ms": 20,
            "schema_id": "s",
            "schema_version": 1,
            "schema_digest": "a".repeat(64),
        }))
        .unwrap()
    }

    #[test]
    fn resumes_from_committed_offset_and_redelivers_unacked_lines() {
        let directory = std::env::temp_dir().join(format!("jev-dirspool-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&directory);
        std::fs::create_dir_all(&directory).unwrap();
        std::fs::write(directory.join("a.ndjson"), "{\"n\":1}\n{\"n\":2}\n").unwrap();
        std::fs::write(directory.join("b.ndjson"), "{\"n\":3}\n").unwrap();
        let state = directory.join("state.json");
        let mut source = DirSpoolSource::open(&config_for(&directory, &state)).unwrap();
        let batch = source.next(10).unwrap();
        assert_eq!(batch.len(), 3);
        source.ack(&batch[0]).unwrap();
        let mut reopened = DirSpoolSource::open(&config_for(&directory, &state)).unwrap();
        let replay = reopened.next(10).unwrap();
        assert_eq!(replay.len(), 2);
        assert_eq!(replay[0].offset, 8);
        reopened.ack(&replay[0]).unwrap();
        reopened.ack(&replay[1]).unwrap();
        std::thread::sleep(Duration::from_millis(30));
        let idle = reopened.next(10).unwrap();
        assert!(idle.is_empty());
        let _ = std::fs::remove_dir_all(&directory);
    }

    #[test]
    fn partial_line_is_not_consumed_until_newline_arrives() {
        let directory =
            std::env::temp_dir().join(format!("jev-dirspool-partial-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&directory);
        std::fs::create_dir_all(&directory).unwrap();
        std::fs::write(directory.join("a.ndjson"), "{\"n\":1}").unwrap();
        let state = directory.join("state.json");
        let mut source = DirSpoolSource::open(&config_for(&directory, &state)).unwrap();
        assert!(source.next(10).unwrap().is_empty());
        std::fs::OpenOptions::new()
            .append(true)
            .open(directory.join("a.ndjson"))
            .unwrap()
            .write_all(b"\n{\"n\":2}\n")
            .unwrap();
        std::thread::sleep(Duration::from_millis(30));
        let batch = source.next(10).unwrap();
        assert_eq!(batch.len(), 2);
        let _ = std::fs::remove_dir_all(&directory);
    }

    #[test]
    fn files_rotate_so_a_fast_file_cannot_starve_others() {
        let directory =
            std::env::temp_dir().join(format!("jev-dirspool-fair-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&directory);
        std::fs::create_dir_all(&directory).unwrap();
        std::fs::write(
            directory.join("a.ndjson"),
            "{\"n\":1}\n{\"n\":2}\n{\"n\":3}\n",
        )
        .unwrap();
        std::fs::write(
            directory.join("b.ndjson"),
            "{\"n\":4}\n{\"n\":5}\n{\"n\":6}\n",
        )
        .unwrap();
        let state = directory.join("state.json");
        let mut source = DirSpoolSource::open(&config_for(&directory, &state)).unwrap();
        let first = source.next(2).unwrap();
        let second = source.next(2).unwrap();
        assert_eq!(first.len(), 2);
        assert_eq!(second.len(), 2);
        assert_ne!(
            first[0].id.split('-').nth(1),
            second[0].id.split('-').nth(1),
            "the second batch must come from the other file"
        );
        let _ = std::fs::remove_dir_all(&directory);
    }

    #[test]
    fn oversized_line_is_dead_lettered_and_skipped() {
        let directory =
            std::env::temp_dir().join(format!("jev-dirspool-big-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&directory);
        std::fs::create_dir_all(&directory).unwrap();
        let mut body = vec![b'x'; 8192];
        body.push(b'\n');
        body.extend_from_slice(b"{\"n\":1}\n");
        std::fs::write(directory.join("a.ndjson"), body).unwrap();
        let state = directory.join("state.json");
        let mut source = DirSpoolSource::open(&config_for(&directory, &state)).unwrap();
        let batch = source.next(10).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(batch[0].raw, b"{\"n\":1}\n");
        assert!(state.with_extension("dlq").exists());
        let _ = std::fs::remove_dir_all(&directory);
    }
}
