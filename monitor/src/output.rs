use crate::sample::Snapshot;
use std::io::Write;
use std::path::Path;

/// Append one sample as a single JSON line, creating parent directories.
/// Append + flush + `sync_data` keeps evidence durable enough for
/// `artifacts/monitor/metrics.jsonl`; a slow or read-only path is reported to
/// the caller and never panics.
pub fn append_jsonl(path: &Path, snapshot: &Snapshot) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    let mut body = serde_json::to_string(snapshot).map_err(std::io::Error::other)?;
    body.push('\n');
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)?;
    file.write_all(body.as_bytes())?;
    file.flush()?;
    file.sync_data()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::procfs::{CpuSample, LoadSample, MemorySample};
    use crate::sample::Snapshot;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    static COUNTER: AtomicU64 = AtomicU64::new(0);

    fn temp_dir() -> PathBuf {
        let mut path = std::env::temp_dir();
        path.push(format!(
            "zenjev-monitor-jsonl-{}-{}",
            std::process::id(),
            COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        path
    }

    fn snapshot(index: u128) -> Snapshot {
        Snapshot {
            monitor_schema_version: 1,
            sampled_at_unix_ms: index,
            cpu: Some(CpuSample {
                percent: Some(10.0),
                cores: 2,
                per_core_percent: vec![Some(5.0), Some(15.0)],
            }),
            load: Some(LoadSample {
                one: 0.1,
                five: 0.2,
                fifteen: 0.3,
                running: 1,
                total: 100,
            }),
            memory: Some(MemorySample {
                total_bytes: 1024,
                used_bytes: 512,
                available_bytes: 512,
                used_percent: 50.0,
            }),
            gpu: None,
            process: None,
            status: None,
            status_age_seconds: None,
            status_error: Some("status file not found".to_string()),
            stale: true,
        }
    }

    #[test]
    fn appends_one_json_object_per_line_and_creates_parents() {
        let dir = temp_dir();
        let path = dir.join("nested").join("metrics.jsonl");
        append_jsonl(&path, &snapshot(1)).expect("first append");
        append_jsonl(&path, &snapshot(2)).expect("second append");
        let body = std::fs::read_to_string(&path).expect("read back");
        let lines: Vec<&str> = body.lines().collect();
        assert_eq!(lines.len(), 2);
        let first: serde_json::Value = serde_json::from_str(lines[0]).expect("valid json");
        assert_eq!(first["monitor_schema_version"], serde_json::json!(1));
        assert_eq!(first["sampled_at_unix_ms"], serde_json::json!(1));
        assert!(first["gpu"].is_null());
        assert!(first["status"].is_null());
        assert_eq!(first["stale"], serde_json::json!(true));
        assert_eq!(first["cpu"]["cores"], serde_json::json!(2));
        let second: serde_json::Value = serde_json::from_str(lines[1]).expect("valid json");
        assert_eq!(second["sampled_at_unix_ms"], serde_json::json!(2));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
