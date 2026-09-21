use serde::{Deserialize, Serialize};
use std::path::Path;

/// Status document schema this monitor understands. Unknown fields are
/// tolerated; any other `schema_version` fails validation and is treated as
/// stale so a future writer cannot be silently misread.
pub const STATUS_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StatusFile {
    pub schema_version: u32,
    pub updated_at_unix_ms: u128,
    #[serde(default)]
    pub pid: Option<u32>,
    #[serde(default)]
    pub phase: Option<String>,
    #[serde(default)]
    pub generation: Option<u64>,
    #[serde(default)]
    pub ema_step: Option<u64>,
    #[serde(default)]
    pub training_steps: Option<u64>,
    #[serde(default)]
    pub reset_id: Option<u64>,
    #[serde(default)]
    pub loss: Option<f64>,
    #[serde(default)]
    pub learning_rate: Option<f64>,
    #[serde(default)]
    pub checkpoint: Option<String>,
    #[serde(default)]
    pub config_digest: Option<String>,
    #[serde(default)]
    pub model_id: Option<String>,
    #[serde(default)]
    pub device: Option<String>,
    #[serde(default)]
    pub note: Option<String>,
}

impl StatusFile {
    pub fn validate(&self) -> Result<(), String> {
        if self.schema_version != STATUS_SCHEMA_VERSION {
            return Err(format!(
                "unsupported status schema_version {} (monitor expects {})",
                self.schema_version, STATUS_SCHEMA_VERSION
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct StatusRead {
    pub status: Option<StatusFile>,
    pub status_age_seconds: Option<f64>,
    pub stale: bool,
    pub status_error: Option<String>,
}

/// Load and validate the status document. Never returns an error: a missing,
/// unreadable, malformed, or unsupported file yields `status: None`,
/// `stale: true` and a human-readable `status_error`.
pub fn load_status(path: &Path, now_unix_ms: u128, stale_after_seconds: f64) -> StatusRead {
    let raw = match std::fs::read_to_string(path) {
        Ok(raw) => raw,
        Err(error) => {
            let reason = if error.kind() == std::io::ErrorKind::NotFound {
                format!("status file not found: {}", path.display())
            } else {
                format!("cannot read status file {}: {error}", path.display())
            };
            return stale_read(reason);
        }
    };
    let parsed: StatusFile = match serde_json::from_str(&raw) {
        Ok(parsed) => parsed,
        Err(error) => return stale_read(format!("invalid status JSON: {error}")),
    };
    if let Err(error) = parsed.validate() {
        return stale_read(error);
    }
    let age = age_seconds(now_unix_ms, parsed.updated_at_unix_ms);
    let stale = age > stale_after_seconds.max(0.0);
    StatusRead {
        status: Some(parsed),
        status_age_seconds: Some(age),
        stale,
        status_error: None,
    }
}

fn stale_read(error: String) -> StatusRead {
    StatusRead {
        status: None,
        status_age_seconds: None,
        stale: true,
        status_error: Some(error),
    }
}

fn age_seconds(now_unix_ms: u128, updated_unix_ms: u128) -> f64 {
    if now_unix_ms <= updated_unix_ms {
        0.0
    } else {
        (now_unix_ms - updated_unix_ms) as f64 / 1000.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    static COUNTER: AtomicU64 = AtomicU64::new(0);

    struct TempFile {
        path: PathBuf,
    }

    impl TempFile {
        fn new(contents: &str) -> Self {
            let mut path = std::env::temp_dir();
            path.push(format!(
                "zenjev-monitor-status-{}-{}.json",
                std::process::id(),
                COUNTER.fetch_add(1, Ordering::Relaxed)
            ));
            std::fs::write(&path, contents).expect("write temp status");
            Self { path }
        }

        fn absent() -> Self {
            let mut path = std::env::temp_dir();
            path.push(format!(
                "zenjev-monitor-status-{}-{}.json",
                std::process::id(),
                COUNTER.fetch_add(1, Ordering::Relaxed)
            ));
            let _ = std::fs::remove_file(&path);
            Self { path }
        }
    }

    impl Drop for TempFile {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.path);
        }
    }

    const VALID: &str = r#"{
        "schema_version": 1,
        "updated_at_unix_ms": 100000,
        "pid": 12345,
        "phase": "training",
        "generation": 9,
        "ema_step": 8,
        "training_steps": 8,
        "reset_id": 0,
        "loss": 1.57,
        "learning_rate": 0.0001,
        "checkpoint": "runs/jev/jevraw/adapter-step-00000008.pt",
        "config_digest": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
        "model_id": "fastino/gliner2-base-v1",
        "device": "cuda",
        "future_field": {"unknown": true}
    }"#;

    #[test]
    fn loads_valid_status_and_ignores_unknown_fields() {
        let file = TempFile::new(VALID);
        let read = load_status(&file.path, 103_000, 15.0);
        assert!(read.status_error.is_none());
        assert!(!read.stale);
        assert_eq!(read.status_age_seconds, Some(3.0));
        let status = read.status.expect("status parsed");
        assert_eq!(status.pid, Some(12345));
        assert_eq!(status.phase.as_deref(), Some("training"));
        assert_eq!(status.generation, Some(9));
        assert_eq!(status.loss, Some(1.57));
        assert_eq!(status.model_id.as_deref(), Some("fastino/gliner2-base-v1"));
    }

    #[test]
    fn missing_file_is_stale_and_null() {
        let file = TempFile::absent();
        let read = load_status(&file.path, 103_000, 15.0);
        assert!(read.status.is_none());
        assert!(read.stale);
        assert_eq!(read.status_age_seconds, None);
        let error = read.status_error.expect("error");
        assert!(error.contains("not found"), "{error}");
    }

    #[test]
    fn malformed_json_is_stale_and_null() {
        let file = TempFile::new("{ this is not json");
        let read = load_status(&file.path, 103_000, 15.0);
        assert!(read.status.is_none());
        assert!(read.stale);
        let error = read.status_error.expect("error");
        assert!(error.contains("invalid status JSON"), "{error}");
    }

    #[test]
    fn unsupported_schema_version_is_rejected() {
        let file = TempFile::new(r#"{"schema_version": 2, "updated_at_unix_ms": 100000}"#);
        let read = load_status(&file.path, 103_000, 15.0);
        assert!(read.status.is_none());
        assert!(read.stale);
        let error = read.status_error.expect("error");
        assert!(error.contains("schema_version"), "{error}");
    }

    #[test]
    fn missing_required_fields_are_rejected() {
        let file = TempFile::new(r#"{"schema_version": 1}"#);
        let read = load_status(&file.path, 103_000, 15.0);
        assert!(read.status.is_none());
        assert!(read.stale);
    }

    #[test]
    fn stale_calculation_and_future_timestamps() {
        let file = TempFile::new(VALID);
        let fresh = load_status(&file.path, 114_000, 15.0);
        assert_eq!(fresh.status_age_seconds, Some(14.0));
        assert!(!fresh.stale);

        let stale = load_status(&file.path, 120_000, 15.0);
        assert_eq!(stale.status_age_seconds, Some(20.0));
        assert!(stale.stale);

        let future = load_status(&file.path, 90_000, 15.0);
        assert_eq!(future.status_age_seconds, Some(0.0));
        assert!(!future.stale);
    }

    #[test]
    fn zero_stale_after_marks_any_age_stale() {
        let file = TempFile::new(VALID);
        assert!(load_status(&file.path, 100_001, 0.0).stale);
    }
}
