//! `zenjev-monitor` — a small, robust, read-only observability tool for the
//! ZenJev train/infer service.
//!
//! The crate is deliberately split into small, dependency-free parsers (unit
//! tested from synthetic text) and a sampler that reads live kernel/GPU state.
//! Every sampler returns `Option`/`Result`; the loop never panics on missing
//! files, permission errors, short reads, a disappearing process, a malformed
//! status document, or a host without `nvidia-smi`.

pub mod args;
pub mod dashboard;
pub mod gpu;
pub mod output;
pub mod process;
pub mod procfs;
pub mod sample;
pub mod status;

pub use sample::{Sampler, Snapshot};
pub use status::StatusFile;

/// Version of the monitor's own JSON envelope (not the service status schema).
pub const MONITOR_SCHEMA_VERSION: u32 = 1;

/// Milliseconds since the Unix epoch. Never panics; a pre-epoch clock yields 0.
pub fn unix_ms() -> u128 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_millis())
        .unwrap_or_default()
}
