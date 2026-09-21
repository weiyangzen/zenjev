use crate::process::{ProcessSample, ProcessSampler};
use crate::procfs::{self, CpuSample, LoadSample, MemorySample, ProcStat};
use crate::status::{self, StatusFile};
use crate::MONITOR_SCHEMA_VERSION;
use serde::Serialize;
use std::path::PathBuf;
use std::time::Duration;

/// How long the very first CPU reading waits between its two `/proc/stat`
/// reads so a `--once` sample still carries a meaningful delta.
const FIRST_CPU_PROBE: Duration = Duration::from_millis(100);

#[derive(Debug, Clone, Serialize)]
pub struct Snapshot {
    pub monitor_schema_version: u32,
    pub sampled_at_unix_ms: u128,
    pub cpu: Option<CpuSample>,
    pub load: Option<LoadSample>,
    pub memory: Option<MemorySample>,
    /// `None` means GPU sampling is unavailable (no `nvidia-smi`/no GPU);
    /// `Some(vec![])` is never produced by the sampler.
    pub gpu: Option<Vec<crate::gpu::GpuSample>>,
    pub process: Option<ProcessSample>,
    pub status: Option<StatusFile>,
    pub status_age_seconds: Option<f64>,
    pub status_error: Option<String>,
    pub stale: bool,
}

/// Owns the cross-tick state needed for delta metrics. Every field is
/// optional in the emitted snapshot; a failing source is simply omitted.
pub struct Sampler {
    status_path: PathBuf,
    stale_after_seconds: f64,
    cpu_previous: Option<ProcStat>,
    process: Option<ProcessSampler>,
}

impl Sampler {
    pub fn new(status_path: PathBuf, stale_after_seconds: f64) -> Self {
        Self {
            status_path,
            stale_after_seconds,
            cpu_previous: None,
            process: None,
        }
    }

    pub fn sample(&mut self, now_unix_ms: u128) -> Snapshot {
        let cpu = self.sample_cpu();
        let load = std::fs::read_to_string("/proc/loadavg")
            .ok()
            .and_then(|text| procfs::parse_loadavg(&text).ok());
        let memory = std::fs::read_to_string("/proc/meminfo")
            .ok()
            .and_then(|text| procfs::parse_meminfo(&text).ok());
        let gpu = crate::gpu::sample_gpu();
        let status_read =
            status::load_status(&self.status_path, now_unix_ms, self.stale_after_seconds);
        let process = status_read
            .status
            .as_ref()
            .and_then(|status| status.pid)
            .and_then(|pid| self.sample_process(pid));
        Snapshot {
            monitor_schema_version: MONITOR_SCHEMA_VERSION,
            sampled_at_unix_ms: now_unix_ms,
            cpu,
            load,
            memory,
            gpu,
            process,
            status: status_read.status,
            status_age_seconds: status_read.status_age_seconds,
            status_error: status_read.status_error,
            stale: status_read.stale,
        }
    }

    fn sample_cpu(&mut self) -> Option<CpuSample> {
        let first = procfs::read_proc_stat().ok()?;
        let (baseline, current) = match self.cpu_previous.take() {
            Some(previous) => (previous, first),
            None => {
                std::thread::sleep(FIRST_CPU_PROBE);
                (first, procfs::read_proc_stat().ok()?)
            }
        };
        let (percent, per_core_percent) = procfs::cpu_percent_delta(&baseline, &current);
        let cores = current.cores.len();
        self.cpu_previous = Some(current);
        Some(CpuSample {
            percent,
            cores,
            per_core_percent,
        })
    }

    fn sample_process(&mut self, pid: u32) -> Option<ProcessSample> {
        if self.process.as_ref().map(ProcessSampler::pid) != Some(pid) {
            self.process = Some(ProcessSampler::new(pid));
        }
        self.process.as_mut()?.sample()
    }
}
