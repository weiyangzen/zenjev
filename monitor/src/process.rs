use crate::procfs;
use serde::Serialize;
use std::time::{Duration, Instant};

/// Wait between the two reads that establish the first process CPU delta so a
/// `--once` sample still carries a meaningful percentage.
const FIRST_PROCESS_PROBE: Duration = Duration::from_millis(100);

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ProcessSample {
    pub pid: u32,
    pub rss_bytes: Option<u64>,
    pub cpu_percent: Option<f64>,
    pub threads: Option<u64>,
    pub state: Option<String>,
}

/// Samples a single PID from `/proc/<pid>/stat` (CPU ticks, threads, fallback
/// RSS) and `/proc/<pid>/status` (VmRSS). The sampler keeps the previous tick
/// count across intervals; the first sample establishes its own 100 ms probe
/// window so one-shot output is not empty.
pub struct ProcessSampler {
    pid: u32,
    previous: Option<(u64, Instant)>,
}

impl ProcessSampler {
    pub fn new(pid: u32) -> Self {
        Self {
            pid,
            previous: None,
        }
    }

    pub fn pid(&self) -> u32 {
        self.pid
    }

    /// `None` when the process is gone, `/proc` is unreadable, or the stat
    /// line is malformed; a process exiting mid-sample must never crash the
    /// monitor.
    pub fn sample(&mut self) -> Option<ProcessSample> {
        let (first_stat, first_status) = read_process(self.pid)?;
        let (baseline_ticks, baseline_at, stat, status, now) = match self.previous {
            Some((ticks, at)) => (ticks, at, first_stat, first_status, Instant::now()),
            None => {
                let at = Instant::now();
                std::thread::sleep(FIRST_PROCESS_PROBE);
                let (stat, status) = read_process(self.pid)?;
                let now = Instant::now();
                (first_stat.cpu_ticks, at, stat, status, now)
            }
        };
        let elapsed = now.saturating_duration_since(baseline_at).as_secs_f64();
        let cpu_percent = if elapsed > 0.0 && stat.cpu_ticks >= baseline_ticks {
            Some(
                stat.cpu_ticks.saturating_sub(baseline_ticks) as f64 / procfs::USER_HZ / elapsed
                    * 100.0,
            )
        } else {
            None
        };
        self.previous = Some((stat.cpu_ticks, now));
        let rss_bytes = status
            .as_ref()
            .and_then(|value| value.rss_bytes)
            .or_else(|| {
                if stat.rss_pages > 0 {
                    Some(stat.rss_pages.saturating_mul(procfs::PAGE_SIZE_BYTES))
                } else {
                    None
                }
            });
        let state = status
            .as_ref()
            .and_then(|value| value.state.clone())
            .or_else(|| Some(stat.state.clone()));
        Some(ProcessSample {
            pid: self.pid,
            rss_bytes,
            cpu_percent,
            threads: Some(stat.threads),
            state,
        })
    }
}

fn read_process(pid: u32) -> Option<(procfs::ProcessStat, Option<procfs::ProcessStatus>)> {
    let stat_text = std::fs::read_to_string(format!("/proc/{pid}/stat")).ok()?;
    let stat = procfs::parse_proc_pid_stat(&stat_text).ok()?;
    let status = std::fs::read_to_string(format!("/proc/{pid}/status"))
        .ok()
        .map(|text| procfs::parse_proc_pid_status(&text));
    Some((stat, status))
}
