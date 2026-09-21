use serde::Serialize;

/// `USER_HZ` assumed for `/proc/<pid>/stat` tick accounting on Linux (100 on
/// every mainstream configuration; the value is documented in the README).
pub const USER_HZ: f64 = 100.0;

/// Typical page size, used only as an RSS fallback when `/proc/<pid>/status`
/// cannot be read.
pub const PAGE_SIZE_BYTES: u64 = 4096;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CpuCounters {
    pub user: u64,
    pub nice: u64,
    pub system: u64,
    pub idle: u64,
    pub iowait: u64,
    pub irq: u64,
    pub softirq: u64,
    pub steal: u64,
}

impl CpuCounters {
    pub fn total(&self) -> u64 {
        self.user
            .saturating_add(self.nice)
            .saturating_add(self.system)
            .saturating_add(self.idle)
            .saturating_add(self.iowait)
            .saturating_add(self.irq)
            .saturating_add(self.softirq)
            .saturating_add(self.steal)
    }

    pub fn idle_total(&self) -> u64 {
        self.idle.saturating_add(self.iowait)
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ProcStat {
    pub aggregate: CpuCounters,
    pub cores: Vec<CpuCounters>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct CpuSample {
    pub percent: Option<f64>,
    pub cores: usize,
    pub per_core_percent: Vec<Option<f64>>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct LoadSample {
    pub one: f64,
    pub five: f64,
    pub fifteen: f64,
    pub running: u64,
    pub total: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct MemorySample {
    pub total_bytes: u64,
    pub used_bytes: u64,
    pub available_bytes: u64,
    pub used_percent: f64,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ProcessStat {
    pub state: String,
    pub threads: u64,
    pub cpu_ticks: u64,
    pub start_time_ticks: u64,
    pub rss_pages: u64,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ProcessStatus {
    pub rss_bytes: Option<u64>,
    pub threads: Option<u64>,
    pub state: Option<String>,
}

pub fn read_proc_stat() -> Result<ProcStat, String> {
    let text = std::fs::read_to_string("/proc/stat")
        .map_err(|error| format!("cannot read /proc/stat: {error}"))?;
    parse_proc_stat(&text)
}

/// Parse the aggregate `cpu` line and every `cpuN` line of `/proc/stat`.
/// Non-CPU lines (`intr`, `ctxt`, `btime`, ...) are ignored. A malformed CPU
/// line or a missing aggregate line is an error so callers can skip the sample.
pub fn parse_proc_stat(text: &str) -> Result<ProcStat, String> {
    let mut stat = ProcStat::default();
    let mut saw_aggregate = false;
    for line in text.lines() {
        let mut fields = line.split_whitespace();
        let Some(label) = fields.next() else { continue };
        if !label.starts_with("cpu") {
            continue;
        }
        let values: Vec<&str> = fields.collect();
        let Some(counters) = parse_counters(&values) else {
            return Err(format!("invalid /proc/stat line: {line}"));
        };
        if label == "cpu" {
            stat.aggregate = counters;
            saw_aggregate = true;
        } else if label.len() > 3 && label[3..].bytes().all(|byte| byte.is_ascii_digit()) {
            stat.cores.push(counters);
        }
    }
    if !saw_aggregate {
        return Err("missing aggregate cpu line".to_string());
    }
    Ok(stat)
}

fn parse_counters(values: &[&str]) -> Option<CpuCounters> {
    if values.len() < 4 {
        return None;
    }
    let get = |index: usize| -> Option<u64> {
        values
            .get(index)
            .and_then(|value| value.parse::<u64>().ok())
    };
    Some(CpuCounters {
        user: get(0)?,
        nice: get(1)?,
        system: get(2)?,
        idle: get(3)?,
        iowait: get(4).unwrap_or(0),
        irq: get(5).unwrap_or(0),
        softirq: get(6).unwrap_or(0),
        steal: get(7).unwrap_or(0),
    })
}

/// Busy percentage between two counter snapshots. `None` on counter wrap,
/// zero elapsed time, or an impossible idle delta.
pub fn counters_percent(previous: &CpuCounters, current: &CpuCounters) -> Option<f64> {
    let total = current.total().checked_sub(previous.total())?;
    let idle = current.idle_total().checked_sub(previous.idle_total())?;
    if total == 0 || idle > total {
        return None;
    }
    Some(100.0 * (total - idle) as f64 / total as f64)
}

pub fn cpu_percent_delta(
    previous: &ProcStat,
    current: &ProcStat,
) -> (Option<f64>, Vec<Option<f64>>) {
    let aggregate = counters_percent(&previous.aggregate, &current.aggregate);
    let per_core = current
        .cores
        .iter()
        .enumerate()
        .map(|(index, core)| {
            previous
                .cores
                .get(index)
                .and_then(|prev| counters_percent(prev, core))
        })
        .collect();
    (aggregate, per_core)
}

/// Parse the `MemTotal` / `MemAvailable` / `MemFree` fields of
/// `/proc/meminfo` (values are KiB). Used memory prefers `MemAvailable`, so
/// reclaimable page cache is not reported as pressure.
pub fn parse_meminfo(text: &str) -> Result<MemorySample, String> {
    let mut total_kb = None;
    let mut available_kb = None;
    let mut free_kb = None;
    for line in text.lines() {
        let Some((key, rest)) = line.split_once(':') else {
            continue;
        };
        let value = rest
            .split_whitespace()
            .next()
            .and_then(|value| value.parse::<u64>().ok());
        match key.trim() {
            "MemTotal" => total_kb = value,
            "MemAvailable" => available_kb = value,
            "MemFree" => free_kb = value,
            _ => {}
        }
    }
    let total_kb = total_kb.ok_or_else(|| "MemTotal missing".to_string())?;
    let available_kb = available_kb
        .or(free_kb)
        .ok_or_else(|| "MemAvailable/MemFree missing".to_string())?;
    let total_bytes = total_kb.saturating_mul(1024);
    let available_bytes = available_kb.saturating_mul(1024).min(total_bytes);
    let used_bytes = total_bytes - available_bytes;
    let used_percent = if total_bytes == 0 {
        0.0
    } else {
        used_bytes as f64 / total_bytes as f64 * 100.0
    };
    Ok(MemorySample {
        total_bytes,
        used_bytes,
        available_bytes,
        used_percent,
    })
}

/// Parse the first four fields of `/proc/loadavg` (`load running/total pid`).
pub fn parse_loadavg(text: &str) -> Result<LoadSample, String> {
    let fields: Vec<&str> = text.split_whitespace().collect();
    if fields.len() < 4 {
        return Err("loadavg has fewer than four fields".to_string());
    }
    let parse = |value: &str| {
        value
            .parse::<f64>()
            .map_err(|_| format!("bad loadavg number: {value}"))
    };
    let (running, total) = fields[3]
        .split_once('/')
        .ok_or_else(|| format!("bad loadavg running/total field: {}", fields[3]))?;
    Ok(LoadSample {
        one: parse(fields[0])?,
        five: parse(fields[1])?,
        fifteen: parse(fields[2])?,
        running: running
            .parse::<u64>()
            .map_err(|_| format!("bad running count: {running}"))?,
        total: total
            .parse::<u64>()
            .map_err(|_| format!("bad total count: {total}"))?,
    })
}

/// Parse `/proc/<pid>/stat`. The `comm` field is parenthesized and may itself
/// contain spaces or parentheses, so the scan starts after the last `)`.
pub fn parse_proc_pid_stat(text: &str) -> Result<ProcessStat, String> {
    let close = text
        .rfind(')')
        .ok_or_else(|| "missing comm field".to_string())?;
    let fields: Vec<&str> = text[close + 1..].split_whitespace().collect();
    let get = |index: usize| -> Option<u64> {
        fields
            .get(index)
            .and_then(|value| value.parse::<u64>().ok())
    };
    let utime = get(11).ok_or_else(|| "utime missing".to_string())?;
    let stime = get(12).ok_or_else(|| "stime missing".to_string())?;
    Ok(ProcessStat {
        state: fields.first().copied().unwrap_or("?").to_string(),
        threads: get(17).ok_or_else(|| "num_threads missing".to_string())?,
        cpu_ticks: utime.saturating_add(stime),
        start_time_ticks: get(19).ok_or_else(|| "starttime missing".to_string())?,
        rss_pages: get(21).unwrap_or(0),
    })
}

/// Parse the few `/proc/<pid>/status` fields needed by the monitor. Missing
/// fields are tolerated because the process may exit mid-read.
pub fn parse_proc_pid_status(text: &str) -> ProcessStatus {
    let mut status = ProcessStatus::default();
    for line in text.lines() {
        let Some((key, rest)) = line.split_once(':') else {
            continue;
        };
        match key.trim() {
            "VmRSS" => {
                status.rss_bytes = rest
                    .split_whitespace()
                    .next()
                    .and_then(|value| value.parse::<u64>().ok())
                    .map(|kb| kb.saturating_mul(1024));
            }
            "Threads" => {
                status.threads = rest.trim().parse::<u64>().ok();
            }
            "State" => {
                status.state = rest.split_whitespace().next().map(str::to_string);
            }
            _ => {}
        }
    }
    status
}

#[cfg(test)]
mod tests {
    use super::*;

    const PROC_STAT: &str = "\
cpu  100 20 30 400 10 5 5 0 2 4
cpu0 60 10 15 200 5 2 3 0 1 2
cpu1 40 10 15 200 5 3 2 0 1 2
intr 12345 0 0
ctxt 999
btime 1600000000
processes 4242
procs_running 3
procs_blocked 0
";

    #[test]
    fn parses_aggregate_and_per_core_counters() {
        let stat = parse_proc_stat(PROC_STAT).expect("valid /proc/stat");
        assert_eq!(stat.cores.len(), 2);
        assert_eq!(stat.aggregate.user, 100);
        assert_eq!(stat.aggregate.idle_total(), 410);
        assert_eq!(stat.aggregate.total(), 570);
        assert_eq!(stat.cores[1].softirq, 2);
    }

    #[test]
    fn computes_percent_from_delta() {
        let previous = parse_proc_stat(PROC_STAT).expect("prev");
        let current_text = "\
cpu  130 20 30 500 10 5 5 0 2 4
cpu0 90 10 15 230 5 2 3 0 1 2
cpu1 40 10 15 270 5 3 2 0 1 2
";
        let current = parse_proc_stat(current_text).expect("cur");
        let (aggregate, per_core) = cpu_percent_delta(&previous, &current);
        // aggregate: total +130, idle +100 -> 30/130 busy
        let aggregate = aggregate.expect("aggregate percent");
        assert!((aggregate - 100.0 * 30.0 / 130.0).abs() < 0.001);
        assert_eq!(per_core.len(), 2);
        // cpu0: total +60, idle +30 -> 50%; cpu1: total +70, idle +70 -> 0%
        assert_eq!(per_core[0], Some(50.0));
        assert_eq!(per_core[1], Some(0.0));

        let busy_text = "\
cpu  200 20 30 500 10 5 5 0 2 4
cpu0 90 10 15 230 5 2 3 0 1 2
cpu1 140 10 15 240 5 3 2 0 1 2
";
        let busy = parse_proc_stat(busy_text).expect("busy");
        let (aggregate, per_core) = cpu_percent_delta(&previous, &busy);
        // aggregate: total +200, idle +100 -> 50% busy
        assert_eq!(aggregate, Some(50.0));
        assert_eq!(per_core[0], Some(50.0));
        let cpu1 = per_core[1].expect("cpu1 percent");
        assert!((cpu1 - 100.0 * 100.0 / 140.0).abs() < 0.001);
    }

    #[test]
    fn counter_wrap_and_invalid_input_do_not_panic() {
        let previous = parse_proc_stat(PROC_STAT).expect("prev");
        let wrapped =
            parse_proc_stat("cpu 1 1 1 1 1 1 1 1\ncpu0 1 1 1 1 1 1 1 1\ncpu1 1 1 1 1 1 1 1 1\n")
                .expect("wrapped");
        let (aggregate, per_core) = cpu_percent_delta(&previous, &wrapped);
        assert_eq!(aggregate, None);
        assert_eq!(per_core, vec![None, None]);

        let invalid = parse_proc_stat("cpu a b c d\n");
        assert!(invalid.is_err());
        let missing_aggregate = parse_proc_stat("cpu0 1 2 3 4\n");
        assert!(missing_aggregate.is_err());
        let short = parse_proc_stat("cpu 1 2 3\n");
        assert!(short.is_err());
    }

    #[test]
    fn handles_missing_per_core_rows() {
        let previous = parse_proc_stat(PROC_STAT).expect("prev");
        let current =
            parse_proc_stat("cpu 200 20 30 500 10 5 5 0 2 4\ncpu0 60 10 15 250 5 2 3 0 1 2\n")
                .expect("single core");
        let (aggregate, per_core) = cpu_percent_delta(&previous, &current);
        assert_eq!(aggregate, Some(50.0));
        assert_eq!(per_core, vec![Some(0.0)]);
    }

    #[test]
    fn parses_meminfo_preferring_available() {
        let text = "\
MemTotal:       16384000 kB
MemFree:         2000000 kB
MemAvailable:    4096000 kB
Buffers:          100000 kB
";
        let memory = parse_meminfo(text).expect("meminfo");
        assert_eq!(memory.total_bytes, 16384000 * 1024);
        assert_eq!(memory.available_bytes, 4096000 * 1024);
        assert_eq!(memory.used_bytes, (16384000 - 4096000) * 1024);
        let percent = memory.used_percent;
        assert!((percent - 75.0).abs() < 0.001);
    }

    #[test]
    fn parses_meminfo_fallback_and_rejects_missing() {
        let fallback = parse_meminfo("MemTotal: 1000 kB\nMemFree: 250 kB\n").expect("fallback");
        assert_eq!(fallback.available_bytes, 250 * 1024);
        assert!(parse_meminfo("MemFree: 250 kB\n").is_err());
    }

    #[test]
    fn parses_loadavg() {
        let load = parse_loadavg("0.52 0.58 0.59 1/789 12345\n").expect("loadavg");
        assert_eq!(load.one, 0.52);
        assert_eq!(load.five, 0.58);
        assert_eq!(load.fifteen, 0.59);
        assert_eq!(load.running, 1);
        assert_eq!(load.total, 789);
        assert!(parse_loadavg("0.52 0.58\n").is_err());
        assert!(parse_loadavg("a b c 1/2 3\n").is_err());
        assert!(parse_loadavg("0.1 0.2 0.3 1 2\n").is_err());
    }

    #[test]
    fn parses_process_stat_after_parenthesized_comm() {
        let text = "123 (weird) name) S 1 2 3 4 5 6 7 8 9 10 111 222 13 14 15 16 17 18 333 20 444";
        let stat = parse_proc_pid_stat(text).expect("pid stat");
        assert_eq!(stat.state, "S");
        assert_eq!(stat.cpu_ticks, 333);
        assert_eq!(stat.start_time_ticks, 333);
        assert_eq!(stat.threads, 17);
        assert_eq!(stat.rss_pages, 444);
        assert!(parse_proc_pid_stat("garbage").is_err());
    }

    #[test]
    fn parses_process_status_fields() {
        let text = "Name:\tjev\nState:\tS (sleeping)\nVmRSS:\t  2048 kB\nThreads:\t8\n";
        let status = parse_proc_pid_status(text);
        assert_eq!(status.rss_bytes, Some(2048 * 1024));
        assert_eq!(status.threads, Some(8));
        assert_eq!(status.state.as_deref(), Some("S"));
    }
}
