use crate::gpu::GpuSample;
use crate::sample::Snapshot;
use std::io::Write;

const RESET: &str = "\x1b[0m";
const BOLD_CYAN: &str = "\x1b[1;36m";
const BOLD_RED: &str = "\x1b[1;31m";
const LABEL_WIDTH: usize = 7;
const BAR_WIDTH: usize = 20;

/// Render the whole dashboard block. `--no-color` yields the same text with
/// ANSI escapes stripped, which is what the rendering tests pin.
pub fn render_dashboard(snapshot: &Snapshot, color: bool, sample_index: u64) -> String {
    let mut lines = Vec::with_capacity(8);
    lines.push(format!(
        "zenjev-monitor/{}  {}  sample {}",
        env!("CARGO_PKG_VERSION"),
        format_utc_ms(snapshot.sampled_at_unix_ms),
        sample_index
    ));
    lines.push(render_status_line(snapshot, color));
    lines.push(render_cpu_line(snapshot, color));
    lines.extend(render_gpu_lines(snapshot, color));
    lines.push(render_memory_line(snapshot, color));
    lines.push(render_process_line(snapshot, color));
    lines.join("\n")
}

fn render_status_line(snapshot: &Snapshot, color: bool) -> String {
    let detail = match &snapshot.status {
        Some(status) => format!(
            "phase={} gen={} ema={} steps={} loss={} reset={}",
            status.phase.as_deref().unwrap_or("n/a"),
            optional_u64(status.generation),
            optional_u64(status.ema_step),
            optional_u64(status.training_steps),
            optional_f64(status.loss, 4),
            optional_u64(status.reset_id),
        ),
        None => match &snapshot.status_error {
            Some(error) => format!("unavailable ({})", truncate(error, 60)),
            None => "unavailable".to_string(),
        },
    };
    let age = snapshot
        .status_age_seconds
        .map(|age| format!("{age:.1}s"))
        .unwrap_or_else(|| "n/a".to_string());
    let freshness = if snapshot.stale {
        paint("STALE", BOLD_RED, color)
    } else {
        "fresh".to_string()
    };
    format!(
        "{} {}  age={} {}",
        label("status", color),
        detail,
        age,
        freshness
    )
}

fn render_cpu_line(snapshot: &Snapshot, color: bool) -> String {
    let percent = snapshot.cpu.as_ref().and_then(|cpu| cpu.percent);
    let cores = match &snapshot.cpu {
        Some(cpu) => cpu.cores.to_string(),
        None => "n/a".to_string(),
    };
    let load = match &snapshot.load {
        Some(load) => format!("{:.2}/{:.2}/{:.2}", load.one, load.five, load.fifteen),
        None => "n/a".to_string(),
    };
    format!(
        "{} {}  cores={}  load={}",
        label("cpu", color),
        render_bar(percent),
        cores,
        load
    )
}

fn render_gpu_lines(snapshot: &Snapshot, color: bool) -> Vec<String> {
    match &snapshot.gpu {
        Some(gpus) if !gpus.is_empty() => {
            gpus.iter().map(|gpu| render_gpu_line(gpu, color)).collect()
        }
        _ => vec![format!(
            "{} unavailable (nvidia-smi missing or failed)",
            label("gpu", color)
        )],
    }
}

fn render_gpu_line(gpu: &GpuSample, color: bool) -> String {
    let vram_percent = match (gpu.memory_used_bytes, gpu.memory_total_bytes) {
        (Some(used), Some(total)) if total > 0 => Some(used as f64 / total as f64 * 100.0),
        _ => None,
    };
    let vram_detail = match (gpu.memory_used_bytes, gpu.memory_total_bytes) {
        (Some(used), Some(total)) => format!(" {}/{}", format_bytes(used), format_bytes(total)),
        _ => String::new(),
    };
    let temperature = gpu
        .temperature_c
        .map(|value| format!("{value:.0}C"))
        .unwrap_or_else(|| "n/a".to_string());
    let power = gpu
        .power_w
        .map(|value| format!("{value:.1}W"))
        .unwrap_or_else(|| "n/a".to_string());
    format!(
        "{} util {}  vram {}{}  temp={}  power={}",
        label(&format!("gpu{}", gpu.index), color),
        render_bar(gpu.utilization_percent),
        render_bar(vram_percent),
        vram_detail,
        temperature,
        power,
    )
}

fn render_memory_line(snapshot: &Snapshot, color: bool) -> String {
    match &snapshot.memory {
        Some(memory) => format!(
            "{} {}  {}/{}",
            label("mem", color),
            render_bar(Some(memory.used_percent)),
            format_bytes(memory.used_bytes),
            format_bytes(memory.total_bytes)
        ),
        None => format!("{} n/a", label("mem", color)),
    }
}

fn render_process_line(snapshot: &Snapshot, color: bool) -> String {
    match &snapshot.process {
        Some(process) => format!(
            "{} pid={} rss={} cpu={} threads={} state={}",
            label("proc", color),
            process.pid,
            process
                .rss_bytes
                .map(format_bytes)
                .unwrap_or_else(|| "n/a".to_string()),
            match process.cpu_percent {
                Some(percent) => format!("{percent:.1}%"),
                None => "n/a".to_string(),
            },
            optional_u64(process.threads),
            process.state.as_deref().unwrap_or("?"),
        ),
        None => match snapshot.status.as_ref().and_then(|status| status.pid) {
            Some(pid) => format!("{} pid={pid} not running", label("proc", color)),
            None => format!("{} n/a", label("proc", color)),
        },
    }
}

/// `[#####---------------]  25.0%`; `n/a` when the metric is unavailable.
pub fn render_bar(percent: Option<f64>) -> String {
    match percent {
        Some(percent) => {
            let percent = percent.clamp(0.0, 100.0);
            let filled = ((percent / 100.0) * BAR_WIDTH as f64).round() as usize;
            let filled = filled.min(BAR_WIDTH);
            format!(
                "[{}{}] {:5.1}%",
                "#".repeat(filled),
                "-".repeat(BAR_WIDTH - filled),
                percent
            )
        }
        None => format!("[{}]   n/a", "-".repeat(BAR_WIDTH)),
    }
}

/// Binary byte units, deterministic formatting for evidence/JSONL dashboards.
pub fn format_bytes(bytes: u64) -> String {
    const KIB: f64 = 1024.0;
    const MIB: f64 = 1024.0 * 1024.0;
    const GIB: f64 = 1024.0 * 1024.0 * 1024.0;
    let value = bytes as f64;
    if value >= GIB {
        format!("{:.2}GiB", value / GIB)
    } else if value >= MIB {
        format!("{:.1}MiB", value / MIB)
    } else if value >= KIB {
        format!("{:.1}KiB", value / KIB)
    } else {
        format!("{bytes}B")
    }
}

/// UTC timestamp without a date library (Howard Hinnant's civil-from-days).
pub fn format_utc_ms(unix_ms: u128) -> String {
    let seconds = (unix_ms / 1000) as i64;
    let millis = (unix_ms % 1000) as u32;
    let days = seconds.div_euclid(86_400);
    let second_of_day = seconds.rem_euclid(86_400);
    let (year, month, day) = civil_from_days(days);
    format!(
        "{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}.{millis:03}Z",
        second_of_day / 3600,
        (second_of_day % 3600) / 60,
        second_of_day % 60
    )
}

fn civil_from_days(days_since_epoch: i64) -> (i64, u32, u32) {
    let shifted = days_since_epoch + 719_468;
    let era = if shifted >= 0 {
        shifted
    } else {
        shifted - 146_096
    } / 146_097;
    let day_of_era = (shifted - era * 146_097) as u64;
    let year_of_era =
        (day_of_era - day_of_era / 1460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let year = year_of_era as i64 + era * 400;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let month_prime = (5 * day_of_year + 2) / 153;
    let day = (day_of_year - (153 * month_prime + 2) / 5 + 1) as u32;
    let month = if month_prime < 10 {
        month_prime + 3
    } else {
        month_prime - 9
    } as u32;
    (if month <= 2 { year + 1 } else { year }, month, day)
}

fn label(text: &str, color: bool) -> String {
    paint(
        &format!("{text:<width$}", width = LABEL_WIDTH),
        BOLD_CYAN,
        color,
    )
}

fn paint(text: &str, code: &str, color: bool) -> String {
    if color {
        format!("{code}{text}{RESET}")
    } else {
        text.to_string()
    }
}

fn optional_u64(value: Option<u64>) -> String {
    value
        .map(|value| value.to_string())
        .unwrap_or_else(|| "n/a".to_string())
}

fn optional_f64(value: Option<f64>, precision: usize) -> String {
    match value {
        Some(value) => format!("{value:.precision$}"),
        None => "n/a".to_string(),
    }
}

fn truncate(text: &str, limit: usize) -> String {
    if text.chars().count() <= limit {
        return text.to_string();
    }
    let mut shortened: String = text.chars().take(limit).collect();
    shortened.push_str("...");
    shortened
}

/// In-place dashboard refresher: moves the cursor back over the previously
/// printed block, clears each line, and prints the new block. The cursor is
/// never hidden and the alternate screen is never entered, so an abrupt
/// SIGINT/SIGTERM termination cannot leave the terminal unusable.
pub struct Dashboard {
    printed_lines: usize,
}

impl Dashboard {
    pub fn new() -> Self {
        Self { printed_lines: 0 }
    }

    pub fn draw(&mut self, block: &str) -> std::io::Result<()> {
        let (sequence, printed) = refresh_sequence(block, self.printed_lines);
        self.printed_lines = printed;
        let mut stdout = std::io::stdout().lock();
        stdout.write_all(sequence.as_bytes())?;
        stdout.flush()
    }
}

impl Default for Dashboard {
    fn default() -> Self {
        Self::new()
    }
}

fn refresh_sequence(block: &str, printed_lines: usize) -> (String, usize) {
    let lines: Vec<&str> = block.lines().collect();
    let target = lines.len().max(printed_lines);
    let mut output = String::new();
    if printed_lines > 0 {
        output.push_str(&format!("\x1b[{printed_lines}A"));
    }
    for index in 0..target {
        output.push_str("\x1b[2K");
        if let Some(line) = lines.get(index) {
            output.push_str(line);
        }
        output.push('\n');
    }
    (output, target)
}

/// Resets ANSI attributes and emits a final newline on normal exit or panic.
/// It does not run when the process is killed directly by SIGINT/SIGTERM
/// (no signal handler is installed), which is safe because the dashboard never
/// hides the cursor or uses the alternate screen. See the README.
pub struct TerminalGuard {
    active: bool,
}

impl TerminalGuard {
    pub fn new(active: bool) -> Self {
        Self { active }
    }
}

impl Drop for TerminalGuard {
    fn drop(&mut self) {
        if !self.active {
            return;
        }
        let mut stdout = std::io::stdout();
        let _ = stdout.write_all(b"\x1b[0m\n");
        let _ = stdout.flush();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::GpuSample;
    use crate::process::ProcessSample;
    use crate::procfs::{CpuSample, LoadSample, MemorySample};
    use crate::sample::Snapshot;
    use crate::status::StatusFile;

    fn sample_snapshot() -> Snapshot {
        Snapshot {
            monitor_schema_version: 1,
            sampled_at_unix_ms: 1_700_000_000_000,
            cpu: Some(CpuSample {
                percent: Some(50.0),
                cores: 16,
                per_core_percent: vec![Some(50.0); 16],
            }),
            load: Some(LoadSample {
                one: 0.5,
                five: 0.6,
                fifteen: 0.7,
                running: 2,
                total: 400,
            }),
            memory: Some(MemorySample {
                total_bytes: 16 * 1024 * 1024 * 1024,
                used_bytes: 8 * 1024 * 1024 * 1024,
                available_bytes: 8 * 1024 * 1024 * 1024,
                used_percent: 50.0,
            }),
            gpu: Some(vec![GpuSample {
                index: 0,
                utilization_percent: Some(50.0),
                memory_used_bytes: Some(4 * 1024 * 1024 * 1024),
                memory_total_bytes: Some(8 * 1024 * 1024 * 1024),
                temperature_c: Some(61.0),
                power_w: Some(120.0),
            }]),
            process: Some(ProcessSample {
                pid: 12345,
                rss_bytes: Some(1536 * 1024 * 1024),
                cpu_percent: Some(12.3),
                threads: Some(8),
                state: Some("S".to_string()),
            }),
            status: Some(StatusFile {
                schema_version: 1,
                updated_at_unix_ms: 1_700_000_000_000,
                pid: Some(12345),
                phase: Some("training".to_string()),
                generation: Some(9),
                ema_step: Some(8),
                training_steps: Some(8),
                reset_id: Some(0),
                loss: Some(1.57),
                learning_rate: Some(0.0001),
                checkpoint: Some("runs/jev/adapter-step-00000008.pt".to_string()),
                config_digest: None,
                model_id: Some("fastino/gliner2-base-v1".to_string()),
                device: Some("cuda".to_string()),
                note: None,
            }),
            status_age_seconds: Some(0.8),
            status_error: None,
            stale: false,
        }
    }

    #[test]
    fn renders_bars() {
        assert_eq!(render_bar(Some(0.0)), "[--------------------]   0.0%");
        assert_eq!(render_bar(Some(50.0)), "[##########----------]  50.0%");
        assert_eq!(render_bar(Some(100.0)), "[####################] 100.0%");
        assert_eq!(render_bar(Some(150.0)), "[####################] 100.0%");
        assert_eq!(render_bar(None), "[--------------------]   n/a");
    }

    #[test]
    fn formats_bytes() {
        assert_eq!(format_bytes(512), "512B");
        assert_eq!(format_bytes(2048), "2.0KiB");
        assert_eq!(format_bytes(1536 * 1024), "1.5MiB");
        assert_eq!(format_bytes(3 * 1024 * 1024 * 1024), "3.00GiB");
    }

    #[test]
    fn formats_utc_timestamps() {
        assert_eq!(format_utc_ms(0), "1970-01-01T00:00:00.000Z");
        assert_eq!(format_utc_ms(1_700_000_000_000), "2023-11-14T22:13:20.000Z");
        assert_eq!(format_utc_ms(1_709_251_200_000), "2024-03-01T00:00:00.000Z");
        assert_eq!(format_utc_ms(999), "1970-01-01T00:00:00.999Z");
    }

    #[test]
    fn renders_stable_no_color_dashboard() {
        let snapshot = sample_snapshot();
        let text = render_dashboard(&snapshot, false, 1);
        assert!(!text.contains('\x1b'));
        assert!(text.starts_with("zenjev-monitor/0.1.0  2023-11-14T22:13:20.000Z  sample 1"));
        assert!(text.contains("phase=training gen=9 ema=8 steps=8 loss=1.5700 reset=0"));
        assert!(text.contains("age=0.8s fresh"));
        assert!(
            text.contains("cpu     [##########----------]  50.0%  cores=16  load=0.50/0.60/0.70")
        );
        assert!(text.contains("gpu0    util [##########----------]  50.0%"));
        assert!(text.contains("temp=61C  power=120.0W"));
        assert!(text.contains("proc    pid=12345 rss=1.50GiB cpu=12.3% threads=8 state=S"));
    }

    #[test]
    fn dashboard_renders_stale_and_unavailable_without_panicking() {
        let mut snapshot = sample_snapshot();
        snapshot.status_age_seconds = Some(30.0);
        snapshot.stale = true;
        snapshot.gpu = None;
        snapshot.process = None;
        snapshot.cpu = None;
        snapshot.load = None;
        snapshot.memory = None;
        let text = render_dashboard(&snapshot, false, 7);
        assert!(text.contains("STALE"));
        assert!(text.contains("gpu     unavailable (nvidia-smi missing or failed)"));
        assert!(text.contains("cpu     [--------------------]   n/a  cores=n/a  load=n/a"));
        assert!(text.contains("mem     n/a"));
        assert!(text.contains("proc    pid=12345 not running"));

        let mut missing = sample_snapshot();
        missing.status = None;
        missing.status_age_seconds = None;
        missing.status_error = Some("status file not found: runs/jev/status.json".to_string());
        missing.stale = true;
        missing.process = None;
        let text = render_dashboard(&missing, false, 1);
        assert!(text.contains("status  unavailable (status file not found: runs/jev/status.json)"));
        assert!(text.contains("proc    n/a"));
    }

    #[test]
    fn color_mode_uses_ansi_escapes() {
        let text = render_dashboard(&sample_snapshot(), true, 1);
        assert!(text.contains("\x1b[1;36m"));
        assert!(text.contains("\x1b[0m"));
    }

    #[test]
    fn refresh_sequence_moves_cursor_and_clears_lines() {
        let (sequence, printed) = refresh_sequence("a\nb", 0);
        assert_eq!(printed, 2);
        assert_eq!(sequence, "\x1b[2Ka\n\x1b[2Kb\n");

        let (sequence, printed) = refresh_sequence("a", 3);
        assert_eq!(printed, 3);
        assert!(sequence.starts_with("\x1b[3A"));
        assert_eq!(sequence.matches("\x1b[2K").count(), 3);
    }
}
