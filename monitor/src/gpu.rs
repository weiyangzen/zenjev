use serde::Serialize;
use std::process::Command;

/// The monitor intentionally samples the GPUs through `nvidia-smi` (parsed
/// text) instead of linking NVML, so the crate has no native build
/// dependencies. See the README for the NVML TODO.
pub const NVIDIA_SMI_ARGS: [&str; 2] = [
    "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
    "--format=csv,noheader,nounits",
];

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct GpuSample {
    pub index: usize,
    pub utilization_percent: Option<f64>,
    pub memory_used_bytes: Option<u64>,
    pub memory_total_bytes: Option<u64>,
    pub temperature_c: Option<f64>,
    pub power_w: Option<f64>,
}

/// Parse `nvidia-smi --query-gpu=... --format=csv,noheader,nounits` output.
/// One line per GPU: `util, mem_used, mem_total, temp, power`. Fields that are
/// `[N/A]`, `[Not Supported]`, empty, or not numeric become `None`; blank lines
/// are skipped. An empty document yields an empty vector.
pub fn parse_nvidia_smi_csv(output: &str) -> Vec<GpuSample> {
    output
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .enumerate()
        .map(|(index, line)| {
            let fields: Vec<&str> = line.split(',').map(str::trim).collect();
            let value = |position: usize| -> Option<f64> {
                fields.get(position).and_then(|field| parse_value(field))
            };
            GpuSample {
                index,
                utilization_percent: value(0),
                memory_used_bytes: value(1).map(mib_to_bytes),
                memory_total_bytes: value(2).map(mib_to_bytes),
                temperature_c: value(3),
                power_w: value(4),
            }
        })
        .collect()
}

fn parse_value(text: &str) -> Option<f64> {
    let text = text.trim();
    if text.is_empty() || text.starts_with('[') {
        return None;
    }
    text.parse::<f64>().ok().filter(|value| value.is_finite())
}

fn mib_to_bytes(value: f64) -> u64 {
    (value * 1024.0 * 1024.0).round().max(0.0) as u64
}

/// Run `nvidia-smi` once. `None` on a missing binary, a failed exit status, or
/// empty output; never panics and never propagates stderr noise.
pub fn sample_gpu() -> Option<Vec<GpuSample>> {
    let output = Command::new("nvidia-smi")
        .args(NVIDIA_SMI_ARGS)
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&output.stdout);
    let samples = parse_nvidia_smi_csv(&text);
    if samples.is_empty() {
        None
    } else {
        Some(samples)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_multiple_gpus() {
        let text = "12, 1024, 8192, 45, 70.5\n0, 2048, 8192, 50, 100.0\n";
        let gpus = parse_nvidia_smi_csv(text);
        assert_eq!(gpus.len(), 2);
        assert_eq!(gpus[0].index, 0);
        assert_eq!(gpus[0].utilization_percent, Some(12.0));
        assert_eq!(gpus[0].memory_used_bytes, Some(1024 * 1024 * 1024));
        assert_eq!(gpus[0].memory_total_bytes, Some(8192 * 1024 * 1024));
        assert_eq!(gpus[0].temperature_c, Some(45.0));
        assert_eq!(gpus[0].power_w, Some(70.5));
        assert_eq!(gpus[1].index, 1);
        assert_eq!(gpus[1].utilization_percent, Some(0.0));
    }

    #[test]
    fn tolerates_not_available_and_short_lines() {
        let gpus = parse_nvidia_smi_csv("[N/A], [N/A], [N/A], [N/A], [N/A]\n");
        assert_eq!(gpus.len(), 1);
        assert_eq!(gpus[0].utilization_percent, None);
        assert_eq!(gpus[0].memory_used_bytes, None);
        assert_eq!(gpus[0].temperature_c, None);
        assert_eq!(gpus[0].power_w, None);

        let short = parse_nvidia_smi_csv("15, 100\n");
        assert_eq!(short[0].utilization_percent, Some(15.0));
        assert_eq!(short[0].memory_total_bytes, None);
        assert_eq!(short[0].power_w, None);

        let unsupported = parse_nvidia_smi_csv("[Not Supported], 0, 0, [N/A], [N/A]\n");
        assert_eq!(unsupported[0].utilization_percent, None);
        assert_eq!(unsupported[0].temperature_c, None);
    }

    #[test]
    fn empty_output_is_an_empty_sample_list() {
        assert!(parse_nvidia_smi_csv("").is_empty());
        assert!(parse_nvidia_smi_csv("\n  \n").is_empty());
    }

    #[test]
    fn negative_or_garbage_fields_do_not_panic() {
        let gpus = parse_nvidia_smi_csv("abc, -1, 0, garbage, 1e400\n");
        assert_eq!(gpus[0].utilization_percent, None);
        assert_eq!(gpus[0].memory_used_bytes, Some(0));
        assert_eq!(gpus[0].temperature_c, None);
        assert_eq!(gpus[0].power_w, None);
    }
}
