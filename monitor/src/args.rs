use std::path::PathBuf;

pub const USAGE: &str = concat!(
    "zenjev-monitor ",
    env!("CARGO_PKG_VERSION"),
    " — live observability for the ZenJev train/infer service

usage:
  zenjev-monitor [options]

options:
  --status-path PATH     status JSON written by the Python service
                         (default: runs/jev/status.json)
  --interval SECONDS     sampling period (default: 1.0)
  --stale-after SECONDS  mark status stale after this age (default: 15.0)
  --format FORMAT        dashboard | json (default: dashboard)
  --jsonl PATH           append every sample as one JSON object per line
  --once                 sample exactly once and exit 0 (CI friendly)
  --duration SECONDS     stop after this many seconds; 0 runs until interrupted
                         (default: 0)
  --no-color             disable ANSI colors in the dashboard
  -h, --help             print this help
  -V, --version          print the version

exit codes:
  0  normal stop, or one successful --once sample
  2  invalid arguments
"
);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Format {
    Dashboard,
    Json,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Config {
    pub status_path: PathBuf,
    pub interval_seconds: f64,
    pub stale_after_seconds: f64,
    pub format: Format,
    pub jsonl: Option<PathBuf>,
    pub once: bool,
    pub duration_seconds: f64,
    pub color: bool,
    pub help: bool,
    pub version: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            status_path: PathBuf::from("runs/jev/status.json"),
            interval_seconds: 1.0,
            stale_after_seconds: 15.0,
            format: Format::Dashboard,
            jsonl: None,
            once: false,
            duration_seconds: 0.0,
            color: true,
            help: false,
            version: false,
        }
    }
}

/// Parse the hand-rolled CLI. Both `--flag value` and `--flag=value` are
/// accepted; booleans never take a value. `Err` means exit code 2.
pub fn parse(args: &[String]) -> Result<Config, String> {
    let mut config = Config::default();
    let mut index = 0;
    while index < args.len() {
        let arg = args[index].as_str();
        let (name, inline) = match arg.split_once('=') {
            Some((name, value)) => (name, Some(value.to_string())),
            None => (arg, None),
        };
        match name {
            "--status-path" => {
                config.status_path = PathBuf::from(value_for(args, &mut index, inline, name)?)
            }
            "--interval" => {
                config.interval_seconds =
                    positive_number(&value_for(args, &mut index, inline, name)?, name, false)?
            }
            "--stale-after" => {
                config.stale_after_seconds =
                    positive_number(&value_for(args, &mut index, inline, name)?, name, true)?
            }
            "--duration" => {
                config.duration_seconds =
                    positive_number(&value_for(args, &mut index, inline, name)?, name, true)?
            }
            "--format" => {
                let value = value_for(args, &mut index, inline, name)?;
                config.format = match value.as_str() {
                    "dashboard" => Format::Dashboard,
                    "json" => Format::Json,
                    other => {
                        return Err(format!("--format expects dashboard or json, got: {other}"))
                    }
                };
            }
            "--jsonl" => {
                config.jsonl = Some(PathBuf::from(value_for(args, &mut index, inline, name)?))
            }
            "--once" => {
                reject_value(name, &inline)?;
                config.once = true;
            }
            "--no-color" => {
                reject_value(name, &inline)?;
                config.color = false;
            }
            "-h" | "--help" => {
                reject_value(name, &inline)?;
                config.help = true;
            }
            "-V" | "--version" => {
                reject_value(name, &inline)?;
                config.version = true;
            }
            other => return Err(format!("unknown argument: {other}")),
        }
        index += 1;
    }
    Ok(config)
}

fn value_for(
    args: &[String],
    index: &mut usize,
    inline: Option<String>,
    flag: &str,
) -> Result<String, String> {
    if let Some(value) = inline {
        return Ok(value);
    }
    *index += 1;
    args.get(*index)
        .cloned()
        .ok_or_else(|| format!("{flag} requires a value"))
}

fn reject_value(flag: &str, inline: &Option<String>) -> Result<(), String> {
    if inline.is_some() {
        return Err(format!("{flag} does not take a value"));
    }
    Ok(())
}

fn positive_number(text: &str, flag: &str, allow_zero: bool) -> Result<f64, String> {
    let value = text
        .parse::<f64>()
        .map_err(|_| format!("{flag} expects a number, got: {text}"))?;
    if !value.is_finite() || value < 0.0 || (!allow_zero && value == 0.0) {
        return Err(format!(
            "{flag} expects a {} number, got: {text}",
            if allow_zero {
                "non-negative"
            } else {
                "positive"
            }
        ));
    }
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn strings(values: &[&str]) -> Vec<String> {
        values.iter().map(|value| value.to_string()).collect()
    }

    #[test]
    fn defaults_are_stable() {
        let config = parse(&[]).expect("defaults");
        assert_eq!(config.status_path, PathBuf::from("runs/jev/status.json"));
        assert_eq!(config.interval_seconds, 1.0);
        assert_eq!(config.stale_after_seconds, 15.0);
        assert_eq!(config.format, Format::Dashboard);
        assert!(config.jsonl.is_none());
        assert!(!config.once);
        assert_eq!(config.duration_seconds, 0.0);
        assert!(config.color);
    }

    #[test]
    fn parses_every_flag_in_both_forms() {
        let config = parse(&strings(&[
            "--status-path",
            "/tmp/status.json",
            "--interval=2.5",
            "--stale-after",
            "9",
            "--format",
            "json",
            "--jsonl",
            "/tmp/out.jsonl",
            "--once",
            "--duration=12.5",
            "--no-color",
        ]))
        .expect("valid args");
        assert_eq!(config.status_path, PathBuf::from("/tmp/status.json"));
        assert_eq!(config.interval_seconds, 2.5);
        assert_eq!(config.stale_after_seconds, 9.0);
        assert_eq!(config.format, Format::Json);
        assert_eq!(config.jsonl, Some(PathBuf::from("/tmp/out.jsonl")));
        assert!(config.once);
        assert_eq!(config.duration_seconds, 12.5);
        assert!(!config.color);
    }

    #[test]
    fn help_and_version_short_flags() {
        assert!(parse(&strings(&["-h"])).expect("help").help);
        assert!(parse(&strings(&["--help"])).expect("help").help);
        assert!(parse(&strings(&["-V"])).expect("version").version);
        assert!(parse(&strings(&["--version"])).expect("version").version);
    }

    #[test]
    fn rejects_bad_arguments() {
        assert!(parse(&strings(&["--interval", "0"])).is_err());
        assert!(parse(&strings(&["--interval", "-1"])).is_err());
        assert!(parse(&strings(&["--stale-after", "nan"])).is_err());
        assert!(parse(&strings(&["--format", "yaml"])).is_err());
        assert!(parse(&strings(&["--once=true"])).is_err());
        assert!(parse(&strings(&["--jsonl"])).is_err());
        assert!(parse(&strings(&["--nope"])).is_err());
        assert!(parse(&strings(&["--duration", "abc"])).is_err());
    }

    #[test]
    fn accepts_zero_for_stale_after_and_duration() {
        let config = parse(&strings(&["--stale-after", "0", "--duration", "0"])).expect("zero ok");
        assert_eq!(config.stale_after_seconds, 0.0);
        assert_eq!(config.duration_seconds, 0.0);
    }
}
