use std::process::ExitCode;
use std::time::{Duration, Instant};
use zenjev_monitor::{args, dashboard, output, sample, unix_ms};

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let config = match args::parse(&args) {
        Ok(config) => config,
        Err(error) => {
            eprintln!("zenjev-monitor: {error}");
            eprintln!("{USAGE_HINT}");
            return ExitCode::from(2);
        }
    };
    if config.help {
        println!("{}", args::USAGE);
        return ExitCode::SUCCESS;
    }
    if config.version {
        println!("zenjev-monitor/{}", env!("CARGO_PKG_VERSION"));
        return ExitCode::SUCCESS;
    }
    run(&config)
}

const USAGE_HINT: &str = "run `zenjev-monitor --help` for usage";

fn run(config: &args::Config) -> ExitCode {
    let dashboard_active = config.format == args::Format::Dashboard;
    let _guard = dashboard::TerminalGuard::new(dashboard_active);
    let mut sampler = sample::Sampler::new(config.status_path.clone(), config.stale_after_seconds);
    let mut live = dashboard::Dashboard::new();
    let started = Instant::now();
    let mut sample_index = 0_u64;
    loop {
        sample_index += 1;
        let snapshot = sampler.sample(unix_ms());
        if let Some(path) = &config.jsonl {
            if let Err(error) = output::append_jsonl(path, &snapshot) {
                eprintln!("zenjev-monitor: cannot append {}: {error}", path.display());
            }
        }
        match config.format {
            args::Format::Json => match serde_json::to_string(&snapshot) {
                Ok(line) => println!("{line}"),
                Err(error) => eprintln!("zenjev-monitor: cannot serialize sample: {error}"),
            },
            args::Format::Dashboard => {
                let block = dashboard::render_dashboard(&snapshot, config.color, sample_index);
                if let Err(error) = live.draw(&block) {
                    eprintln!("zenjev-monitor: cannot write dashboard: {error}");
                }
            }
        }
        if config.once {
            break;
        }
        let elapsed = started.elapsed().as_secs_f64();
        if config.duration_seconds > 0.0 && elapsed >= config.duration_seconds {
            break;
        }
        let wait = if config.duration_seconds > 0.0 {
            (config.duration_seconds - elapsed).min(config.interval_seconds)
        } else {
            config.interval_seconds
        };
        if wait > 0.0 {
            std::thread::sleep(Duration::from_secs_f64(wait));
        }
    }
    ExitCode::SUCCESS
}
