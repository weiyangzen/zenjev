use jev_mq_bridge::{config, sink};
use std::path::PathBuf;
use std::process::ExitCode;

fn usage() {
    eprintln!(
        "jev-mq-bridge {}\n\
         usage:\n\
         \x20 jev-mq-bridge run --config <bridge-config.json>\n\
         \x20 jev-mq-bridge validate --config <bridge-config.json>\n\
         \x20 jev-mq-bridge version",
        env!("CARGO_PKG_VERSION")
    );
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let result = match args.first().map(String::as_str) {
        Some("version") | Some("--version") | Some("-V") => {
            println!("jev-mq-bridge/{}", env!("CARGO_PKG_VERSION"));
            return ExitCode::SUCCESS;
        }
        Some("validate") => with_config(&args).and_then(|config| {
            config.validate()?;
            println!(
                "{}",
                serde_json::json!({
                    "valid": true,
                    "adapter": config.adapter,
                    "schema_digest": config.schema_digest,
                })
            );
            Ok(())
        }),
        Some("run") => with_config(&args).and_then(|config| {
            let mut sink = sink::Sink::open(config)?;
            sink.run()?;
            println!(
                "{}",
                serde_json::to_string(sink.metrics()).unwrap_or_else(|_| "{}".to_string())
            );
            Ok(())
        }),
        _ => {
            usage();
            return ExitCode::from(2);
        }
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("jev-mq-bridge: {error}");
            ExitCode::from(2)
        }
    }
}

fn with_config(args: &[String]) -> Result<config::BridgeConfig, String> {
    let mut path: Option<PathBuf> = None;
    let mut index = 1;
    while index < args.len() {
        match args[index].as_str() {
            "--config" => {
                index += 1;
                let value = args
                    .get(index)
                    .ok_or_else(|| "--config requires a path".to_string())?;
                path = Some(PathBuf::from(value));
            }
            other => return Err(format!("unexpected argument: {other}")),
        }
        index += 1;
    }
    let path = path.ok_or_else(|| "--config is required".to_string())?;
    config::BridgeConfig::from_path(&path)
}
