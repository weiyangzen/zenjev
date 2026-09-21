use std::process::Command;

/// `--once --format json` is the CI contract: exit 0 even when no status file
/// exists, and emit exactly one JSON object with `status: null`.
#[test]
fn once_json_reports_null_status_with_exit_zero() {
    let mut status = std::env::temp_dir();
    status.push(format!(
        "zenjev-monitor-cli-{}-absent-status.json",
        std::process::id()
    ));
    let _ = std::fs::remove_file(&status);

    let output = Command::new(env!("CARGO_BIN_EXE_zenjev-monitor"))
        .args(["--once", "--format", "json", "--no-color", "--status-path"])
        .arg(&status)
        .output()
        .expect("run zenjev-monitor");

    assert!(output.status.success(), "exit status: {:?}", output.status);
    let stdout = String::from_utf8_lossy(&output.stdout);
    let lines: Vec<&str> = stdout.lines().collect();
    assert_eq!(lines.len(), 1, "expected exactly one JSON line: {stdout}");
    let value: serde_json::Value = serde_json::from_str(lines[0]).expect("valid JSON sample");
    assert_eq!(value["monitor_schema_version"], serde_json::json!(1));
    assert!(value["status"].is_null());
    assert_eq!(value["stale"], serde_json::json!(true));
    assert!(value["sampled_at_unix_ms"].is_u64());
}

#[test]
fn invalid_argument_exits_two() {
    let output = Command::new(env!("CARGO_BIN_EXE_zenjev-monitor"))
        .args(["--definitely-not-a-flag"])
        .output()
        .expect("run zenjev-monitor");
    assert_eq!(output.status.code(), Some(2));
}
