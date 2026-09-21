//! Protocol-level integration tests for the bridge binary.
//!
//! These tests use the deterministic `mock` source so at-least-once replay,
//! duplicate suppression, DLQ/quarantine routing, credit backpressure and
//! drain semantics can be proven without a live broker.

use jev_mq_bridge::canonical::envelope_content_hash;
use serde_json::{json, Value};
use std::io::{Read, Write};
use std::os::unix::net::UnixStream;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

const SCHEMA_ID: &str = "stage0_test";
const SCHEMA_VERSION: u64 = 1;

fn schema_digest() -> String {
    "a".repeat(64)
}

fn envelope(record_id: &str) -> Value {
    let mut value = json!({
        "envelope_version": 1,
        "record_id": record_id,
        "content_sha256": "",
        "schema": {"id": SCHEMA_ID, "version": SCHEMA_VERSION, "digest": schema_digest()},
        "kind": "raw_document",
        "source": {
            "uri": format!("test://{record_id}"),
            "content_sha256": "b".repeat(64),
            "retrieved_at": "2026-09-21T00:00:00Z",
            "license": "test"
        },
        "document": {"text": record_id},
        "created_at": "2026-09-21T00:00:00Z"
    });
    let hash = envelope_content_hash(&value).expect("hash");
    value["content_sha256"] = Value::String(hash);
    value
}

struct Harness {
    directory: PathBuf,
    child: Child,
    socket: PathBuf,
}

impl Harness {
    fn start(name: &str, records: &[Vec<u8>], exit_after_drain: bool) -> Self {
        Self::start_with(name, records, exit_after_drain, false, 0)
    }

    fn start_with(
        name: &str,
        records: &[Vec<u8>],
        exit_after_drain: bool,
        loop_enabled: bool,
        max_cycles: u64,
    ) -> Self {
        let directory = std::env::temp_dir().join(format!(
            "jev-mq-it-{name}-{}-{}",
            std::process::id(),
            Instant::now().elapsed().as_nanos()
        ));
        let _ = std::fs::remove_dir_all(&directory);
        std::fs::create_dir_all(&directory).unwrap();
        let source = directory.join("records.jsonl");
        let mut body = String::new();
        for record in records {
            body.push_str(&String::from_utf8_lossy(record));
            body.push('\n');
        }
        std::fs::write(&source, body).unwrap();
        let socket = directory.join("bridge.sock");
        let config = directory.join("bridge.json");
        let value = json!({
            "adapter": "mock",
            "source_path": source,
            "state_path": directory.join("state.json"),
            "socket_path": socket,
            "wal_path": directory.join("wal.bin"),
            "dlq_path": directory.join("dlq.jsonl"),
            "quarantine_path": directory.join("quarantine.jsonl"),
            "metrics_path": directory.join("metrics.json"),
            "schema_id": SCHEMA_ID,
            "schema_version": SCHEMA_VERSION,
            "schema_digest": schema_digest(),
            "max_bytes": 1048576,
            "dedup_window": 64,
            "batch": 8,
            "max_ack_pending": 8,
            "max_deliver": 3,
            "loop": loop_enabled,
            "max_cycles": max_cycles,
            "exit_after_drain": exit_after_drain
        });
        std::fs::write(&config, serde_json::to_string_pretty(&value).unwrap()).unwrap();
        let child = Command::new(env!("CARGO_BIN_EXE_jev-mq-bridge"))
            .arg("run")
            .arg("--config")
            .arg(&config)
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .expect("spawn bridge");
        let deadline = Instant::now() + Duration::from_secs(10);
        while !socket.exists() {
            assert!(Instant::now() < deadline, "bridge socket never appeared");
            std::thread::sleep(Duration::from_millis(20));
        }
        Self {
            directory,
            child,
            socket,
        }
    }

    fn connect(&self) -> UnixStream {
        let stream = UnixStream::connect(&self.socket).expect("connect");
        stream
            .set_read_timeout(Some(Duration::from_secs(5)))
            .unwrap();
        stream
    }

    fn state_committed(&self) -> u64 {
        let path = self.directory.join("state.json");
        let Ok(raw) = std::fs::read_to_string(path) else {
            return 0;
        };
        serde_json::from_str::<Value>(&raw)
            .ok()
            .and_then(|value| value.get("committed").and_then(Value::as_u64))
            .unwrap_or(0)
    }
}

impl Drop for Harness {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
        let _ = std::fs::remove_dir_all(&self.directory);
    }
}

fn connect_retry(socket: &Path) -> UnixStream {
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        match UnixStream::connect(socket) {
            Ok(stream) => {
                stream
                    .set_read_timeout(Some(Duration::from_secs(5)))
                    .unwrap();
                return stream;
            }
            Err(error) => {
                assert!(Instant::now() < deadline, "connect retry failed: {error}");
                std::thread::sleep(Duration::from_millis(50));
            }
        }
    }
}

fn write_frame(stream: &mut UnixStream, value: &Value) {
    let body = serde_json::to_vec(value).unwrap();
    stream
        .write_all(&(body.len() as u32).to_be_bytes())
        .unwrap();
    stream.write_all(&body).unwrap();
    stream.flush().unwrap();
}

fn read_frame(stream: &mut UnixStream) -> Option<Value> {
    let mut length = [0u8; 4];
    if stream.read_exact(&mut length).is_err() {
        return None;
    }
    let length = u32::from_be_bytes(length) as usize;
    let mut body = vec![0u8; length];
    stream.read_exact(&mut body).ok()?;
    serde_json::from_slice(&body).ok()
}

fn read_until(stream: &mut UnixStream, kind: &str) -> Value {
    for _ in 0..64 {
        let Some(value) = read_frame(stream) else {
            panic!("connection closed while waiting for {kind}");
        };
        if value.get("type").and_then(Value::as_str) == Some(kind) {
            return value;
        }
    }
    panic!("did not observe {kind}");
}

fn text_record(value: &Value) -> String {
    value["envelope"]["record_id"].as_str().unwrap().to_string()
}

#[test]
fn ack_after_durable_acceptance_and_backpressure() {
    let harness = Harness::start(
        "ack",
        &[
            serde_json::to_vec(&envelope("r1")).unwrap(),
            serde_json::to_vec(&envelope("r2")).unwrap(),
        ],
        true,
    );
    let mut stream = harness.connect();
    write_frame(
        &mut stream,
        &json!({"type": "hello", "protocol": 1, "credits": 1}),
    );
    read_until(&mut stream, "ready");
    let first = read_until(&mut stream, "record");
    assert_eq!(text_record(&first), "r1");
    // One credit means exactly one in-flight record; the bridge must not pull
    // ahead of the client.
    assert_eq!(
        harness.state_committed(),
        0,
        "ack must follow durable acceptance"
    );
    let mut probe = stream.try_clone().unwrap();
    probe
        .set_read_timeout(Some(Duration::from_millis(250)))
        .unwrap();
    assert!(
        read_frame(&mut probe).is_none(),
        "bridge ignored credit bound"
    );
    write_frame(
        &mut stream,
        &json!({"type": "credit", "count": 1, "acks": [first["delivery_id"]]}),
    );
    let second = read_until(&mut stream, "record");
    assert_eq!(text_record(&second), "r2");
    let deadline = Instant::now() + Duration::from_secs(5);
    while harness.state_committed() < 1 && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(20));
    }
    assert!(harness.state_committed() >= 1);
    write_frame(
        &mut stream,
        &json!({"type": "credit", "count": 1, "acks": [second["delivery_id"]]}),
    );
    write_frame(&mut stream, &json!({"type": "drain"}));
    let drained = read_until(&mut stream, "drained");
    assert_eq!(drained["inflight"], 0);
    write_frame(&mut stream, &json!({"type": "goodbye"}));
}

#[test]
fn poison_dlq_schema_quarantine_and_duplicate_suppression() {
    let mismatch = {
        let mut value = envelope("mismatch");
        value["schema"]["digest"] = Value::String("c".repeat(64));
        let hash = envelope_content_hash(&value).unwrap();
        value["content_sha256"] = Value::String(hash);
        value
    };
    let records = vec![
        serde_json::to_vec(&envelope("r1")).unwrap(),
        b"{\"envelope_version\": 1, \"record_id\":".to_vec(),
        serde_json::to_vec(&mismatch).unwrap(),
        serde_json::to_vec(&envelope("r2")).unwrap(),
        serde_json::to_vec(&envelope("r1")).unwrap(),
    ];
    let harness = Harness::start("faults", &records, true);
    let mut stream = harness.connect();
    write_frame(
        &mut stream,
        &json!({"type": "hello", "protocol": 1, "credits": 2}),
    );
    read_until(&mut stream, "ready");
    let mut seen = Vec::new();
    // First credit window: r1 plus the poison record (DLQed without delivery).
    let first = read_until(&mut stream, "record");
    seen.push(text_record(&first));
    write_frame(
        &mut stream,
        &json!({"type": "credit", "count": 2, "acks": [first["delivery_id"]]}),
    );
    // Second window: schema mismatch is quarantined, r2 is delivered.
    let second = read_until(&mut stream, "record");
    seen.push(text_record(&second));
    write_frame(
        &mut stream,
        &json!({"type": "credit", "count": 1, "acks": [second["delivery_id"]]}),
    );
    // Third window: the repeated r1 is suppressed as a duplicate.
    std::thread::sleep(Duration::from_millis(500));
    write_frame(&mut stream, &json!({"type": "drain"}));
    read_until(&mut stream, "drained");
    write_frame(&mut stream, &json!({"type": "goodbye"}));
    assert_eq!(seen, vec!["r1".to_string(), "r2".to_string()]);
    let dlq = std::fs::read_to_string(harness.directory.join("dlq.jsonl")).unwrap_or_default();
    assert!(
        dlq.contains("envelope_not_json"),
        "poison must reach DLQ: {dlq}"
    );
    let quarantine =
        std::fs::read_to_string(harness.directory.join("quarantine.jsonl")).unwrap_or_default();
    assert!(
        quarantine.contains("schema_digest_mismatch"),
        "schema mismatch must be quarantined: {quarantine}"
    );
    let metrics_path = harness.directory.join("metrics.json");
    let deadline = Instant::now() + Duration::from_secs(5);
    while !metrics_path.exists() && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(20));
    }
    let metrics = std::fs::read_to_string(metrics_path).unwrap();
    assert!(metrics.contains("duplicates_suppressed"));
    assert!(metrics.contains("\"dlq\": 1") || metrics.contains("\"dlq\": 2"));
}

fn read_metrics_until(harness: &Harness, predicate: impl Fn(&Value) -> bool) -> Value {
    let path = harness.directory.join("metrics.json");
    let deadline = Instant::now() + Duration::from_secs(5);
    while Instant::now() < deadline {
        if let Ok(raw) = std::fs::read_to_string(&path) {
            if let Ok(value) = serde_json::from_str::<Value>(&raw) {
                if predicate(&value) {
                    return value;
                }
            }
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    panic!("bridge metrics predicate was never satisfied");
}

fn loop_cycle_of(frame: &Value) -> Option<u64> {
    frame["envelope"].get("loop_cycle").and_then(Value::as_u64)
}

fn content_hash(frame: &Value) -> String {
    frame["envelope"]["content_sha256"]
        .as_str()
        .expect("content_sha256")
        .to_string()
}

#[test]
fn loop_mode_replays_cycles_with_distinct_keys_and_stops_at_max_cycles() {
    let harness = Harness::start_with(
        "loop",
        &[
            serde_json::to_vec(&envelope("r1")).unwrap(),
            serde_json::to_vec(&envelope("r2")).unwrap(),
        ],
        true,
        true,
        3,
    );
    let mut stream = harness.connect();
    write_frame(
        &mut stream,
        &json!({"type": "hello", "protocol": 1, "credits": 16}),
    );
    read_until(&mut stream, "ready");

    let mut frames = Vec::new();
    for _ in 0..6 {
        frames.push(read_until(&mut stream, "record"));
    }
    let mut probe = stream.try_clone().unwrap();
    probe
        .set_read_timeout(Some(Duration::from_millis(250)))
        .unwrap();
    assert!(
        read_frame(&mut probe).is_none(),
        "max_cycles must exhaust the source"
    );

    let record_ids: Vec<String> = frames.iter().map(text_record).collect();
    assert_eq!(record_ids, vec!["r1", "r2", "r1", "r2", "r1", "r2"]);
    let envelope_cycles: Vec<Option<u64>> = frames.iter().map(loop_cycle_of).collect();
    assert_eq!(
        envelope_cycles,
        vec![None, None, Some(1), Some(1), Some(2), Some(2)],
        "cycle 0 must stay unchanged and later cycles carry loop_cycle"
    );
    let frame_cycles: Vec<u64> = frames
        .iter()
        .map(|frame| frame["loop_cycle"].as_u64().expect("frame loop_cycle"))
        .collect();
    assert_eq!(frame_cycles, vec![0, 0, 1, 1, 2, 2]);
    assert_ne!(content_hash(&frames[0]), content_hash(&frames[2]));
    assert_ne!(content_hash(&frames[0]), content_hash(&frames[4]));
    assert_ne!(content_hash(&frames[2]), content_hash(&frames[4]));
    assert_ne!(content_hash(&frames[1]), content_hash(&frames[3]));

    for frame in &frames {
        write_frame(
            &mut stream,
            &json!({"type": "credit", "count": 1, "acks": [frame["delivery_id"]]}),
        );
    }
    write_frame(&mut stream, &json!({"type": "drain"}));
    read_until(&mut stream, "drained");
    write_frame(&mut stream, &json!({"type": "goodbye"}));

    let metrics = read_metrics_until(&harness, |value| {
        value["delivered"].as_u64() == Some(6) && value["loop_cycles"].as_u64() == Some(2)
    });
    assert_eq!(
        metrics["duplicates_suppressed"].as_u64(),
        Some(0),
        "distinct cycle keys must not be suppressed"
    );
    assert_eq!(metrics["pulled"].as_u64(), Some(6));
}

#[test]
fn loop_mode_max_cycles_two_stops_after_four_deliveries() {
    let harness = Harness::start_with(
        "loopstop",
        &[
            serde_json::to_vec(&envelope("r1")).unwrap(),
            serde_json::to_vec(&envelope("r2")).unwrap(),
        ],
        true,
        true,
        2,
    );
    let mut stream = harness.connect();
    write_frame(
        &mut stream,
        &json!({"type": "hello", "protocol": 1, "credits": 16}),
    );
    read_until(&mut stream, "ready");

    let mut frames = Vec::new();
    for _ in 0..4 {
        frames.push(read_until(&mut stream, "record"));
    }
    let mut probe = stream.try_clone().unwrap();
    probe
        .set_read_timeout(Some(Duration::from_millis(250)))
        .unwrap();
    assert!(
        read_frame(&mut probe).is_none(),
        "source must stop at max_cycles=2"
    );
    let envelope_cycles: Vec<Option<u64>> = frames.iter().map(loop_cycle_of).collect();
    assert_eq!(envelope_cycles, vec![None, None, Some(1), Some(1)]);

    for frame in &frames {
        write_frame(
            &mut stream,
            &json!({"type": "credit", "count": 1, "acks": [frame["delivery_id"]]}),
        );
    }
    write_frame(&mut stream, &json!({"type": "drain"}));
    read_until(&mut stream, "drained");
    write_frame(&mut stream, &json!({"type": "goodbye"}));

    let metrics = read_metrics_until(&harness, |value| {
        value["delivered"].as_u64() == Some(4) && value["loop_cycles"].as_u64() == Some(1)
    });
    assert_eq!(metrics["pulled"].as_u64(), Some(4));
}

#[test]
fn crash_before_ack_redelivers_after_restart() {
    let mut harness = Harness::start(
        "crash",
        &[serde_json::to_vec(&envelope("r1")).unwrap()],
        false,
    );
    {
        let mut stream = harness.connect();
        write_frame(
            &mut stream,
            &json!({"type": "hello", "protocol": 1, "credits": 1}),
        );
        read_until(&mut stream, "ready");
        let record = read_until(&mut stream, "record");
        assert_eq!(text_record(&record), "r1");
        // Drop the connection without acking.
    }
    harness.child.kill().unwrap();
    harness.child.wait().unwrap();
    std::thread::sleep(Duration::from_millis(100));

    let mut restarted = Command::new(env!("CARGO_BIN_EXE_jev-mq-bridge"))
        .arg("run")
        .arg("--config")
        .arg(harness.directory.join("bridge.json"))
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("restart bridge");
    let mut stream = connect_retry(&harness.socket);
    write_frame(
        &mut stream,
        &json!({"type": "hello", "protocol": 1, "credits": 1}),
    );
    read_until(&mut stream, "ready");
    let replay = read_until(&mut stream, "record");
    assert_eq!(
        text_record(&replay),
        "r1",
        "unacked record must be redelivered"
    );
    write_frame(
        &mut stream,
        &json!({"type": "credit", "count": 1, "acks": [replay["delivery_id"]]}),
    );
    write_frame(&mut stream, &json!({"type": "drain"}));
    read_until(&mut stream, "drained");
    write_frame(&mut stream, &json!({"type": "goodbye"}));
    let _ = restarted.kill();
    let _ = restarted.wait();
}
