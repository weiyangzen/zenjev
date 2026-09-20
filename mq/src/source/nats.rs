use async_nats::jetstream::consumer::{pull, AckPolicy, DeliverPolicy, PullConsumer};
use async_nats::jetstream::{self, Message};
use futures_util::StreamExt;
use std::collections::HashMap;
use std::sync::mpsc::{channel, Sender};
use std::time::Duration;
use tokio::sync::mpsc::{unbounded_channel, UnboundedSender};

use super::{Delivery, Source, SourceStatus};
use crate::canonical::sha256_hex;
use crate::config::BridgeConfig;
use crate::metrics::append_jsonl;

enum Command {
    Pull {
        max: usize,
        reply: Sender<Result<Vec<Delivery>, String>>,
    },
    Ack {
        id: String,
        reply: Sender<Result<(), String>>,
    },
    DeadLetter {
        raw: Vec<u8>,
        reason: String,
        attempts: u32,
    },
    Status {
        reply: Sender<SourceStatus>,
    },
    Shutdown,
}

/// NATS JetStream durable pull source. The async client runs on its own
/// runtime thread; the bridge itself stays synchronous.
pub struct NatsSource {
    commands: UnboundedSender<Command>,
    worker: Option<std::thread::JoinHandle<()>>,
    quarantine_path: std::path::PathBuf,
}

impl NatsSource {
    pub fn open(config: &BridgeConfig) -> Result<Self, String> {
        let (commands, receiver) = unbounded_channel();
        let (ready_tx, ready_rx) = channel::<Result<(), String>>();
        let worker_config = config.clone();
        let worker = std::thread::Builder::new()
            .name("jev-mq-nats".to_string())
            .spawn(move || {
                let runtime = match tokio::runtime::Builder::new_multi_thread()
                    .worker_threads(2)
                    .enable_all()
                    .build()
                {
                    Ok(runtime) => runtime,
                    Err(error) => {
                        let _ = ready_tx.send(Err(format!("cannot build tokio runtime: {error}")));
                        return;
                    }
                };
                runtime.block_on(worker(worker_config, receiver, ready_tx));
            })
            .map_err(|error| format!("cannot spawn NATS worker: {error}"))?;
        match ready_rx.recv_timeout(Duration::from_secs(30)) {
            Ok(Ok(())) => Ok(Self {
                commands,
                worker: Some(worker),
                quarantine_path: config.quarantine_path.clone(),
            }),
            Ok(Err(error)) => Err(error),
            Err(error) => Err(format!("NATS bridge did not become ready: {error}")),
        }
    }

    fn request<T>(
        &self,
        build: impl FnOnce(Sender<Result<T, String>>) -> Command,
    ) -> Result<T, String> {
        let (reply, receiver) = channel();
        self.commands
            .send(build(reply))
            .map_err(|_| "NATS worker is not running".to_string())?;
        receiver
            .recv_timeout(Duration::from_secs(120))
            .map_err(|error| format!("NATS worker reply timed out: {error}"))?
    }
}

impl Source for NatsSource {
    fn adapter(&self) -> &'static str {
        "nats-jetstream"
    }

    fn next(&mut self, max: usize) -> Result<Vec<Delivery>, String> {
        self.request(|reply| Command::Pull { max, reply })
    }

    fn ack(&mut self, delivery: &Delivery) -> Result<(), String> {
        let id = delivery.id.clone();
        self.request(|reply| Command::Ack { id, reply })
    }

    fn dead_letter(&mut self, raw: &[u8], reason: &str, attempts: u32) -> Result<(), String> {
        self.commands
            .send(Command::DeadLetter {
                raw: raw.to_vec(),
                reason: reason.to_string(),
                attempts,
            })
            .map_err(|_| "NATS worker is not running".to_string())
    }

    fn quarantine(&mut self, raw: &[u8], reason: &str, attempts: u32) -> Result<(), String> {
        append_jsonl(
            &self.quarantine_path,
            &serde_json::json!({
                "class": "quarantine",
                "reason": reason,
                "attempts": attempts,
                "raw_sha256": sha256_hex(raw),
                "raw_bytes": raw.len(),
            }),
        );
        let _ = (reason, attempts);
        Ok(())
    }

    fn status(&self) -> SourceStatus {
        let (reply, receiver) = channel();
        if self.commands.send(Command::Status { reply }).is_err() {
            return SourceStatus::default();
        }
        receiver
            .recv_timeout(Duration::from_secs(5))
            .unwrap_or_default()
    }

    fn shutdown(&mut self) {
        let _ = self.commands.send(Command::Shutdown);
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

async fn worker(
    config: BridgeConfig,
    mut commands: tokio::sync::mpsc::UnboundedReceiver<Command>,
    ready: Sender<Result<(), String>>,
) {
    let mut options = async_nats::ConnectOptions::new().require_tls(config.tls_required);
    if let Some(ca) = &config.ca_cert {
        options = options.add_root_certificates(ca.clone());
    }
    if let (Some(cert), Some(key)) = (&config.client_cert, &config.client_key) {
        options = options.add_client_certificate(cert.clone(), key.clone());
    }
    let client = match options.connect(config.endpoints.clone()).await {
        Ok(client) => client,
        Err(error) => {
            let _ = ready.send(Err(format!("NATS connect failed: {error}")));
            return;
        }
    };
    let context = jetstream::new(client.clone());
    let stream = match context.get_stream(&config.stream).await {
        Ok(stream) => stream,
        Err(error) => {
            let _ = ready.send(Err(format!(
                "NATS stream {} is unavailable: {error}",
                config.stream
            )));
            return;
        }
    };
    let deliver_policy = match config.start_position.as_str() {
        "first" => DeliverPolicy::All,
        "last" => DeliverPolicy::Last,
        "by_offset" => DeliverPolicy::ByStartSequence {
            start_sequence: config.start_offset.unwrap_or(1),
        },
        "by_timestamp" => DeliverPolicy::ByStartTime {
            start_time: async_nats::datetime::DateTime::from_unix_timestamp(
                config.start_timestamp.unwrap_or(0),
            )
            .unwrap_or(async_nats::datetime::DateTime::UNIX_EPOCH),
        },
        _ => DeliverPolicy::New,
    };
    let consumer_config = pull::Config {
        durable_name: Some(config.durable_name.clone()),
        ack_policy: AckPolicy::Explicit,
        deliver_policy,
        max_ack_pending: config.max_ack_pending as i64,
        ack_wait: Duration::from_secs(config.ack_wait_seconds),
        ..Default::default()
    };
    let mut consumer = match stream
        .get_or_create_consumer(&config.durable_name, consumer_config)
        .await
    {
        Ok(consumer) => consumer,
        Err(error) => {
            let _ = ready.send(Err(format!("NATS consumer creation failed: {error}")));
            return;
        }
    };
    let _ = ready.send(Ok(()));

    let mut pending: HashMap<String, Message> = HashMap::new();
    while let Some(command) = commands.recv().await {
        match command {
            Command::Pull { max, reply } => {
                let result = pull_batch(&consumer, &mut pending, max).await;
                let _ = reply.send(result);
            }
            Command::Ack { id, reply } => {
                let result = match pending.remove(&id) {
                    Some(message) => match message.ack().await {
                        Ok(()) => client.flush().await.map_err(|error| error.to_string()),
                        Err(error) => Err(error.to_string()),
                    },
                    None => Err(format!("unknown delivery {id}")),
                };
                let _ = reply.send(result);
            }
            Command::DeadLetter {
                raw,
                reason,
                attempts,
            } => {
                let mut headers = async_nats::HeaderMap::new();
                headers.insert("Jev-Dlq-Reason", reason.as_str());
                headers.insert("Jev-Dlq-Attempts", attempts.to_string().as_str());
                let _ = client
                    .publish_with_headers(config.dlq_subject.clone(), headers, raw.into())
                    .await;
                let _ = client.flush().await;
            }
            Command::Status { reply } => {
                let info = consumer.info().await;
                let status = match info {
                    Ok(info) => SourceStatus {
                        lag: info.num_pending as i64,
                        checkpoint: info.delivered.stream_sequence.into(),
                    },
                    Err(_) => SourceStatus::default(),
                };
                let _ = reply.send(status);
            }
            Command::Shutdown => break,
        }
    }
}

async fn pull_batch(
    consumer: &PullConsumer,
    pending: &mut HashMap<String, Message>,
    max: usize,
) -> Result<Vec<Delivery>, String> {
    if max == 0 {
        return Ok(Vec::new());
    }
    let mut stream = consumer
        .fetch()
        .max_messages(max)
        .expires(Duration::from_millis(800))
        .messages()
        .await
        .map_err(|error| error.to_string())?;
    let mut batch = Vec::new();
    while let Some(message) = stream.next().await {
        let message = message.map_err(|error| error.to_string())?;
        let (stream_sequence, delivered) = {
            let info = message.info().map_err(|error| error.to_string())?;
            (info.stream_sequence, info.delivered)
        };
        let id = format!("nats-{stream_sequence}");
        let attempts = (delivered.max(0) + 1) as u32;
        let raw = message.payload.to_vec();
        pending.insert(id.clone(), message);
        batch.push(Delivery {
            id,
            offset: stream_sequence,
            attempts,
            raw,
        });
    }
    Ok(batch)
}
