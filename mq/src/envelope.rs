use serde::Deserialize;
use serde_json::Value;

use crate::canonical::envelope_content_hash;
use crate::config::{is_sha256, BridgeConfig};

pub const ENVELOPE_VERSION: u32 = 1;
pub const KINDS: [&str; 3] = ["labelled_example", "tool_task_pair", "raw_document"];

#[derive(Debug, Clone, Deserialize)]
pub struct SchemaRef {
    pub id: String,
    pub version: u32,
    pub digest: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct SourceRef {
    pub uri: String,
    pub content_sha256: String,
    pub retrieved_at: String,
    #[serde(default)]
    pub license: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ObservedModel {
    #[serde(default)]
    pub provider: String,
    #[serde(alias = "model_id", alias = "model")]
    pub model_id: String,
}

/// A validated envelope. The raw JSON value is retained for provenance and
/// possible rewrite; only validated fields are exposed.
#[derive(Debug, Clone)]
pub struct Envelope {
    pub value: Value,
    pub envelope_version: u32,
    pub record_id: String,
    pub content_sha256: String,
    pub kind: String,
    pub schema: SchemaRef,
    pub source: SourceRef,
    pub observed_model: Option<ObservedModel>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Rejection {
    /// Malformed or oversized input. Goes to the DLQ and is acked so poison
    /// input never blocks the consumer.
    DeadLetter(String),
    /// Schema lineage mismatch. Goes to quarantine and is never trained.
    Quarantine(String),
}

impl Rejection {
    pub fn class(&self) -> &'static str {
        match self {
            Rejection::DeadLetter(_) => "dlq",
            Rejection::Quarantine(_) => "quarantine",
        }
    }

    pub fn reason(&self) -> &str {
        match self {
            Rejection::DeadLetter(reason) | Rejection::Quarantine(reason) => reason,
        }
    }
}

/// Validate one raw message against the frozen §1.5 envelope contract.
pub fn validate(raw: &[u8], config: &BridgeConfig) -> Result<Envelope, Rejection> {
    if raw.len() > config.max_bytes {
        return Err(Rejection::DeadLetter("oversized_message".to_string()));
    }
    let text = std::str::from_utf8(raw)
        .map_err(|_| Rejection::DeadLetter("envelope_not_utf8".to_string()))?;
    let value: Value = serde_json::from_str(text)
        .map_err(|_| Rejection::DeadLetter("envelope_not_json".to_string()))?;
    let object = value
        .as_object()
        .ok_or_else(|| Rejection::DeadLetter("envelope_not_object".to_string()))?;

    let version = object
        .get("envelope_version")
        .and_then(Value::as_u64)
        .ok_or_else(|| Rejection::DeadLetter("missing_envelope_version".to_string()))?;
    if version != u64::from(ENVELOPE_VERSION) {
        return Err(Rejection::DeadLetter(
            "unsupported_envelope_version".to_string(),
        ));
    }
    let record_id = required_string(object, "record_id")?;
    let content_sha256 = required_string(object, "content_sha256")?;
    if !is_sha256(&content_sha256) {
        return Err(Rejection::DeadLetter("invalid_content_sha256".to_string()));
    }
    let computed = envelope_content_hash(&value).map_err(Rejection::DeadLetter)?;
    if computed != content_sha256 {
        return Err(Rejection::DeadLetter("content_hash_mismatch".to_string()));
    }

    let schema: SchemaRef = serde_json::from_value(
        object
            .get("schema")
            .cloned()
            .ok_or_else(|| Rejection::DeadLetter("missing_schema".to_string()))?,
    )
    .map_err(|_| Rejection::DeadLetter("invalid_schema_block".to_string()))?;
    if schema.id != config.schema_id
        || schema.version != config.schema_version
        || schema.digest != config.schema_digest
    {
        return Err(Rejection::Quarantine("schema_digest_mismatch".to_string()));
    }
    if !is_sha256(&schema.digest) {
        return Err(Rejection::Quarantine(
            "schema_digest_not_sha256".to_string(),
        ));
    }

    let kind = required_string(object, "kind")?;
    if !KINDS.contains(&kind.as_str()) {
        return Err(Rejection::DeadLetter("unknown_kind".to_string()));
    }

    let source: SourceRef = serde_json::from_value(
        object
            .get("source")
            .cloned()
            .ok_or_else(|| Rejection::DeadLetter("missing_source".to_string()))?,
    )
    .map_err(|_| Rejection::DeadLetter("invalid_source_block".to_string()))?;
    if source.uri.is_empty() || source.retrieved_at.is_empty() {
        return Err(Rejection::DeadLetter(
            "source_requires_uri_and_retrieved_at".to_string(),
        ));
    }
    if !is_sha256(&source.content_sha256) {
        return Err(Rejection::DeadLetter(
            "invalid_source_content_sha256".to_string(),
        ));
    }

    match kind.as_str() {
        "labelled_example" => {
            if object.get("labels").is_none() && object.get("example").is_none() {
                return Err(Rejection::DeadLetter(
                    "labelled_example_requires_labels".to_string(),
                ));
            }
        }
        "tool_task_pair" => {
            for section in ["request", "response"] {
                let section_value = object.get(section).ok_or_else(|| {
                    Rejection::DeadLetter(format!("tool_task_pair_requires_{section}"))
                })?;
                let text = section_value
                    .get("text")
                    .and_then(Value::as_str)
                    .ok_or_else(|| Rejection::DeadLetter(format!("{section}_requires_text")))?;
                if text.is_empty() {
                    return Err(Rejection::DeadLetter(format!("{section}_text_is_empty")));
                }
            }
        }
        "raw_document" => {
            if object.get("document").is_none() && object.get("example").is_none() {
                return Err(Rejection::DeadLetter(
                    "raw_document_requires_document".to_string(),
                ));
            }
        }
        _ => unreachable!(),
    }

    let observed_model = match object.get("observed_model") {
        None | Some(Value::Null) => None,
        Some(value) => {
            let model: ObservedModel = serde_json::from_value(value.clone())
                .map_err(|_| Rejection::DeadLetter("invalid_observed_model".to_string()))?;
            if model.model_id.is_empty() {
                return Err(Rejection::DeadLetter(
                    "observed_model_requires_model_id".to_string(),
                ));
            }
            if !config.allowed_models.is_empty() && !config.allowed_models.contains(&model.model_id)
            {
                return Err(Rejection::DeadLetter(
                    "observed_model_not_allowlisted".to_string(),
                ));
            }
            Some(model)
        }
    };

    Ok(Envelope {
        value,
        envelope_version: version as u32,
        record_id,
        content_sha256,
        kind,
        schema,
        source,
        observed_model,
    })
}

impl Envelope {
    pub fn idempotency_key(&self) -> String {
        format!("{}|{}", self.record_id, self.content_sha256)
    }

    pub fn observed_model_id(&self) -> Option<&str> {
        self.observed_model
            .as_ref()
            .map(|model| model.model_id.as_str())
    }

    pub fn observed_model_provider(&self) -> Option<&str> {
        self.observed_model
            .as_ref()
            .map(|model| model.provider.as_str())
    }
}

fn required_string(
    object: &serde_json::Map<String, Value>,
    key: &str,
) -> Result<String, Rejection> {
    let value = object
        .get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| Rejection::DeadLetter(format!("missing_{key}")))?
        .to_string();
    if value.is_empty() {
        return Err(Rejection::DeadLetter(format!("empty_{key}")));
    }
    Ok(value)
}
