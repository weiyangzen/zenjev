use serde_json::{Map, Value};
use sha2::{Digest, Sha256};

/// Canonical JSON: object keys sorted, no insignificant whitespace, UTF-8.
/// Python's `json.dumps(sort_keys=True, separators=(",", ":"), ensure_ascii=False)`
/// produces the same bytes for the shared contract.
pub fn canonical_json(value: &Value, out: &mut String) {
    match value {
        Value::Null => out.push_str("null"),
        Value::Bool(flag) => out.push_str(if *flag { "true" } else { "false" }),
        Value::Number(number) => out.push_str(&number.to_string()),
        Value::String(text) => {
            out.push_str(&Value::String(text.clone()).to_string());
        }
        Value::Array(items) => {
            out.push('[');
            for (index, item) in items.iter().enumerate() {
                if index > 0 {
                    out.push(',');
                }
                canonical_json(item, out);
            }
            out.push(']');
        }
        Value::Object(map) => {
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort();
            out.push('{');
            for (index, key) in keys.iter().enumerate() {
                if index > 0 {
                    out.push(',');
                }
                out.push_str(&Value::String((*key).clone()).to_string());
                out.push(':');
                canonical_json(&map[*key], out);
            }
            out.push('}');
        }
    }
}

pub fn canonical_string(value: &Value) -> String {
    let mut out = String::new();
    canonical_json(value, &mut out);
    out
}

/// Hash of the canonical envelope body with the `content_sha256` field removed.
pub fn envelope_content_hash(value: &Value) -> Result<String, String> {
    let map = value
        .as_object()
        .ok_or_else(|| "envelope must be a JSON object".to_string())?;
    let mut body: Map<String, Value> = map.clone();
    body.remove("content_sha256");
    Ok(sha256_hex(
        canonical_string(&Value::Object(body)).as_bytes(),
    ))
}

pub fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    hex::encode(hasher.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn canonical_sorts_keys_and_is_stable() {
        let value = json!({"b": 1, "a": {"d": [2, 1], "c": "x"}});
        assert_eq!(
            canonical_string(&value),
            r#"{"a":{"c":"x","d":[2,1]},"b":1}"#
        );
    }

    #[test]
    fn envelope_hash_ignores_existing_hash_field() {
        let value = json!({"record_id": "r1", "content_sha256": "aa", "kind": "raw_document"});
        let expected = sha256_hex(br#"{"kind":"raw_document","record_id":"r1"}"#);
        assert_eq!(envelope_content_hash(&value).unwrap(), expected);
    }
}
