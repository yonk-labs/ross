use base64::Engine;
use serde_json::{json, Value};
use std::time::Duration;

pub const DEFAULT_ASK: &str = "You are a media analyst. For the provided media return ONLY a JSON object, no markdown fence, no prose: {\"tags\": [3-10 short lowercase tags], \"description\": \"1-2 sentence factual description\", \"summary\": \"one paragraph summary\"}";

/// Per-modality endpoint selection. Each modality can point at its own server
/// and model, or fall back to the global one.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Modality {
    Vision,
    Audio,
    Video,
}

impl Modality {
    pub fn as_str(self) -> &'static str {
        match self {
            Modality::Vision => "vision",
            Modality::Audio => "audio",
            Modality::Video => "video",
        }
    }
    fn env_prefix(self) -> &'static str {
        match self {
            Modality::Vision => "ROSS_VISION",
            Modality::Audio => "ROSS_AUDIO",
            Modality::Video => "ROSS_VIDEO",
        }
    }
}

/// Global CLI overrides plus optional per-modality CLI overrides.
#[derive(Default, Clone)]
pub struct Overrides {
    pub url: Option<String>,
    pub model: Option<String>,
    pub vision_url: Option<String>,
    pub vision_model: Option<String>,
    pub audio_url: Option<String>,
    pub audio_model: Option<String>,
    pub video_url: Option<String>,
    pub video_model: Option<String>,
}

impl Overrides {
    fn for_modality(&self, m: Modality) -> (Option<String>, Option<String>) {
        match m {
            Modality::Vision => (self.vision_url.clone(), self.vision_model.clone()),
            Modality::Audio => (self.audio_url.clone(), self.audio_model.clone()),
            Modality::Video => (self.video_url.clone(), self.video_model.clone()),
        }
    }
}

fn env(k: &str) -> Option<String> {
    std::env::var(k).ok().filter(|v| !v.is_empty())
}

pub struct Endpoint {
    pub url: String,
    pub model: String,
    pub key: Option<String>,
    pub max_tokens: u64,
    pub max_inline_bytes: usize,
    client: reqwest::blocking::Client,
}

impl Endpoint {
    /// Resolve one modality. Precedence, most specific first:
    ///   --<mod>-url  >  ROSS_<MOD>_URL  >  --url  >  ROSS_URL
    /// The key follows the same shape: ROSS_<MOD>_API_KEY > ROSS_API_KEY.
    pub fn resolve(m: Modality, ov: &Overrides) -> Option<Self> {
        let p = m.env_prefix();
        let (mod_url, mod_model) = ov.for_modality(m);
        let url = mod_url
            .or_else(|| env(&format!("{p}_URL")))
            .or_else(|| ov.url.clone())
            .or_else(|| env("ROSS_URL"))?;
        let model = mod_model
            .or_else(|| env(&format!("{p}_MODEL")))
            .or_else(|| ov.model.clone())
            .or_else(|| env("ROSS_MODEL"))?;
        let key = env(&format!("{p}_API_KEY")).or_else(|| env("ROSS_API_KEY"));
        let max_tokens = env(&format!("{p}_MAX_TOKENS"))
            .or_else(|| env("ROSS_MAX_TOKENS"))
            .and_then(|v| v.parse().ok())
            // ponytail: reasoning models (Qwen3 on vLLM) spend most of the budget
            // thinking; 4096 covers it. ROSS_MAX_TOKENS overrides.
            .unwrap_or(4096);
        let timeout = env("ROSS_TIMEOUT_S").and_then(|v| v.parse().ok()).unwrap_or(120);
        let max_inline_mb: usize = env("ROSS_MAX_INLINE_MB").and_then(|v| v.parse().ok()).unwrap_or(24);
        let client = reqwest::blocking::Client::builder()
            .timeout(Duration::from_secs(timeout))
            .build()
            .ok()?;
        Some(Endpoint {
            url,
            model,
            key,
            max_tokens,
            max_inline_bytes: max_inline_mb * 1_000_000,
            client,
        })
    }

    pub fn chat_url(&self) -> String {
        format!("{}/chat/completions", self.url.trim_end_matches('/'))
    }
}

/// Endpoints for every modality, resolved once at startup.
pub struct Endpoints {
    pub vision: Option<Endpoint>,
    pub audio: Option<Endpoint>,
    pub video: Option<Endpoint>,
}

impl Endpoints {
    pub fn resolve(ov: &Overrides) -> Self {
        Endpoints {
            vision: Endpoint::resolve(Modality::Vision, ov),
            audio: Endpoint::resolve(Modality::Audio, ov),
            video: Endpoint::resolve(Modality::Video, ov),
        }
    }
    pub fn get(&self, m: Modality) -> Option<&Endpoint> {
        match m {
            Modality::Vision => self.vision.as_ref(),
            Modality::Audio => self.audio.as_ref(),
            Modality::Video => self.video.as_ref(),
        }
    }
    pub fn any(&self) -> bool {
        self.vision.is_some() || self.audio.is_some() || self.video.is_some()
    }
}

/// One inline media part: raw bytes plus the mime type they actually are.
pub struct Part<'a> {
    pub bytes: &'a [u8],
    pub mime: &'a str,
}

pub fn analyze(
    ep: &Endpoint,
    ask: &str,
    images: &[Part<'_>],
    audio: Option<(&[u8], &str)>,
    text: &str,
    vision: bool,
) -> Result<Value, String> {
    let mut parts = vec![json!({"type": "text", "text": text})];
    if vision {
        for img in images {
            if img.bytes.len() > ep.max_inline_bytes {
                return Err(format!(
                    "image is {} MB, over the {} MB inline cap (raise ROSS_MAX_INLINE_MB)",
                    img.bytes.len() / 1_000_000,
                    ep.max_inline_bytes / 1_000_000
                ));
            }
            let b64 = base64::engine::general_purpose::STANDARD.encode(img.bytes);
            // the data URL must name the real format; servers that trust it
            // will reject or misdecode a JPEG announced as PNG
            parts.push(json!({
                "type": "image_url",
                "image_url": {"url": format!("data:{};base64,{b64}", img.mime)}
            }));
        }
    }
    if let Some((bytes, fmt)) = audio {
        if bytes.len() > ep.max_inline_bytes {
            return Err(format!(
                "audio is {} MB, over the {} MB inline cap (raise ROSS_MAX_INLINE_MB)",
                bytes.len() / 1_000_000,
                ep.max_inline_bytes / 1_000_000
            ));
        }
        // ponytail: audio passthrough gated behind --audio; needs an audio-capable
        // endpoint (Qwen2-Audio/Omni). Many OpenAI-compat servers reject the part —
        // the error surfaces per-file and the batch continues.
        let b64 = base64::engine::general_purpose::STANDARD.encode(bytes);
        parts.push(json!({
            "type": "input_audio",
            "input_audio": {"data": b64, "format": fmt}
        }));
    }
    let content = if parts.len() == 1 {
        Value::String(text.to_string())
    } else {
        Value::Array(parts)
    };
    let body = json!({
        "model": ep.model,
        "max_tokens": ep.max_tokens,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": ask},
            {"role": "user", "content": content}
        ]
    });

    const ATTEMPTS: usize = 3;
    let mut last_err = String::new();
    for attempt in 0..ATTEMPTS {
        if attempt > 0 {
            // ponytail: fixed short backoff; a local vLLM that 503s is usually
            // busy for well under a second
            std::thread::sleep(Duration::from_millis(250 * attempt as u64));
        }
        let mut req = ep.client.post(ep.chat_url()).json(&body);
        if let Some(k) = &ep.key {
            req = req.bearer_auth(k);
        }
        // a transport error is the *most* retryable case, so it must not bail early
        let resp = match req.send() {
            Ok(r) => r,
            Err(e) => {
                last_err = format!("request failed: {e}");
                continue;
            }
        };
        let status = resp.status();
        // read as text first: an error response is often HTML or plain text
        // (nginx 502, a proxy timeout) and parsing it as JSON loses the status
        let raw = match resp.text() {
            Ok(t) => t,
            Err(e) => {
                last_err = format!("HTTP {status}: unreadable body: {e}");
                continue;
            }
        };
        if !status.is_success() {
            let msg = serde_json::from_str::<Value>(&raw)
                .ok()
                .and_then(|v| v["error"]["message"].as_str().map(str::to_string))
                .unwrap_or_else(|| truncate(raw.trim(), 200));
            last_err = format!("HTTP {status}: {msg}");
            if status.is_client_error() && status.as_u16() != 429 {
                break; // a 400/401/404 will not fix itself on retry
            }
            continue;
        }
        let v: Value = match serde_json::from_str(&raw) {
            Ok(v) => v,
            Err(e) => {
                last_err = format!("bad response body: {e}: {}", truncate(raw.trim(), 200));
                continue;
            }
        };
        let msg = &v["choices"][0]["message"];
        let c = match msg["content"].as_str().filter(|c| !c.trim().is_empty()) {
            Some(c) => c.to_string(),
            None => match msg["reasoning"].as_str() {
                // ponytail: reasoning models (vLLM Qwen3 etc.) put the answer after a
                // thinking pass; content can stay null when the budget runs out
                Some(r) if !r.trim().is_empty() => r.to_string(),
                _ => {
                    last_err = "response had no message content".into();
                    continue;
                }
            },
        };
        match extract_json(&c) {
            Some(obj) => return Ok(obj),
            None => last_err = format!("no JSON object in model output: {}", truncate(&c, 200)),
        }
    }
    Err(last_err)
}

/// Truncate to `n` *characters*. Byte-slicing a &str panics when the cut lands
/// inside a multi-byte character, and this only ever runs on the error path.
fn truncate(s: &str, n: usize) -> String {
    if s.chars().count() <= n {
        s.to_string()
    } else {
        format!("{}…", s.chars().take(n).collect::<String>())
    }
}

pub fn extract_json(s: &str) -> Option<Value> {
    let start = s.find('{')?;
    let bytes = s.as_bytes();
    let mut depth = 0i32;
    let mut in_str = false;
    let mut esc = false;
    for (i, &c) in bytes.iter().enumerate().skip(start) {
        if in_str {
            if esc {
                esc = false;
            } else if c == b'\\' {
                esc = true;
            } else if c == b'"' {
                in_str = false;
            }
        } else {
            match c {
                b'"' => in_str = true,
                b'{' => depth += 1,
                b'}' => {
                    depth -= 1;
                    if depth == 0 {
                        return serde_json::from_slice(&bytes[start..=i]).ok();
                    }
                }
                _ => {}
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_plain() {
        assert_eq!(
            extract_json(r#"{"tags": ["a", "b"], "description": "x"}"#).unwrap()["tags"][0],
            "a"
        );
    }

    #[test]
    fn extracts_with_prose_around() {
        let s = "Sure! Here is the analysis:\n{\"tags\": [\"cat\"], \"description\": \"a cat\", \"summary\": \"cat sits\"}\nHope that helps.";
        assert_eq!(extract_json(s).unwrap()["description"], "a cat");
    }

    #[test]
    fn handles_braces_inside_strings() {
        let s = r#"prefix {"description": "curly } brace inside", "tags": []} suffix"#;
        assert_eq!(extract_json(s).unwrap()["description"], "curly } brace inside");
    }

    #[test]
    fn handles_code_fence() {
        assert!(extract_json("```json\n{\"tags\": [\"x\"]}\n```").is_some());
    }

    #[test]
    fn rejects_garbage() {
        assert!(extract_json("no json here").is_none());
        assert!(extract_json("{unterminated").is_none());
    }

    #[test]
    fn truncate_survives_multibyte_boundary() {
        // regression: byte-slicing panicked when the cut landed inside a char
        let s = "a".repeat(199) + "é" + &"b".repeat(50);
        assert_eq!(truncate(&s, 200).chars().count(), 201); // 200 + ellipsis
        assert_eq!(truncate("héllo", 200), "héllo");
        assert_eq!(truncate("😀😀😀", 2), "😀😀…");
    }

    #[test]
    fn modality_precedence() {
        // most specific wins: per-modality CLI > per-modality env > global CLI
        let ov = Overrides {
            url: Some("http://global".into()),
            model: Some("gm".into()),
            vision_url: Some("http://vision".into()),
            ..Default::default()
        };
        let v = Endpoint::resolve(Modality::Vision, &ov).unwrap();
        assert_eq!(v.url, "http://vision");
        assert_eq!(v.model, "gm", "model falls back to global when unset");
        let a = Endpoint::resolve(Modality::Audio, &ov).unwrap();
        assert_eq!(a.url, "http://global", "audio inherits the global url");
    }

    #[test]
    fn no_endpoint_without_config() {
        let ov = Overrides { url: Some("http://x".into()), ..Default::default() };
        // url alone is not enough — a model is required too
        assert!(Endpoint::resolve(Modality::Vision, &ov).is_none() || std::env::var("ROSS_MODEL").is_ok());
    }
}
