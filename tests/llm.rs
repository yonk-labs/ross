//! The endpoint path is the most logic-dense part of ross and the easiest to
//! regress silently: retries, status handling, the exact JSON shape sent, and
//! which modality reaches which server. Exercised here against a mock server
//! built on std::net so the suite gains no dependency and no network access.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpListener;
use std::sync::mpsc;

/// Canned replies, consumed in order; the last one repeats.
fn mock(replies: Vec<(u16, String)>) -> (String, mpsc::Receiver<String>) {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
    let port = listener.local_addr().unwrap().port();
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        for (i, stream) in listener.incoming().enumerate() {
            let mut s = match stream {
                Ok(s) => s,
                Err(_) => break,
            };
            let mut r = BufReader::new(s.try_clone().unwrap());
            let mut len = 0usize;
            loop {
                let mut line = String::new();
                if r.read_line(&mut line).unwrap_or(0) == 0 || line == "\r\n" {
                    break;
                }
                if let Some(v) = line.to_ascii_lowercase().strip_prefix("content-length:") {
                    len = v.trim().parse().unwrap_or(0);
                }
            }
            let mut body = vec![0u8; len];
            let _ = r.read_exact(&mut body);
            let _ = tx.send(String::from_utf8_lossy(&body).into_owned());
            let (code, payload) = &replies[i.min(replies.len() - 1)];
            let _ = write!(
                s,
                "HTTP/1.1 {code} X\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{payload}",
                payload.len()
            );
            let _ = s.flush();
        }
    });
    (format!("http://127.0.0.1:{port}/v1"), rx)
}

fn ok_body(tag: &str) -> String {
    format!(
        r#"{{"choices":[{{"message":{{"content":"{{\"tags\":[\"{tag}\"],\"description\":\"d\",\"summary\":\"s\"}}"}}}}]}}"#
    )
}

fn ep(url: &str) -> ross::semantic::Endpoint {
    let ov = ross::semantic::Overrides {
        url: Some(url.to_string()),
        model: Some("m".into()),
        ..Default::default()
    };
    ross::semantic::Endpoint::resolve(ross::semantic::Modality::Vision, &ov).expect("endpoint")
}

#[test]
fn parses_a_successful_response() {
    let (url, rx) = mock(vec![(200, ok_body("cat"))]);
    let v = ross::semantic::analyze(&ep(&url), "ask", &[], None, "text", true).expect("ok");
    assert_eq!(v["tags"][0], "cat");
    let sent: serde_json::Value = serde_json::from_str(&rx.recv().unwrap()).unwrap();
    assert_eq!(sent["model"], "m");
    assert_eq!(sent["messages"][0]["content"], "ask");
    assert_eq!(sent["messages"][1]["content"], "text", "text-only stays a plain string");
}

#[test]
fn inlines_images_with_their_real_mime_type() {
    let (url, rx) = mock(vec![(200, ok_body("x"))]);
    let png = [0x89u8, b'P', b'N', b'G'];
    let jpg = [0xFFu8, 0xD8, 0xFF];
    let parts = [
        ross::semantic::Part { bytes: &png, mime: "image/png" },
        ross::semantic::Part { bytes: &jpg, mime: "image/jpeg" },
    ];
    ross::semantic::analyze(&ep(&url), "ask", &parts, None, "t", true).expect("ok");
    let sent = rx.recv().unwrap();
    // a JPEG announced as PNG is the bug this guards
    assert!(sent.contains("data:image/png;base64,"), "{sent}");
    assert!(sent.contains("data:image/jpeg;base64,"), "{sent}");
}

#[test]
fn no_vision_sends_text_only() {
    let (url, rx) = mock(vec![(200, ok_body("x"))]);
    let png = [0x89u8, b'P', b'N', b'G'];
    let parts = [ross::semantic::Part { bytes: &png, mime: "image/png" }];
    ross::semantic::analyze(&ep(&url), "ask", &parts, None, "t", false).expect("ok");
    let sent = rx.recv().unwrap();
    assert!(!sent.contains("image_url"), "vision disabled must not inline images: {sent}");
}

#[test]
fn attaches_audio_when_given() {
    let (url, rx) = mock(vec![(200, ok_body("x"))]);
    ross::semantic::analyze(&ep(&url), "ask", &[], Some((&[1, 2, 3], "mp3")), "t", true)
        .expect("ok");
    let sent: serde_json::Value = serde_json::from_str(&rx.recv().unwrap()).unwrap();
    let parts = sent["messages"][1]["content"].as_array().expect("array");
    let audio = parts.iter().find(|p| p["type"] == "input_audio").expect("audio part");
    assert_eq!(audio["input_audio"]["format"], "mp3");
}

#[test]
fn retries_a_server_error_then_succeeds() {
    let (url, rx) = mock(vec![
        (503, "{\"error\":{\"message\":\"busy\"}}".into()),
        (200, ok_body("recovered")),
    ]);
    let v = ross::semantic::analyze(&ep(&url), "ask", &[], None, "t", true).expect("should retry");
    assert_eq!(v["tags"][0], "recovered");
    assert_eq!(rx.iter().take(2).count(), 2, "expected exactly two attempts");
}

#[test]
fn does_not_retry_a_client_error() {
    let (url, rx) = mock(vec![(400, "{\"error\":{\"message\":\"bad model\"}}".into())]);
    let e = ross::semantic::analyze(&ep(&url), "ask", &[], None, "t", true).unwrap_err();
    assert!(e.contains("400") && e.contains("bad model"), "{e}");
    drop(rx);
}

#[test]
fn reports_a_non_json_error_body_without_losing_the_status() {
    let (url, _rx) = mock(vec![(502, "<html>bad gateway</html>".into())]);
    let e = ross::semantic::analyze(&ep(&url), "ask", &[], None, "t", true).unwrap_err();
    assert!(e.contains("502"), "status must survive an HTML body: {e}");
}

/// Regression: truncating the model's reply for the error message used to
/// byte-slice a &str and panic when the cut landed inside a character.
#[test]
fn multibyte_prose_reply_errors_instead_of_panicking() {
    let junk = "a".repeat(199) + "é" + &"b".repeat(80);
    let body = format!(r#"{{"choices":[{{"message":{{"content":"{junk}"}}}}]}}"#);
    let (url, _rx) = mock(vec![(200, body)]);
    let e = ross::semantic::analyze(&ep(&url), "ask", &[], None, "t", true).unwrap_err();
    assert!(e.contains("no JSON object"), "{e}");
}

/// Reasoning models leave `content` empty and put the answer in `reasoning`.
#[test]
fn falls_back_to_the_reasoning_field() {
    let body = r#"{"choices":[{"message":{"content":null,"reasoning":"thinking... {\"tags\":[\"r\"]}"}}]}"#;
    let (url, _rx) = mock(vec![(200, body.into())]);
    let v = ross::semantic::analyze(&ep(&url), "ask", &[], None, "t", true).expect("ok");
    assert_eq!(v["tags"][0], "r");
}

#[test]
fn refuses_to_inline_media_over_the_cap() {
    let (url, _rx) = mock(vec![(200, ok_body("x"))]);
    // set the cap on the endpoint rather than through the environment: tests in
    // one binary share a process, and Endpoint::resolve reads that variable, so
    // set_var here would randomly cap unrelated tests
    let mut endpoint = ep(&url);
    endpoint.max_inline_bytes = 1_000_000;
    let big = vec![0u8; 2_000_000];
    let parts = [ross::semantic::Part { bytes: &big, mime: "image/png" }];
    let e = ross::semantic::analyze(&endpoint, "ask", &parts, None, "t", true).unwrap_err();
    assert!(e.contains("inline cap"), "{e}");
}

/// Embedded catalogue tags crowd out attached media: several different sound
/// effects from one asset pack all came back described as the pack itself, so
/// they are kept out of the prompt when the media is inlined. Verified here at
/// the wire level — whatever the caller passes as `text` is what gets sent, so
/// this pins the contract the pipeline relies on.
#[test]
fn the_text_block_is_sent_verbatim_alongside_media() {
    let (url, rx) = mock(vec![(200, ok_body("x"))]);
    let png = [0x89u8, b'P', b'N', b'G'];
    let parts = [ross::semantic::Part { bytes: &png, mime: "image/png" }];
    let lean = "Image file.\nMetadata:\n{\n  \"width\": 4\n}";
    ross::semantic::analyze(&ep(&url), "ask", &parts, None, lean, true).expect("ok");
    let sent: serde_json::Value = serde_json::from_str(&rx.recv().unwrap()).unwrap();
    let content = sent["messages"][1]["content"].as_array().expect("array");
    let text = content[0]["text"].as_str().unwrap();
    assert_eq!(text, lean, "the prompt text must reach the model unmodified");
    assert!(!text.contains("format_tags"));
}
