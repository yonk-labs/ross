use serde_json::Value;
use std::process::Command;

/// A skipped test that silently passes is a test that proves nothing. In CI,
/// set ROSS_TEST_STRICT=1 so a missing dependency fails loudly instead.
fn skip(why: &str) {
    if std::env::var("ROSS_TEST_STRICT").is_ok() {
        panic!("ROSS_TEST_STRICT set but cannot run: {why}");
    }
    eprintln!("skipping: {why}");
}

fn have(binary: &str, arg: &str) -> bool {
    Command::new(binary)
        .arg(arg)
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

fn tmp(name: &str) -> std::path::PathBuf {
    let d = std::env::temp_dir().join(format!("ross-it-{}", std::process::id()));
    let _ = std::fs::create_dir_all(&d);
    d.join(name)
}

fn ffmpeg(args: &[&str]) {
    let s = Command::new("ffmpeg")
        .args(["-hide_banner", "-loglevel", "error"])
        .args(args)
        .status()
        .expect("ffmpeg");
    assert!(s.success(), "ffmpeg {args:?} failed");
}

fn ross(args: &[&str]) -> (String, String, i32) {
    let out = Command::new(env!("CARGO_BIN_EXE_ross"))
        .args(args)
        .output()
        .expect("ross binary");
    (
        String::from_utf8_lossy(&out.stdout).into_owned(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
        out.status.code().unwrap_or(-1),
    )
}

#[test]
fn deterministic_pass_end_to_end() {
    if !have("ffmpeg", "-version") || !have("ffprobe", "-version") {
        skip("ffmpeg/ffprobe not available");
        return;
    }
    let wav = tmp("tone.wav");
    let png = tmp("img.png");
    ffmpeg(&[
        "-f", "lavfi", "-i", "sine=frequency=440:duration=2", "-y",
        wav.to_str().unwrap(),
    ]);
    ffmpeg(&[
        "-f", "lavfi", "-i", "testsrc=duration=1:size=64x64:rate=5",
        "-frames:v", "1", "-update", "1", "-y", png.to_str().unwrap(),
    ]);

    let (stdout, _stderr, code) = ross(&[
        "--no-llm", "--json", "--quiet", wav.to_str().unwrap(), png.to_str().unwrap(),
    ]);
    assert_eq!(code, 0, "exit code: {stdout}");
    let v: Value = serde_json::from_str(stdout.trim()).unwrap();
    let arr: Vec<Value> = v.as_array().cloned().unwrap_or_else(|| vec![v]);
    assert_eq!(arr.len(), 2);
    let tone = arr.iter().find(|x| x["path"].as_str().unwrap().ends_with("tone.wav")).unwrap();
    assert_eq!(tone["type"], "audio");
    assert_eq!(tone["sha256"].as_str().unwrap().len(), 64);
    assert!((tone["metadata"]["duration_s"].as_f64().unwrap() - 2.0).abs() < 0.1);
    let img = arr.iter().find(|x| x["path"].as_str().unwrap().ends_with("img.png")).unwrap();
    assert_eq!(img["type"], "image");
    assert_eq!(img["metadata"]["width"], 64);

    let (stdout, _, _) = ross(&["--no-llm", "--text", "--quiet", wav.to_str().unwrap()]);
    assert!(stdout.contains("== "));
    assert!(stdout.contains("type: audio"));

    let (_, stderr, code) = ross(&["--no-llm", "--json", "/definitely/not/here"]);
    assert_eq!(code, 4);
    assert!(stderr.contains("ross:"));

    let (_, stderr, code) = ross(&[wav.to_str().unwrap()]);
    assert_eq!(code, 3, "no endpoint configured should exit 3: {stderr}");

    let (stdout, _, _) = ross(&["--no-llm", "--md", "--quiet", wav.to_str().unwrap()]);
    assert!(stdout.contains("## "), "md output missing headings");

    // --ask-file must be read, and a missing one is a usage error
    let askf = tmp("ask.txt");
    std::fs::write(&askf, "describe it").unwrap();
    let (_, _, code) = ross(&["--no-llm", "--quiet", "--ask-file", askf.to_str().unwrap(), wav.to_str().unwrap()]);
    assert_eq!(code, 0);
    let (_, _, code) = ross(&["--no-llm", "--quiet", "--ask-file", "/no/such/file", wav.to_str().unwrap()]);
    assert_eq!(code, 4);
}

/// --strict must still print everything it successfully produced. Losing a whole
/// batch because the last file was corrupt is worse than the error it reports.
#[test]
fn strict_reports_failure_without_discarding_results() {
    if !have("ffmpeg", "-version") || !have("ffprobe", "-version") {
        skip("ffmpeg/ffprobe not available");
        return;
    }
    let dir = tmp("strictdir");
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("a_broken.mp4"), b"\x00\x00\x00\x20ftypisom").unwrap();
    ffmpeg(&["-f", "lavfi", "-i", "sine=frequency=440:duration=1", "-y",
             dir.join("z_good.wav").to_str().unwrap()]);

    let (stdout, stderr, code) = ross(&["--no-llm", "--json", "--quiet", "--strict", dir.to_str().unwrap()]);
    assert_eq!(code, 2, "a file error under --strict must exit 2");
    assert!(stderr.contains("--strict"), "stderr should name the failing file");
    let v: Value = serde_json::from_str(stdout.trim()).expect("results must still be on stdout");
    let arr = v.as_array().expect("array");
    assert_eq!(arr.len(), 2, "both the good and the failed file must be reported");
    assert!(arr.iter().any(|e| e["sha256"].is_string()), "the good file's work survived");

    let (stdout, _, code) = ross(&["--no-llm", "--json", "--quiet", dir.to_str().unwrap()]);
    assert_eq!(code, 0, "without --strict a file error must not fail the batch");
    assert!(stdout.contains("error"));
}

/// Per-modality endpoints: each modality resolves independently, falling back
/// to the global setting when it has none of its own.
#[test]
fn doctor_reports_per_modality_endpoints() {
    let (out, _, _) = ross(&["--doctor", "--url", "http://global", "--model", "gm"]);
    for m in ["vision", "audio", "video"] {
        assert!(out.lines().any(|l| l.starts_with(m) && l.contains("http://global")),
                "{m} should inherit the global endpoint:\n{out}");
    }
    let (out, _, _) = ross(&["--doctor", "--url", "http://global", "--model", "gm",
                             "--vision-url", "http://vision", "--vision-model", "vm"]);
    assert!(out.lines().any(|l| l.starts_with("vision") && l.contains("http://vision")));
    assert!(out.lines().any(|l| l.starts_with("audio") && l.contains("http://global")));

    let (out, _, _) = ross(&["--doctor"]);
    assert!(out.contains("NOT SET") || out.contains("model="), "doctor must report endpoint state");
}

/// CLAP is the description path when no LLM is configured, so the contract is:
/// a sound it genuinely recognizes gets tags; anything it does not clears out to
/// a filename-derived description rather than confident nonsense.
#[test]
fn clap_native_pass_end_to_end() {
    if !have("ffmpeg", "-version") {
        skip("ffmpeg not available");
        return;
    }
    // --doctor is the single source of truth for whether the model is cached
    let (doc, _, _) = ross(&["--doctor"]);
    if !doc.lines().any(|l| l.starts_with("clap") && l.contains("ok")) {
        skip("CLAP model not downloaded yet");
        return;
    }
    // a broadband decaying burst is squarely in the vocabulary (impact/explosion)
    let boom = tmp("clap_boom.wav");
    ffmpeg(&["-f", "lavfi", "-i", "anoisesrc=d=5:c=brown:a=0.9",
             "-af", "afade=t=out:st=0:d=1.2", "-ar", "48000", "-y", boom.to_str().unwrap()]);
    // a pure tone matches nothing in the vocabulary
    let tone = tmp("clap_tone.wav");
    ffmpeg(&["-f", "lavfi", "-i", "sine=frequency=440:duration=3", "-y", tone.to_str().unwrap()]);
    // too brief to yield a usable spectrogram at all
    let short = tmp("clap_short.wav");
    ffmpeg(&["-f", "lavfi", "-i", "sine=frequency=440:duration=0.2", "-y", short.to_str().unwrap()]);

    let (stdout, _stderr, code) = ross(&[
        "--clap", "--no-llm", "--json", "--quiet",
        boom.to_str().unwrap(), tone.to_str().unwrap(), short.to_str().unwrap(),
    ]);
    assert_eq!(code, 0);
    let v: Value = serde_json::from_str(stdout.trim()).unwrap();
    let arr: Vec<Value> = v.as_array().cloned().unwrap_or_else(|| vec![v]);
    let get = |n: &str| arr.iter().find(|x| x["path"].as_str().unwrap().contains(n)).unwrap();

    let b = get("clap_boom");
    let tags = b["tags"].as_array().expect("a recognizable sound must get tags");
    assert!(!tags.is_empty() && tags.len() <= 6, "got {tags:?}");
    assert_eq!(b["model"], "clap:larger_clap_general");
    assert!(b["description"].as_str().unwrap().len() > 3);

    // every file gets a description with no LLM configured — that is the point
    for name in ["clap_boom", "clap_tone", "clap_short"] {
        assert!(get(name)["description"].as_str().is_some_and(|d| !d.is_empty()),
                "{name} must have a non-LLM description");
    }
    // whatever a file is tagged, the confidence that earned it must be reported
    for r in arr.iter().filter(|r| r["tags"].is_array()) {
        let c = r["tag_confidence"].as_f64().expect("tagged files carry a confidence");
        assert!((0.2..=1.0).contains(&c), "implausible confidence {c}");
    }
    assert!(get("clap_short")["description"].as_str().unwrap().contains("short sound effect"));
}

/// The image path mirrors the audio one: local tags with no network, and a
/// description for every file whether or not the model was confident.
#[test]
fn clip_image_tagging_and_custom_labels() {
    if !have("ffmpeg", "-version") {
        skip("ffmpeg not available");
        return;
    }
    let (doc, _, _) = ross(&["--doctor"]);
    if !doc.lines().any(|l| l.starts_with("clip") && l.contains("ok")) {
        skip("CLIP model not downloaded yet");
        return;
    }
    let png = tmp("clip_src.png");
    ffmpeg(&["-f", "lavfi", "-i", "testsrc=duration=1:size=256x256:rate=1",
             "-frames:v", "1", "-update", "1", "-y", png.to_str().unwrap()]);

    let (stdout, _, code) = ross(&["--clip", "--no-llm", "--json", "--quiet", png.to_str().unwrap()]);
    assert_eq!(code, 0);
    let v: Value = serde_json::from_str(stdout.trim()).unwrap();
    assert_eq!(v["type"], "image");
    assert!(v["description"].as_str().is_some_and(|d| !d.is_empty()),
            "every image gets a non-LLM description");
    if v["tags"].is_array() {
        assert!(v["tag_confidence"].as_f64().is_some(), "tags must carry a confidence");
    }

    // a custom vocabulary must actually replace the built-in one
    let (stdout, _, code) = ross(&["--clip", "--no-llm", "--json", "--quiet",
                                   "--labels", "a test pattern,a photograph of a dog",
                                   png.to_str().unwrap()]);
    assert_eq!(code, 0, "custom labels should work: {stdout}");
    let v: Value = serde_json::from_str(stdout.trim()).unwrap();
    if let Some(tags) = v["tags"].as_array() {
        for t in tags {
            let t = t.as_str().unwrap();
            assert!(t == "a test pattern" || t == "a photograph of a dog",
                    "tag {t:?} is not from the custom vocabulary");
        }
    }
}

/// Label-list handling is a usage concern: it must be validated before we go
/// near the filesystem, so a typo is reported as a typo.
#[test]
fn label_flags_are_validated_up_front() {
    for (args, want) in [
        (vec!["--clip", "--labels", "only-one", "x.png"], "at least 2 labels"),
        (vec!["--clip", "--labels", "a,b", "--labels-file", "f.txt", "x.png"], "not both"),
        (vec!["--no-llm", "--labels", "a,b", "x.png"], "only applies to"),
        (vec!["--clip", "--labels-file", "/definitely/not/here", "x.png"], "--labels-file"),
    ] {
        let (_, stderr, code) = ross(&args);
        assert_eq!(code, 4, "expected a usage error for {args:?}");
        assert!(stderr.contains(want), "for {args:?} wanted {want:?}, got: {stderr}");
    }

    let f = tmp("labels.txt");
    std::fs::write(&f, "# comment\nsword\n\nshield # trailing\n").unwrap();
    let (_, stderr, code) = ross(&["--clip", "--labels-file", f.to_str().unwrap(), "/no/such/file"]);
    assert_eq!(code, 4);
    assert!(stderr.contains("/no/such/file"), "should fail on the path, not the labels: {stderr}");
}

/// Rust ignores SIGPIPE, so writing through println! makes `ross ... | head`
/// die with "failed printing to stdout: Broken pipe". Piping into a short
/// reader is normal CLI use and must exit cleanly.
#[test]
fn survives_a_closed_pipe() {
    if !have("ffmpeg", "-version") || !have("ffprobe", "-version") {
        skip("ffmpeg/ffprobe not available");
        return;
    }
    let wav = tmp("pipe.wav");
    ffmpeg(&["-f", "lavfi", "-i", "sine=frequency=440:duration=1", "-y", wav.to_str().unwrap()]);
    let bin = env!("CARGO_BIN_EXE_ross");
    for args in [
        format!("{bin} --doctor | head -1"),
        format!("{bin} --no-llm --json --quiet {} | head -1", wav.display()),
        format!("{bin} --no-llm --text --quiet {} | head -1", wav.display()),
    ] {
        let out = Command::new("sh").arg("-c").arg(&args).output().expect("sh");
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(!stderr.contains("Broken pipe") && !stderr.contains("panicked"),
                "`{args}` panicked on a closed pipe: {stderr}");
        assert!(!out.stdout.is_empty(), "`{args}` produced nothing");
    }
}
