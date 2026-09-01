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

/// Everything in a result except the per-file timing, which is meant to vary.
/// Comparing raw stdout across runs would only ever test the clock.
fn stable(json: &str) -> Value {
    fn strip(mut e: Value) -> Value {
        if let Some(o) = e.as_object_mut() {
            o.remove("duration_ms");
        }
        e
    }
    match serde_json::from_str(json.trim()).expect("json") {
        Value::Array(a) => Value::Array(a.into_iter().map(strip).collect()),
        one => strip(one),
    }
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

/// Directory walking: recurses, sorts deterministically, and skips symlinks so a
/// self-referential link cannot loop the traversal.
#[test]
fn walks_directories_deterministically_and_skips_symlinks() {
    if !have("ffmpeg", "-version") || !have("ffprobe", "-version") {
        skip("ffmpeg/ffprobe not available");
        return;
    }
    let root = tmp("walkdir");
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(root.join("nested/deeper")).unwrap();
    for p in ["b.wav", "nested/a.wav", "nested/deeper/c.wav"] {
        ffmpeg(&["-f", "lavfi", "-i", "sine=frequency=440:duration=1", "-y",
                 root.join(p).to_str().unwrap()]);
    }
    std::fs::write(root.join("notmedia.txt"), b"skip me").unwrap();
    #[cfg(unix)]
    let _ = std::os::unix::fs::symlink(&root, root.join("loop"));

    let (stdout, _, code) = ross(&["--no-llm", "--json", "--quiet", root.to_str().unwrap()]);
    assert_eq!(code, 0);
    let v: serde_json::Value = serde_json::from_str(stdout.trim()).unwrap();
    let paths: Vec<String> = v.as_array().unwrap().iter()
        .map(|e| e["path"].as_str().unwrap().to_string()).collect();
    assert_eq!(paths.len(), 3, "3 media files, no .txt, no symlink recursion: {paths:?}");
    let mut sorted = paths.clone();
    sorted.sort();
    assert_eq!(paths, sorted, "output order must be deterministic");

    // same tree twice must give identical results (timing aside)
    let (again, _, _) = ross(&["--no-llm", "--json", "--quiet", root.to_str().unwrap()]);
    assert_eq!(stable(&stdout), stable(&again), "repeat runs must be reproducible");
}

/// Flags that shape a run but were only ever exercised by hand.
#[test]
fn output_and_batching_flags() {
    if !have("ffmpeg", "-version") || !have("ffprobe", "-version") {
        skip("ffmpeg/ffprobe not available");
        return;
    }
    let dir = tmp("flags");
    std::fs::create_dir_all(&dir).unwrap();
    for i in 0..4 {
        ffmpeg(&["-f", "lavfi", "-i", "sine=frequency=440:duration=1", "-y",
                 dir.join(format!("t{i}.wav")).to_str().unwrap()]);
    }
    // --format is the long form of --json/--md/--text
    for (fmt, needle) in [("json", "\"sha256\""), ("md", "## "), ("text", "== ")] {
        let (out, _, code) = ross(&["--no-llm", "--quiet", "--format", fmt, dir.to_str().unwrap()]);
        assert_eq!(code, 0);
        assert!(out.contains(needle), "--format {fmt} produced: {}", &out[..out.len().min(80)]);
    }
    // concurrency must not change results, only how fast they arrive
    let (one, _, _) = ross(&["--no-llm", "--json", "--quiet", "-c", "1", dir.to_str().unwrap()]);
    let (four, _, _) = ross(&["--no-llm", "--json", "--quiet", "-c", "4", dir.to_str().unwrap()]);
    assert_eq!(stable(&one), stable(&four), "concurrency changed the output");

    // --ask is accepted and does not disturb the deterministic pass
    let (out, _, code) = ross(&["--no-llm", "--json", "--quiet", "--ask", "describe tersely",
                                dir.join("t0.wav").to_str().unwrap()]);
    assert_eq!(code, 0);
    assert!(out.contains("\"sha256\""));
}

/// Video: frames are pulled with ffmpeg, so a clip must sniff as video, carry
/// stream metadata, and — with --clip — get tagged from an extracted frame.
/// A file that sniffs as video but yields no frame must error, not silently
/// hand the model an empty attachment.
#[test]
fn video_frames_and_metadata() {
    if !have("ffmpeg", "-version") || !have("ffprobe", "-version") {
        skip("ffmpeg/ffprobe not available");
        return;
    }
    let mp4 = tmp("clip.mp4");
    ffmpeg(&["-f", "lavfi", "-i", "testsrc=duration=2:size=128x96:rate=10",
             "-pix_fmt", "yuv420p", "-y", mp4.to_str().unwrap()]);

    let (stdout, _, code) = ross(&["--no-llm", "--json", "--quiet", mp4.to_str().unwrap()]);
    assert_eq!(code, 0);
    let v: Value = serde_json::from_str(stdout.trim()).unwrap();
    assert_eq!(v["type"], "video");
    assert_eq!(v["metadata"]["width"], 128);
    assert_eq!(v["metadata"]["height"], 96);
    assert!((v["metadata"]["duration_s"].as_f64().unwrap() - 2.0).abs() < 0.2);

    // --clip tags video off a sampled frame
    let (doc, _, _) = ross(&["--doctor"]);
    if doc.lines().any(|l| l.starts_with("clip") && l.contains("ok")) {
        let (stdout, _, code) = ross(&["--clip", "--no-llm", "--json", "--quiet",
                                       mp4.to_str().unwrap()]);
        assert_eq!(code, 0);
        let v: Value = serde_json::from_str(stdout.trim()).unwrap();
        assert!(v["description"].as_str().is_some_and(|d| !d.is_empty()),
                "video should get a description from its frame");
    }

    // sniffs as mp4 by magic bytes, but ffprobe/ffmpeg can make nothing of it
    let broken = tmp("broken.mp4");
    std::fs::write(&broken, b"\x00\x00\x00\x20ftypisom").unwrap();
    let (stdout, _, code) = ross(&["--no-llm", "--json", "--quiet", broken.to_str().unwrap()]);
    assert_eq!(code, 0, "a bad file must not fail the batch");
    let v: Value = serde_json::from_str(stdout.trim()).unwrap();
    assert!(v["error"].as_str().is_some(), "expected a reported error: {v}");
}

/// The per-modality flags each need their own wiring; env coverage does not
/// prove the CLI plumbing.
#[test]
fn every_modality_flag_is_wired() {
    for (flag, model, row) in [
        ("--vision-url", "--vision-model", "vision"),
        ("--audio-url", "--audio-model", "audio"),
        ("--video-url", "--video-model", "video"),
    ] {
        let (out, _, _) = ross(&["--doctor", "--url", "http://global", "--model", "gm",
                                 flag, "http://specific", model, "sm"]);
        assert!(out.lines().any(|l| l.starts_with(row) && l.contains("http://specific")
                                    && l.contains("model=sm")),
                "{flag} did not reach the {row} endpoint:\n{out}");
        // the other two must still fall back to the global setting
        for other in ["vision", "audio", "video"].iter().filter(|r| **r != row) {
            assert!(out.lines().any(|l| l.starts_with(other) && l.contains("http://global")),
                    "{other} should have inherited the global endpoint:\n{out}");
        }
    }
}
