// the crate's guts live in lib.rs so callers can link them; this binary is just
// the CLI wiring on top
use ross::{clap_tag, clip_tag, labels, media, output, semantic};

use clap::{Parser, ValueEnum};
use media::Kind;
use semantic::{Endpoints, Modality, Overrides, Part};
use serde_json::{json, Value};
use std::io::{IsTerminal, Write};
use std::path::{Path, PathBuf};
use std::process::exit;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::Instant;

#[derive(Parser)]
#[command(version, about = "media in → metadata, tags, description, summary out")]
struct Cli {
    #[arg(required_unless_present = "doctor")]
    paths: Vec<PathBuf>,

    #[arg(short, long, value_enum)]
    format: Option<Format>,

    #[arg(long, conflicts_with_all = ["format", "md", "text"])]
    json: bool,

    #[arg(long, conflicts_with_all = ["format", "json", "text"])]
    md: bool,

    #[arg(long, conflicts_with_all = ["format", "json", "md"])]
    text: bool,

    #[arg(long, help = "deterministic fields only, no model call")]
    no_llm: bool,

    #[arg(long, help = "send text (metadata) only, never inline images/frames")]
    no_vision: bool,

    #[arg(long, help = "send raw audio to an audio-capable model (transcodes to mp3)")]
    audio: bool,

    #[arg(long, help = "fast zero-shot audio tags via local CLAP (native ONNX, no python)")]
    clap: bool,

    #[arg(long, help = "zero-shot image/video tags via local CLIP (native ONNX, no python)")]
    clip: bool,

    #[arg(long, value_name = "A,B,C",
          help = "replace the built-in tag vocabulary for --clap/--clip")]
    labels: Option<String>,

    #[arg(long, value_name = "FILE",
          help = "same, one label per line (# comments allowed)")]
    labels_file: Option<String>,

    #[arg(long)]
    ask: Option<String>,

    #[arg(long)]
    ask_file: Option<String>,

    #[arg(long, default_value_t = 4)]
    frames: usize,

    #[arg(short, long)]
    concurrency: Option<usize>,

    #[arg(short, long)]
    quiet: bool,

    #[arg(long, help = "exit 2 if any file errored (results are still printed)")]
    strict: bool,

    #[arg(short, long, help = "global endpoint base URL (ROSS_URL)")]
    url: Option<String>,

    #[arg(short, long, help = "global model name (ROSS_MODEL)")]
    model: Option<String>,

    #[arg(long, help = "image endpoint URL (ROSS_VISION_URL)")]
    vision_url: Option<String>,
    #[arg(long, help = "image model (ROSS_VISION_MODEL)")]
    vision_model: Option<String>,

    #[arg(long, help = "audio endpoint URL (ROSS_AUDIO_URL)")]
    audio_url: Option<String>,
    #[arg(long, help = "audio model (ROSS_AUDIO_MODEL)")]
    audio_model: Option<String>,

    #[arg(long, help = "video endpoint URL (ROSS_VIDEO_URL)")]
    video_url: Option<String>,
    #[arg(long, help = "video model (ROSS_VIDEO_MODEL)")]
    video_model: Option<String>,

    #[arg(long)]
    doctor: bool,
}

impl Cli {
    fn overrides(&self) -> Overrides {
        Overrides {
            url: self.url.clone(),
            model: self.model.clone(),
            vision_url: self.vision_url.clone(),
            vision_model: self.vision_model.clone(),
            audio_url: self.audio_url.clone(),
            audio_model: self.audio_model.clone(),
            video_url: self.video_url.clone(),
            video_model: self.video_model.clone(),
        }
    }
}

#[derive(Clone, Copy, PartialEq, ValueEnum)]
enum Format {
    Json,
    Md,
    Text,
}

/// Rust ignores SIGPIPE, so `ross ... | head` would panic inside println!
/// instead of ending quietly the way every other CLI does. Write explicitly and
/// treat a closed pipe as a normal finish.
fn emit(s: &str) {
    match std::io::stdout().write_all(s.as_bytes()).and_then(|_| std::io::stdout().flush()) {
        Ok(()) => {}
        Err(e) if e.kind() == std::io::ErrorKind::BrokenPipe => exit(0),
        Err(e) => {
            eprintln!("ross: stdout: {e}");
            exit(4);
        }
    }
}

enum Outcome {
    Entry(Value),
    Skipped(String),
}

/// "explosion, impact" -> "Explosion, impact." Shared by the CLAP and CLIP paths.
fn tags_to_description(tags: &[String]) -> String {
    let head: Vec<&str> = tags.iter().take(3).map(|s| s.as_str()).collect();
    let s = format!("{}.", head.join(", "));
    let mut c = s.chars();
    match c.next() {
        Some(f) => f.to_uppercase().collect::<String>() + c.as_str(),
        None => s,
    }
}

fn modality_of(kind: Kind) -> Modality {
    match kind {
        Kind::Image => Modality::Vision,
        Kind::Audio => Modality::Audio,
        Kind::Video => Modality::Video,
    }
}

fn main() {
    let cli = Cli::parse();
    if cli.doctor {
        doctor(&cli);
        return;
    }
    let custom_labels = match labels::parse(cli.labels.as_deref(), cli.labels_file.as_deref()) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("ross: {e}");
            exit(4);
        }
    };
    if custom_labels.is_some() && !(cli.clip || cli.clap) {
        eprintln!("ross: --labels only applies to --clap / --clip");
        exit(4);
    }

    let mut files: Vec<PathBuf> = Vec::new();
    for p in &cli.paths {
        if let Err(e) = walk(p, &mut files) {
            eprintln!("ross: {}: {e}", p.display());
            exit(4);
        }
    }
    if files.is_empty() {
        eprintln!("ross: no input files matched");
        exit(4);
    }
    let single = files.len() == 1;
    let format = cli
        .format
        .or(if cli.json {
            Some(Format::Json)
        } else if cli.md {
            Some(Format::Md)
        } else if cli.text {
            Some(Format::Text)
        } else {
            None
        })
        .unwrap_or({
            if std::io::stdout().is_terminal() {
                Format::Text
            } else {
                Format::Json
            }
        });

    let ask = if let Some(f) = &cli.ask_file {
        match std::fs::read_to_string(f) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("ross: cannot read --ask-file {f}: {e}");
                exit(4);
            }
        }
    } else {
        cli.ask.clone().unwrap_or_else(|| semantic::DEFAULT_ASK.to_string())
    };

    let eps = if cli.no_llm {
        None
    } else {
        let e = Endpoints::resolve(&cli.overrides());
        if !e.any() {
            eprintln!(
                "ross: no endpoint configured. Set ROSS_URL (e.g. http://localhost:11434/v1, \
                 a vLLM /v1, or an OpenAI base) and ROSS_MODEL — or set them per modality with \
                 ROSS_VISION_URL / ROSS_AUDIO_URL / ROSS_VIDEO_URL — or pass --no-llm."
            );
            exit(3);
        }
        Some(e)
    };

    let clipper: Option<std::sync::Arc<clip_tag::ClipTagger>> = if cli.clip {
        match clip_tag::ClipTagger::load(custom_labels.clone()) {
            Ok(t) => Some(std::sync::Arc::new(t)),
            Err(e) => {
                eprintln!("ross: --clip unavailable: {e}");
                exit(5);
            }
        }
    } else {
        None
    };

    let tagger: Option<std::sync::Arc<clap_tag::ClapTagger>> = if cli.clap {
        match clap_tag::ClapTagger::load(custom_labels.clone()) {
            Ok(t) => Some(std::sync::Arc::new(t)),
            Err(e) => {
                // --clap was asked for explicitly; failing silently would hand back
                // results that quietly lack the tags the user came for
                eprintln!("ross: --clap unavailable: {e}");
                exit(5);
            }
        }
    } else {
        None
    };

    let n = cli
        .concurrency
        .unwrap_or_else(|| {
            std::cmp::min(
                8,
                std::thread::available_parallelism().map(|v| v.get()).unwrap_or(4),
            )
        })
        .clamp(1, files.len());
    let next = AtomicUsize::new(0);
    let done = AtomicUsize::new(0);
    let results: Mutex<Vec<Option<Value>>> = Mutex::new((0..files.len()).map(|_| None).collect());

    std::thread::scope(|s| {
        for _ in 0..n {
            s.spawn(|| loop {
                let i = next.fetch_add(1, Ordering::SeqCst);
                if i >= files.len() {
                    break;
                }
                let path = &files[i];
                let outcome = process_file(path, &cli, eps.as_ref(), &ask, tagger.as_ref(), clipper.as_ref());
                let d = done.fetch_add(1, Ordering::SeqCst) + 1;
                if !cli.quiet {
                    match &outcome {
                        Outcome::Entry(e) if e.get("error").is_some() => {
                            eprintln!("[{}/{}] {} (error)", d, files.len(), path.display())
                        }
                        _ => eprintln!("[{}/{}] {}", d, files.len(), path.display()),
                    }
                }
                match outcome {
                    Outcome::Entry(v) => results.lock().unwrap()[i] = Some(v),
                    Outcome::Skipped(msg) => {
                        if !cli.quiet {
                            eprintln!("skipped: {msg}");
                        }
                    }
                }
            });
        }
    });

    let results: Vec<Value> = results.into_inner().unwrap().into_iter().flatten().collect();
    // print before deciding the exit code: a --strict failure on file 4999 must not
    // discard the 4998 results already paid for
    emit(&match format {
        Format::Json => output::json_out(&results, single),
        Format::Md => output::md_out(&results),
        Format::Text => output::text_out(&results),
    });
    if cli.strict {
        if let Some(e) = results.iter().find(|v| v.get("error").is_some()) {
            eprintln!(
                "ross: --strict: {}: {}",
                e["path"].as_str().unwrap_or(""),
                e["error"].as_str().unwrap_or("")
            );
            exit(2);
        }
    }
}

fn walk(p: &Path, out: &mut Vec<PathBuf>) -> std::io::Result<()> {
    // ponytail: explicit stack, not recursion — a deep tree should not blow the
    // thread stack, and this is the same amount of code
    let mut stack = vec![p.to_path_buf()];
    while let Some(p) = stack.pop() {
        let md = std::fs::metadata(&p)?;
        if md.is_dir() {
            let mut entries: Vec<PathBuf> = std::fs::read_dir(&p)?
                .filter_map(|e| e.ok())
                .map(|e| e.path())
                .filter(|e| {
                    !std::fs::symlink_metadata(e)
                        .map(|m| m.file_type().is_symlink())
                        .unwrap_or(true)
                })
                .collect();
            entries.sort();
            // pushed in reverse so the pop order stays alphabetical
            stack.extend(entries.into_iter().rev());
        } else if md.is_file() {
            out.push(p);
        }
    }
    out.sort();
    Ok(())
}

fn process_file(
    path: &Path,
    cli: &Cli,
    eps: Option<&Endpoints>,
    ask: &str,
    tagger: Option<&std::sync::Arc<clap_tag::ClapTagger>>,
    clipper: Option<&std::sync::Arc<clip_tag::ClipTagger>>,
) -> Outcome {
    let t0 = Instant::now();
    let (kind, mime) = match media::sniff(path) {
        Some(k) => k,
        None => return Outcome::Skipped(path.display().to_string()),
    };
    let fs_meta = match std::fs::metadata(path) {
        Ok(m) => m,
        Err(e) => return err_entry(path, e.to_string()),
    };
    let mut entry = json!({
        "path": path.display().to_string(),
        "type": kind.as_str(),
        "mime": mime,
        "bytes": fs_meta.len(),
        "mtime": fs_meta
            .modified()
            .ok()
            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
            .map(|d| d.as_secs())
            .unwrap_or(0),
    });
    match media::sha256_hex(path) {
        Ok(h) => entry["sha256"] = json!(h),
        Err(e) => return err_entry(path, e),
    }
    let (meta, meta_text) = match match kind {
        Kind::Image => media::exif_metadata(path),
        _ => media::ffprobe_metadata(path),
    } {
        Ok(pair) => pair,
        Err(e) => return err_entry(path, e),
    };
    entry["metadata"] = meta;

    let ep = eps.and_then(|e| e.get(modality_of(kind)));

    if let Some(c) = clipper.filter(|_| kind != Kind::Audio) {
        let tagged = match kind {
            Kind::Image => c.tag_path(path),
            // one frame from the middle is enough to say what a clip is of
            _ => match media::grab_frames(path, 1, entry["metadata"]["duration_s"].as_f64()) {
                Ok(f) => match f.first() {
                    Some(b) => c.tag_bytes(b),
                    None => Err("no frame extracted".into()),
                },
                Err(e) => Err(e),
            },
        };
        match tagged {
            Ok(clip_tag::TagResult::Tagged(tags, score)) => {
                entry["description"] = json!(tags_to_description(&tags));
                entry["tags"] = json!(tags);
                entry["tag_confidence"] = json!((score * 1000.0).round() / 1000.0);
                if ep.is_none() {
                    entry["model"] = json!(clip_tag::MODEL_LABEL);
                }
            }
            Ok(clip_tag::TagResult::Gated) => {
                // nothing cleared the bar — fall back to what we actually know
                entry["description"] =
                    json!(format!("{}.", clap_tag::humanize(&path.display().to_string())));
                if ep.is_none() {
                    entry["model"] = json!(clip_tag::MODEL_LABEL);
                }
            }
            Err(e) => eprintln!("clip: {}: {e}", path.display()),
        }
    }

    if let Some(t) = tagger.filter(|_| kind == Kind::Audio) {
        let dur = entry["metadata"]["duration_s"].as_f64().unwrap_or(0.0);
        let gated_short = dur > 0.0 && dur < clap_tag::min_seconds_gate();
        match if gated_short {
            Ok(clap_tag::TagResult::Gated)
        } else {
            t.tag_path(path)
        } {
            Ok(clap_tag::TagResult::Gated) => {
                // no confident match (or too short to judge): say what we actually
                // know — the filename — rather than inventing tags
                entry["description"] = json!(format!(
                    "{}{}.",
                    clap_tag::humanize(&path.display().to_string()),
                    if gated_short { " (short sound effect)" } else { "" }
                ));
                if ep.is_none() {
                    entry["model"] = json!(clap_tag::MODEL_LABEL);
                }
            }
            Ok(clap_tag::TagResult::Tagged(tags, score)) => {
                let desc = tags_to_description(&tags);
                entry["tags"] = json!(tags);
                entry["tag_confidence"] = json!((score * 1000.0).round() / 1000.0);
                entry["description"] = json!(desc);
                if ep.is_none() {
                    entry["model"] = json!(clap_tag::MODEL_LABEL);
                }
            }
            Err(e) => eprintln!("clap: {}: {e}", path.display()),
        }
    }

    if let Some(eps) = eps {
        match eps.get(modality_of(kind)) {
            None => {
                entry["llm_error"] = json!(format!(
                    "no endpoint configured for {}",
                    modality_of(kind).as_str()
                ));
            }
            Some(ep) => {
                // When the media itself is attached, embedded catalogue tags
                // (album/artist/publisher) demonstrably crowd it out: three
                // different sound effects from one asset pack came back with the
                // same publisher-derived description until these were dropped.
                // They stay in the JSON output, they just leave the prompt.
                let prompt_meta = if entry["metadata"]["format_tags"].is_null() {
                    meta_text.clone()
                } else {
                    let mut lean = entry["metadata"].clone();
                    if let Some(o) = lean.as_object_mut() {
                        o.remove("format_tags");
                    }
                    serde_json::to_string_pretty(&lean).unwrap_or_else(|_| meta_text.clone())
                };
                let owned: Vec<(Vec<u8>, String)>;
                let audio: Option<(Vec<u8>, &str)>;
                let text: String;
                match kind {
                    Kind::Image => {
                        let img = match std::fs::read(path) {
                            Ok(b) => b,
                            Err(e) => return err_entry(path, format!("read for vision: {e}")),
                        };
                        owned = vec![(img, mime.to_string())];
                        audio = None;
                        text = format!("Image file.\nMetadata:\n{prompt_meta}");
                    }
                    Kind::Video => {
                        let dur = entry["metadata"]["duration_s"].as_f64();
                        let frames = match media::grab_frames(path, cli.frames.max(1), dur) {
                            Ok(f) => f,
                            Err(e) => return err_entry(path, e),
                        };
                        owned = frames.into_iter().map(|f| (f, "image/png".to_string())).collect();
                        audio = None;
                        text = format!("Video (sample frames attached).\nMetadata:\n{prompt_meta}");
                    }
                    Kind::Audio => {
                        owned = vec![];
                        audio = if cli.audio {
                            match media::audio_for_llm(path, mime == "audio/mpeg") {
                                Ok((bytes, fmt)) => Some((bytes, fmt)),
                                Err(e) => return err_entry(path, e),
                            }
                        } else {
                            None
                        };
                        text = format!("Audio file.\nMetadata:\n{prompt_meta}");
                    }
                }
                let images: Vec<Part<'_>> = owned
                    .iter()
                    .map(|(b, m)| Part { bytes: b, mime: m })
                    .collect();
                let audio_ref = audio.as_ref().map(|(b, f)| (b.as_slice(), *f));
                match semantic::analyze(ep, ask, &images, audio_ref, &text, !cli.no_vision) {
                    Ok(v) => {
                        for k in ["tags", "description", "summary"] {
                            if !v[k].is_null() {
                                entry[k] = clean_field(&v[k]);
                            }
                        }
                        entry["model"] = json!(ep.model);
                    }
                    Err(e) => {
                        if entry.get("tags").is_some() || entry.get("description").is_some() {
                            entry["llm_error"] = json!(e);
                        } else {
                            return err_entry(path, e);
                        }
                    }
                }
            }
        }
    }
    entry["duration_ms"] = json!(t0.elapsed().as_millis() as u64);
    Outcome::Entry(entry)
}

/// Models return the odd `" warrior"` or a blank tag; trim before storing so the
/// output is not littered with whitespace variants of the same label.
fn clean_field(v: &Value) -> Value {
    match v {
        Value::String(s) => json!(s.trim()),
        Value::Array(a) => Value::Array(
            a.iter()
                .filter_map(|x| x.as_str().map(str::trim).filter(|s| !s.is_empty()))
                .map(|s| json!(s))
                .collect(),
        ),
        other => other.clone(),
    }
}

fn err_entry(path: &Path, e: String) -> Outcome {
    Outcome::Entry(json!({
        "path": path.display().to_string(),
        "error": e,
    }))
}

fn doctor(cli: &Cli) {
    use std::fmt::Write as _;
    let mut out = String::new();
    for (name, check) in [("ffprobe", "-version"), ("ffmpeg", "-version"), ("exiftool", "-ver")] {
        let ok = std::process::Command::new(name)
            .arg(check)
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false);
        let _ = writeln!(out, "{name:<10} {}", if ok { "ok" } else { "MISSING (optional)" });
    }
    let eps = Endpoints::resolve(&cli.overrides());
    for m in [Modality::Vision, Modality::Audio, Modality::Video] {
        let _ = match eps.get(m) {
            Some(e) => writeln!(
                out,
                "{:<10} {} model={} key={}",
                m.as_str(),
                e.url,
                e.model,
                if e.key.is_some() { "set" } else { "none" }
            ),
            None => writeln!(out, "{:<10} NOT SET", m.as_str()),
        };
    }
    if !eps.any() {
        out.push_str("\nset ROSS_URL + ROSS_MODEL for all modalities, or ROSS_VISION_URL /\n");
        out.push_str("ROSS_AUDIO_URL / ROSS_VIDEO_URL (+ _MODEL, _API_KEY) to split them.\n");
    }
    out.push('\n');
    for (flag, p) in [("clap", clap_tag::model_path()), ("clip", clip_tag::model_path())] {
        let _ = writeln!(
            out,
            "{flag:<10} {}",
            if p.exists() {
                format!("ok ({})", p.display())
            } else {
                format!("not downloaded (fetched on first --{flag}) -> {}", p.display())
            }
        );
    }
    emit(&out);
}
