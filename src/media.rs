use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::io::Read;
use std::path::Path;
use std::process::{Command, Output, Stdio};
use std::time::Duration;
use wait_timeout::ChildExt;
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Kind {
    Image,
    Audio,
    Video,
}

impl Kind {
    pub fn as_str(self) -> &'static str {
        match self {
            Kind::Image => "image",
            Kind::Audio => "audio",
            Kind::Video => "video",
        }
    }
}

pub fn sniff(path: &Path) -> Option<(Kind, &'static str)> {
    let mut f = std::fs::File::open(path).ok()?;
    let mut b = [0u8; 12];
    let n = f.read(&mut b).ok()?;
    sniff_bytes(&b[..n])
}

/// Magic-byte sniff. Pure so it is testable without touching the filesystem.
pub fn sniff_bytes(b: &[u8]) -> Option<(Kind, &'static str)> {
    let s = |r: std::ops::Range<usize>| b.get(r).map(|x| x as &[u8]);
    let m = (b.first().copied().unwrap_or(0), b.get(1).copied().unwrap_or(0));
    match m {
        (0xFF, 0xD8) => return Some((Kind::Image, "image/jpeg")),
        (0x89, b'P') if s(0..8) == Some(&[0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A]) => {
            return Some((Kind::Image, "image/png"))
        }
        _ => {}
    }
    if b.starts_with(b"GIF8") {
        return Some((Kind::Image, "image/gif"));
    }
    if b.starts_with(b"BM") {
        return Some((Kind::Image, "image/bmp"));
    }
    if b.starts_with(b"fLaC") {
        return Some((Kind::Audio, "audio/flac"));
    }
    if b.starts_with(b"OggS") {
        return Some((Kind::Audio, "audio/ogg"));
    }
    if b.starts_with(b"ID3") || (m.0 == 0xFF && (m.1 & 0xE0) == 0xE0 && m.1 != 0xD8) {
        return Some((Kind::Audio, "audio/mpeg"));
    }
    if b.starts_with(b"RIFF") && b.len() >= 12 {
        if &b[8..12] == b"WAVE" {
            return Some((Kind::Audio, "audio/wav"));
        }
        if &b[8..12] == b"WEBP" {
            return Some((Kind::Image, "image/webp"));
        }
        if &b[8..12] == b"AVI " {
            return Some((Kind::Video, "video/avi"));
        }
    }
    if b.starts_with(&[0x1A, 0x45, 0xDF, 0xA3]) {
        return Some((Kind::Video, "video/x-matroska"));
    }
    if b.starts_with(b"II*\0") || b.starts_with(b"MM\0*") {
        return Some((Kind::Image, "image/tiff"));
    }
    if b.len() >= 12 && &b[4..8] == b"ftyp" {
        let brand = &b[8..12];
        return match brand {
            b"M4A " | b"M4B " => Some((Kind::Audio, "audio/mp4")),
            b"avif" | b"avis" => Some((Kind::Image, "image/avif")),
            b"heic" | b"heix" | b"mif1" | b"msf1" => Some((Kind::Image, "image/heic")),
            _ => Some((Kind::Video, "video/mp4")),
        };
    }
    None
}

pub fn sha256_hex(path: &Path) -> Result<String, String> {
    let mut f = std::fs::File::open(path).map_err(|e| e.to_string())?;
    let mut h = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = f.read(&mut buf).map_err(|e| e.to_string())?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    Ok(format!("{:x}", h.finalize()))
}

fn run_cmd(cmd: &mut Command, timeout: Duration) -> Result<Output, String> {
    let prog = cmd.get_program().to_string_lossy().to_string();
    cmd.stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let mut child = cmd
        .spawn()
        .map_err(|e| format!("{prog}: {e} (is it installed / on PATH?)"))?;
    let so = child.stdout.take();
    let se = child.stderr.take();
    let t1 = std::thread::spawn(move || {
        let mut v = Vec::new();
        if let Some(mut s) = so {
            let _ = s.read_to_end(&mut v);
        }
        v
    });
    let t2 = std::thread::spawn(move || {
        let mut v = Vec::new();
        if let Some(mut s) = se {
            let _ = s.read_to_end(&mut v);
        }
        v
    });
    let status = match child.wait_timeout(timeout).map_err(|e| e.to_string())? {
        Some(s) => s,
        None => {
            let _ = child.kill();
            let _ = child.wait();
            // join the drain threads so they end with the process, not later
            let _ = t1.join();
            let _ = t2.join();
            return Err(format!("{prog}: timed out after {timeout:?}"));
        }
    };
    let out = Output {
        status,
        stdout: t1.join().unwrap_or_default(),
        stderr: t2.join().unwrap_or_default(),
    };
    if !out.status.success() {
        let tail = String::from_utf8_lossy(&out.stderr);
        let tail: Vec<&str> = tail.lines().collect();
        let tail = tail[tail.len().saturating_sub(5)..].join("\n");
        return Err(format!("{prog} exited {}: {}", out.status, tail));
    }
    Ok(out)
}

fn missing_binary(e: &str) -> bool {
    e.contains("No such file") || e.contains("not found") || e.contains("is it installed")
}

pub fn ffprobe_metadata(path: &Path) -> Result<(Value, String), String> {
    let out = run_cmd(
        Command::new("ffprobe")
            .args(["-v", "error", "-print_format", "json", "-show_format", "-show_streams"])
            .arg(path),
        Duration::from_secs(30),
    )?;
    let v: Value = serde_json::from_slice(&out.stdout).map_err(|e| format!("ffprobe output: {e}"))?;
    let fmt = &v["format"];
    let mut meta = json!({});
    if let Some(d) = fmt["duration"].as_str().and_then(|s| s.parse::<f64>().ok()) {
        meta["duration_s"] = json!(d);
    }
    if let Some(br) = fmt["bit_rate"].as_str().and_then(|s| s.parse::<u64>().ok()) {
        meta["bitrate"] = json!(br);
    }
    if let Some(name) = fmt["format_name"].as_str() {
        meta["container"] = json!(name);
    }
    if let Some(tags) = fmt["tags"].as_object() {
        let clean: serde_json::Map<String, Value> = tags
            .iter()
            .filter(|(_, v)| v.as_str().map(|s| s.len() <= 512).unwrap_or(true))
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        meta["format_tags"] = Value::Object(clean);
    }
    for s in v["streams"].as_array().into_iter().flatten() {
        if s["codec_type"] == "video" && meta.get("codec").is_none() {
            meta["codec"] = s["codec_name"].clone();
            meta["width"] = s["width"].clone();
            meta["height"] = s["height"].clone();
            meta["fps"] = s["avg_frame_rate"].clone();
        } else if s["codec_type"] == "audio" && meta.get("audio_codec").is_none() {
            meta["audio_codec"] = s["codec_name"].clone();
            if let Some(sr) = s["sample_rate"].as_str() {
                meta["sample_rate"] = json!(sr.parse::<u64>().unwrap_or(0));
            }
        }
    }
    let text = serde_json::to_string_pretty(&meta).map_err(|e| e.to_string())?;
    Ok((meta, text))
}

pub fn exif_metadata(path: &Path) -> Result<(Value, String), String> {
    let out = match run_cmd(Command::new("exiftool").arg("-j").arg(path), Duration::from_secs(30)) {
        Ok(o) => o,
        Err(e) if missing_binary(&e) => {
            let mut meta = json!({});
            if let Some((w, h)) = png_webp_dims(path) {
                meta["width"] = json!(w);
                meta["height"] = json!(h);
            }
            meta["note"] = json!("exiftool not found; minimal metadata only");
            return Ok((meta.clone(), serde_json::to_string_pretty(&meta).unwrap()));
        }
        Err(e) => return Err(e),
    };
    let v: Value = serde_json::from_slice(&out.stdout).map_err(|e| format!("exiftool output: {e}"))?;
    let obj = v
        .as_array()
        .and_then(|a| a.first())
        .cloned()
        .unwrap_or(json!({}));
    // ponytail: width/height promoted to the top level because callers read them
    // there; everything else lives once, under `exif`. Emitting the same field as
    // ImageWidth + width + exif.ImageWidth tripled the tokens sent to the model.
    let mut meta = json!({});
    if let Some(w) = obj["ImageWidth"].as_u64() {
        meta["width"] = json!(w);
    }
    if let Some(h) = obj["ImageHeight"].as_u64() {
        meta["height"] = json!(h);
    }
    let mut exif = json!({});
    if let Some(o) = obj.as_object() {
        for (k, v) in o {
            let short = match v.as_str() {
                Some(s) => s.len() <= 512,
                None => true,
            };
            if short && !k.starts_with("SourceFile") && k != "ExifTool" {
                exif[k] = v.clone();
            }
        }
    }
    meta["exif"] = exif;
    let text = serde_json::to_string_pretty(&meta).map_err(|e| e.to_string())?;
    Ok((meta, text))
}

fn png_webp_dims(path: &Path) -> Option<(u32, u32)> {
    let mut f = std::fs::File::open(path).ok()?;
    let mut b = [0u8; 32];
    let n = f.read(&mut b).ok()?;
    let b = &b[..n];
    if b.starts_with(&[0x89, b'P', b'N', b'G']) && b.len() >= 24 && &b[12..16] == b"IHDR" {
        let w = u32::from_be_bytes([b[16], b[17], b[18], b[19]]);
        let h = u32::from_be_bytes([b[20], b[21], b[22], b[23]]);
        return Some((w, h));
    }
    if b.starts_with(b"RIFF") && b.len() >= 30 && &b[8..12] == b"WEBP" {
        if &b[12..16] == b"VP8X" {
            let w = 1 + (b[24] as u32 | (b[25] as u32) << 8 | (b[26] as u32) << 16);
            let h = 1 + (b[27] as u32 | (b[28] as u32) << 8 | (b[29] as u32) << 16);
            return Some((w, h));
        }
        if &b[12..16] == b"VP8 " && b.len() >= 30 {
            let w = u16::from_le_bytes([b[26], b[27]]) as u32 & 0x3fff;
            let h = u16::from_le_bytes([b[28], b[29]]) as u32 & 0x3fff;
            return Some((w, h));
        }
        if &b[12..16] == b"VP8L" {
            let bits = u32::from_le_bytes([b[21], b[22], b[23], b[24]]);
            let w = (bits & 0x3fff) + 1;
            let h = ((bits >> 14) & 0x3fff) + 1;
            return Some((w, h));
        }
    }
    None
}

static FRAME_SEQ: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

pub fn grab_frames(path: &Path, n: usize, duration_s: Option<f64>) -> Result<Vec<Vec<u8>>, String> {
    let mut frames = Vec::new();
    for i in 0..n {
        let t = match duration_s {
            Some(d) if d > 0.0 => d * (i as f64 + 0.5) / n as f64,
            _ => 0.0,
        };
        let seq = FRAME_SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let tmp = std::env::temp_dir().join(format!(
            "ross-frame-{}-{seq}.png",
            std::process::id()
        ));
        let out = run_cmd(
            Command::new("ffmpeg")
                .args(["-hide_banner", "-loglevel", "error", "-ss"])
                .arg(format!("{t:.3}"))
                .arg("-i")
                .arg(path)
                .args(["-frames:v", "1", "-y"])
                .arg(&tmp),
            Duration::from_secs(60),
        );
        match out {
            Ok(_) => {
                if let Ok(bytes) = std::fs::read(&tmp) {
                    if !bytes.is_empty() {
                        frames.push(bytes);
                    }
                }
            }
            Err(e) if i == 0 && frames.is_empty() && missing_binary(&e) => {
                let _ = std::fs::remove_file(&tmp);
                return Err(e);
            }
            Err(_) => {}
        }
        let _ = std::fs::remove_file(&tmp);
        if frames.len() == n {
            break;
        }
    }
    if frames.is_empty() {
        // otherwise the model is asked to describe a video with nothing attached
        return Err(format!("ffmpeg extracted no frames from {}", path.display()));
    }
    Ok(frames)
}

pub fn audio_for_llm(path: &Path, is_mp3: bool) -> Result<(Vec<u8>, &'static str), String> {
    if is_mp3 {
        return Ok((std::fs::read(path).map_err(|e| e.to_string())?, "mp3"));
    }
    // ponytail: transcode everything else to 64k mono mp3 — wav base64 is ~10x bigger than needed
    let tmp = std::env::temp_dir().join(format!(
        "ross-audio-{}-{}.mp3",
        std::process::id(),
        FRAME_SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    ));
    let out = run_cmd(
        Command::new("ffmpeg")
            .args(["-hide_banner", "-loglevel", "error", "-i"])
            .arg(path)
            .args(["-ac", "1", "-ar", "16000", "-b:a", "64k", "-y"])
            .arg(&tmp),
        Duration::from_secs(300),
    );
    match out {
        Ok(_) => {
            let bytes = std::fs::read(&tmp).unwrap_or_default();
            let _ = std::fs::remove_file(&tmp);
            if bytes.is_empty() {
                return Err("ffmpeg produced empty audio".into());
            }
            Ok((bytes, "mp3"))
        }
        Err(e) => {
            let _ = std::fs::remove_file(&tmp);
            Err(e)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sniff_formats() {
        let png = [0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A];
        assert!(matches!(sniff_bytes(&png), Some((Kind::Image, "image/png"))));
        let jpeg = [0xFF, 0xD8, 0xFF, 0xE0];
        assert!(matches!(sniff_bytes(&jpeg), Some((Kind::Image, "image/jpeg"))));
        let mp4 = [0, 0, 0, 0x20, b'f', b't', b'y', b'p', b'i', b's', b'o', b'm'];
        assert!(matches!(sniff_bytes(&mp4), Some((Kind::Video, "video/mp4"))));
        let m4a = [0, 0, 0, 0x20, b'f', b't', b'y', b'p', b'M', b'4', b'A', b' '];
        assert!(matches!(sniff_bytes(&m4a), Some((Kind::Audio, "audio/mp4"))));
        let wav = *b"RIFF\x00\x00\x00\x00WAVEfmt ";
        assert!(matches!(sniff_bytes(&wav), Some((Kind::Audio, "audio/wav"))));
        let webm = [0x1A, 0x45, 0xDF, 0xA3, 0x00, 0x00];
        assert!(matches!(
            sniff_bytes(&webm),
            Some((Kind::Video, "video/x-matroska"))
        ));
        let mp3 = *b"ID3\x04\x00\x00\x00\x00\x00\x00";
        assert!(matches!(sniff_bytes(&mp3), Some((Kind::Audio, "audio/mpeg"))));
        let mp3raw = [0xFF, 0xFB, 0x90, 0x00];
        assert!(matches!(sniff_bytes(&mp3raw), Some((Kind::Audio, "audio/mpeg"))));
        let txt = *b"hello world!!";
        assert!(sniff_bytes(&txt).is_none());
        let webp = *b"RIFF\x00\x00\x00\x00WEBPVP8 ";
        assert!(matches!(sniff_bytes(&webp), Some((Kind::Image, "image/webp"))));
    }

    #[test]
    fn sniff_short_and_empty_inputs() {
        assert!(sniff_bytes(&[]).is_none());
        assert!(sniff_bytes(&[0xFF]).is_none());
        assert!(sniff_bytes(b"RIFF").is_none()); // needs 12 bytes to disambiguate
    }
}
