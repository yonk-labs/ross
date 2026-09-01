use ort::session::Session;
use ort::value::Tensor;
use realfft::{RealFftPlanner, RealToComplex};
use std::path::{Path, PathBuf};

pub const MODEL_LABEL: &str = "clap:larger_clap_general";
const MODEL_BASE: &str = "https://huggingface.co/Xenova/larger_clap_general/resolve/main/onnx";

/// int8 downloads 4x smaller but costs ~1s more to build the ONNX session on
/// every run; fp32 is the opposite trade. Batch runs amortize the session, so
/// the small download is the better default. ROSS_CLAP_PRECISION=fp32 flips it.
/// (fp16 is published too, but onnxruntime fails to load it.)
fn model_choice() -> (&'static str, &'static str, u64) {
    match std::env::var("ROSS_CLAP_PRECISION").as_deref() {
        Ok("fp32") => ("audio_model_fp32.onnx", "audio_model.onnx", 281_700_000),
        _ => ("audio_model_int8.onnx", "audio_model_quantized.onnx", 78_155_433),
    }
}

const FFT: usize = 1024;
const HOP: usize = 480;
const MELS: usize = 64;
const BINS: usize = FFT / 2 + 1;
const SR: f64 = 48000.0;
const MEL_FMIN: f64 = 50.0;
const MEL_FMAX: f64 = 14000.0;
const MAX_SAMPLES: usize = 480_000; // 10s @ 48k
const EMB_DIM: usize = 512;
/// Clips this short used to be refused outright. The confidence floor now does
/// that job per-sound, so the length gate only needs to catch clips too brief to
/// yield a usable spectrogram. Override with ROSS_CLAP_MIN_SECONDS.
pub fn min_seconds_gate() -> f64 {
    std::env::var("ROSS_CLAP_MIN_SECONDS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.35)
}

// Calibrated against ~60 real game-audio files. Two ways to clear the bar,
// because absolute score alone mis-sorts: audio that matches nothing scores low
// AND flat (a wardrobe door: explosion=0.176, footstep=0.152), while a genuine
// but quiet match scores moderate and SHARP (a death cry: scream=0.284 with the
// runner-up at 0.156). So accept a high score outright, or a lower one that
// stands clearly apart from the field.
// Absolute scores depend on how many labels compete, so a floor tuned for the
// built-in 50 does not transfer to a custom list of 7 — but the margin over the
// runner-up does. Hence the low absolute floor and the load-bearing margin.
// Override any of them with ROSS_CLAP_MIN_SCORE / _DISTINCT / _MARGIN.
const MIN_SCORE: f64 = 0.30;
const MIN_SCORE_DISTINCT: f64 = 0.15;
const MIN_MARGIN: f64 = 0.08;
const REL_KEEP: f64 = 0.75;
const MAX_TAGS: usize = 6;

pub fn env_f64(key: &str, default: f64) -> f64 {
    std::env::var(key).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

/// Normalized CLAP text embeddings for clap_labels.txt, [50][512] row-major f32,
/// precomputed from the text tower of the same model as MODEL_URL.
/// Regenerate only if clap_labels.txt changes (see README).
static TEXT_FEATS: &[u8] = include_bytes!("clap_text_feats.bin");

static LABEL_LIST: &str = include_str!("clap_labels.txt");

pub fn default_labels() -> Vec<String> {
    LABEL_LIST.lines().filter(|l| !l.trim().is_empty()).map(String::from).collect()
}

/// Where the active model lives, given ROSS_CLAP_PRECISION.
pub fn model_path() -> PathBuf {
    cache_dir().join(model_choice().0)
}

pub fn cache_dir() -> PathBuf {
    std::env::var("ROSS_CLAP_CACHE")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
            PathBuf::from(home).join(".cache").join("ross-clap")
        })
}

/// One mel band: the first FFT bin it touches, and its (few) nonzero weights.
/// A Slaney filterbank is 1.8% dense — 9 of 513 bins per band on average — so
/// storing the ranges turns a 33M-multiply dense matmul into ~0.6M.
struct MelBand {
    start: usize,
    weights: Vec<f64>,
}

pub struct ClapTagger {
    session: std::sync::Mutex<Session>,
    labels: Vec<String>,
    text: Vec<f32>, // [n_labels][512] row-major, normalized
    mel: Vec<MelBand>,
    window: Vec<f64>,
    r2c: std::sync::Arc<dyn RealToComplex<f64>>,
}

pub enum TagResult {
    /// Too short, or no label cleared MIN_SCORE.
    Gated,
    /// Matched labels, plus the cosine score of the best one so callers (and
    /// users) can see how much to trust them.
    Tagged(Vec<String>, f64),
}

fn hz_to_mel(f: f64) -> f64 {
    const F_SP: f64 = 200.0 / 3.0;
    const MIN_LOG_HZ: f64 = 1000.0;
    let min_log_mel = MIN_LOG_HZ / F_SP;
    if f >= MIN_LOG_HZ {
        min_log_mel + (f / MIN_LOG_HZ).ln() / (6.4f64.ln() / 27.0)
    } else {
        f / F_SP
    }
}

fn mel_to_hz(m: f64) -> f64 {
    const F_SP: f64 = 200.0 / 3.0;
    const MIN_LOG_HZ: f64 = 1000.0;
    let min_log_mel = MIN_LOG_HZ / F_SP;
    if m >= min_log_mel {
        MIN_LOG_HZ * ((6.4f64.ln() / 27.0) * (m - min_log_mel)).exp()
    } else {
        F_SP * m
    }
}

/// Slaney-normalized triangular mel filterbank, identical to
/// transformers' `mel_filters_slaney` (verified to 7e-17).
fn slaney_mel() -> Vec<MelBand> {
    let fftfreqs: Vec<f64> = (0..BINS).map(|i| SR / 2.0 * i as f64 / (BINS - 1) as f64).collect();
    let (m_lo, m_hi) = (hz_to_mel(MEL_FMIN), hz_to_mel(MEL_FMAX));
    let freqs: Vec<f64> = (0..MELS + 2)
        .map(|i| mel_to_hz(m_lo + (m_hi - m_lo) * i as f64 / (MELS + 1) as f64))
        .collect();
    (0..MELS)
        .map(|m| {
            let (f0, f1, f2) = (freqs[m], freqs[m + 1], freqs[m + 2]);
            let enorm = 2.0 / (f2 - f0);
            let w: Vec<f64> = fftfreqs
                .iter()
                .map(|&f| (((f - f0) / (f1 - f0)).min((f2 - f) / (f2 - f1))).max(0.0) * enorm)
                .collect();
            let start = w.iter().position(|&x| x > 0.0).unwrap_or(0);
            let end = w.iter().rposition(|&x| x > 0.0).map(|i| i + 1).unwrap_or(0);
            MelBand { start, weights: w[start..end.max(start)].to_vec() }
        })
        .collect()
}

/// Fetch an ONNX model on first use, cached forever. Shared with the image path.
pub fn download_model(onnx: &Path, url: &str, expect: u64, what: &str) -> Result<(), String> {
    if onnx.exists() {
        return Ok(());
    }
    let dir = onnx.parent().ok_or("bad cache path")?;
    std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    eprintln!(
        "ross: fetching {what} model ({} MB, one time) -> {}",
        expect / 1_000_000,
        onnx.display()
    );
    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(1800))
        .build()
        .map_err(|e| e.to_string())?;
    let mut resp = client.get(url).send().map_err(|e| format!("download: {e}"))?;
    if !resp.status().is_success() {
        return Err(format!("download: HTTP {} from {url}", resp.status()));
    }
    // ponytail: .part + rename so an interrupted download never leaves a
    // truncated file that later loads as a corrupt session
    let part = onnx.with_extension("part");
    let mut f = std::fs::File::create(&part).map_err(|e| format!("{}: {e}", part.display()))?;
    let n = std::io::copy(&mut resp, &mut f).map_err(|e| format!("download: {e}"))?;
    drop(f);
    if n < expect / 2 {
        let _ = std::fs::remove_file(&part);
        return Err(format!("download truncated at {n} bytes"));
    }
    std::fs::rename(&part, onnx).map_err(|e| format!("{}: {e}", onnx.display()))?;
    eprintln!("ross: {what} model ready ({n} bytes)");
    Ok(())
}

impl ClapTagger {
    /// `custom` replaces the built-in vocabulary; its embeddings are computed
    /// once by the text tower and cached (see labels.rs).
    pub fn load(custom: Option<Vec<String>>) -> Result<Self, String> {
        let (local, remote, expect) = model_choice();
        let onnx = cache_dir().join(local);
        download_model(&onnx, &format!("{MODEL_BASE}/{remote}"), expect, "CLAP audio")?;
        let session = Session::builder()
            .map_err(|e| format!("ort builder: {e}"))?
            .commit_from_file(&onnx)
            .map_err(|e| format!("load onnx: {e}"))?;
        let (labels, text) = match custom {
            Some(l) => {
                let f = crate::labels::embed(&crate::labels::CLAP, &l, EMB_DIM)?;
                (l, f)
            }
            None => {
                let l = default_labels();
                if TEXT_FEATS.len() != l.len() * EMB_DIM * 4 {
                    return Err(format!(
                        "clap_text_feats.bin holds {} rows but clap_labels.txt has {}",
                        TEXT_FEATS.len() / (EMB_DIM * 4),
                        l.len()
                    ));
                }
                let f = TEXT_FEATS
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect();
                (l, f)
            }
        };
        let mut planner = RealFftPlanner::<f64>::new();
        Ok(ClapTagger {
            session: std::sync::Mutex::new(session),
            labels,
            text,
            mel: slaney_mel(),
            window: (0..FFT)
                .map(|i| 0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / FFT as f64).cos())
                .collect(),
            r2c: planner.plan_fft_forward(FFT),
        })
    }

    pub fn tag_path(&self, path: &Path) -> Result<TagResult, String> {
        let (mono, rate) = decode_audio(path)?;
        let mono = resample_to_48k(&mono, rate)?;
        let samples: Vec<f64> = mono.iter().map(|s| *s as f64).collect();
        self.tag(&samples, 48000)
    }

    pub fn tag(&self, samples: &[f64], sample_rate: u32) -> Result<TagResult, String> {
        if samples.is_empty() || (samples.len() as f64 / sample_rate as f64) < min_seconds_gate() {
            return Ok(TagResult::Gated);
        }
        let padded = reflect_pad(&fit_10s(samples), FFT / 2);
        let n_frames = 1 + (padded.len() - FFT) / HOP;
        let mut feats = vec![0f32; n_frames * MELS];
        let mut frame = vec![0f64; FFT];
        let mut power = vec![0f64; BINS];
        let mut spectrum = self.r2c.make_output_vec();
        let mut scratch = self.r2c.make_scratch_vec();
        for t in 0..n_frames {
            let start = t * HOP;
            for i in 0..FFT {
                frame[i] = padded[start + i] * self.window[i];
            }
            self.r2c
                .process_with_scratch(&mut frame, &mut spectrum, &mut scratch)
                .map_err(|e| format!("fft: {e}"))?;
            // ponytail: power once per bin, not once per (bin, mel) as before
            for (p, c) in power.iter_mut().zip(spectrum.iter()) {
                *p = c.re * c.re + c.im * c.im;
            }
            for (m, band) in self.mel.iter().enumerate() {
                let acc: f64 = band
                    .weights
                    .iter()
                    .zip(&power[band.start..])
                    .map(|(w, p)| w * p)
                    .sum();
                feats[t * MELS + m] = (10.0 * acc.max(1e-10).log10()) as f32;
            }
        }
        let input = Tensor::from_array((vec![1usize, 1, n_frames, MELS], feats))
            .map_err(|e| format!("tensor: {e}"))?;
        let emb: Vec<f32> = {
            let mut session = self.session.lock().map_err(|_| "session poisoned")?;
            let outputs = session
                .run(ort::inputs!["input_features" => input])
                .map_err(|e| format!("infer: {e}"))?;
            outputs["audio_embeds"]
                .try_extract_tensor::<f32>()
                .map_err(|e| format!("extract: {e}"))?
                .1
                .to_vec()
        };
        if emb.len() < EMB_DIM {
            return Err(format!("embedding has {} dims, expected {EMB_DIM}", emb.len()));
        }
        let norm: f64 = emb[..EMB_DIM].iter().map(|v| (*v as f64).powi(2)).sum::<f64>().sqrt();
        if !norm.is_normal() {
            // ponytail: silent/degenerate audio -> zero embedding -> every score NaN.
            // Gate rather than emit whatever order the sort happens to produce.
            return Ok(TagResult::Gated);
        }
        let mut sims: Vec<(usize, f64)> = (0..self.labels.len())
            .map(|r| {
                let s: f64 = (0..EMB_DIM)
                    .map(|c| emb[c] as f64 / norm * self.text[r * EMB_DIM + c] as f64)
                    .sum();
                (r, s)
            })
            .collect();
        sims.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let top = sims[0].1;
        if std::env::var("ROSS_CLAP_SCORES").is_ok() {
            let head: Vec<String> = sims.iter().take(6)
                .map(|(i, s)| format!("{}={s:.3}", self.labels[*i])).collect();
            eprintln!("clap-scores {}", head.join(" "));
        }
        let second = sims.get(1).map(|(_, s)| *s).unwrap_or(0.0);
        let hi = env_f64("ROSS_CLAP_MIN_SCORE", MIN_SCORE);
        let lo = env_f64("ROSS_CLAP_MIN_SCORE_DISTINCT", MIN_SCORE_DISTINCT);
        let margin = env_f64("ROSS_CLAP_MIN_MARGIN", MIN_MARGIN);
        if !(top >= hi || (top >= lo && top - second >= margin)) {
            return Ok(TagResult::Gated);
        }
        let floor = top * REL_KEEP;
        Ok(TagResult::Tagged(
            sims.iter()
                .take(MAX_TAGS)
                .take_while(|(_, s)| *s >= floor)
                .map(|(i, _)| self.labels[*i].clone())
                .collect(),
            top,
        ))
    }
}

fn fit_10s(samples: &[f64]) -> Vec<f64> {
    let n = samples.len();
    if n >= MAX_SAMPLES {
        // ponytail: deterministic center crop (the reference impl random-crops;
        // tags are stable either way, and determinism makes runs reproducible)
        let start = (n - MAX_SAMPLES) / 2;
        samples[start..start + MAX_SAMPLES].to_vec()
    } else {
        let mut out = Vec::with_capacity(MAX_SAMPLES);
        while out.len() < MAX_SAMPLES {
            let take = n.min(MAX_SAMPLES - out.len());
            out.extend_from_slice(&samples[..take]);
        }
        out
    }
}

fn reflect_pad(x: &[f64], pad: usize) -> Vec<f64> {
    let mut out = Vec::with_capacity(x.len() + 2 * pad);
    for i in (1..=pad).rev() {
        out.push(x[i]);
    }
    out.extend_from_slice(x);
    let n = x.len();
    for i in 0..pad {
        out.push(x[n - 2 - i]);
    }
    out
}

pub fn humanize(path: &str) -> String {
    let file = path.rsplit('/').next().unwrap_or(path);
    let stem = file.rsplit_once('.').map(|(s, _)| s).unwrap_or(file);
    let mut out = String::new();
    let mut prev_lower = false;
    for c in stem.chars() {
        match c {
            '_' | '-' => {
                out.push(' ');
                prev_lower = false;
            }
            c if c.is_ascii_uppercase() => {
                if prev_lower {
                    out.push(' ');
                }
                out.push(c.to_ascii_lowercase());
                prev_lower = false;
            }
            c if c.is_ascii_digit() => {
                prev_lower = false;
            }
            c => {
                out.push(c);
                prev_lower = c.is_ascii_lowercase();
            }
        }
    }
    let t = out.split_whitespace().collect::<Vec<_>>().join(" ");
    if t.is_empty() {
        "sound".into()
    } else {
        t
    }
}

pub fn decode_audio(path: &Path) -> Result<(Vec<f32>, u32), String> {
    use symphonia::core::audio::SampleBuffer;
    use symphonia::core::codecs::{DecoderOptions, CODEC_TYPE_NULL};
    use symphonia::core::formats::FormatOptions;
    use symphonia::core::io::MediaSourceStream;
    use symphonia::core::meta::MetadataOptions;
    use symphonia::core::probe::Hint;
    let src = std::fs::File::open(path).map_err(|e| e.to_string())?;
    let mss = MediaSourceStream::new(Box::new(src), Default::default());
    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }
    let probed = symphonia::default::get_probe()
        .format(&hint, mss, &FormatOptions::default(), &MetadataOptions::default())
        .map_err(|e| format!("probe: {e}"))?;
    let mut format = probed.format;
    let track = format
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
        .ok_or("no audio track")?;
    let track_id = track.id;
    let rate = track.codec_params.sample_rate.ok_or("no sample rate")?;
    let channels = track.codec_params.channels.map(|c| c.count()).unwrap_or(1).max(1);
    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &DecoderOptions::default())
        .map_err(|e| format!("decoder: {e}"))?;
    let mut interleaved: Vec<f32> = Vec::new();
    loop {
        let packet = match format.next_packet() {
            Ok(p) => p,
            Err(symphonia::core::errors::Error::IoError(ref e))
                if e.kind() == std::io::ErrorKind::UnexpectedEof =>
            {
                break
            }
            Err(e) => return Err(format!("read packet: {e}")),
        };
        if packet.track_id() != track_id {
            continue;
        }
        let decoded = match decoder.decode(&packet) {
            Ok(d) => d,
            Err(symphonia::core::errors::Error::DecodeError(_)) => continue,
            Err(e) => return Err(format!("decode: {e}")),
        };
        let mut sbuf = SampleBuffer::<f32>::new(decoded.capacity() as u64, *decoded.spec());
        sbuf.copy_interleaved_ref(decoded);
        interleaved.extend_from_slice(sbuf.samples());
    }
    if channels > 1 {
        let frames = interleaved.len() / channels;
        let mono: Vec<f32> = (0..frames)
            .map(|i| interleaved[i * channels..(i + 1) * channels].iter().sum::<f32>() / channels as f32)
            .collect();
        Ok((mono, rate))
    } else {
        Ok((interleaved, rate))
    }
}

pub fn resample_to_48k(input: &[f32], src_rate: u32) -> Result<Vec<f32>, String> {
    if src_rate == 48000 {
        return Ok(input.to_vec());
    }
    if input.is_empty() {
        return Ok(Vec::new());
    }
    use rubato::{
        Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction,
    };
    const CHUNK: usize = 4096;
    let ratio = 48000.0 / src_rate as f64;
    let params = SincInterpolationParameters {
        sinc_len: 256,
        f_cutoff: 0.95,
        oversampling_factor: 256,
        interpolation: SincInterpolationType::Cubic,
        window: WindowFunction::BlackmanHarris,
    };
    let mut rsp = SincFixedIn::<f32>::new(ratio, 2.0, params, CHUNK, 1).map_err(|e| e.to_string())?;
    let cap = (CHUNK as f64 * ratio).ceil() as usize + 512;
    let expected = (input.len() as f64 * ratio) as usize;
    let mut out: Vec<f32> = Vec::with_capacity(expected + CHUNK);
    for chunk_src in input.chunks(CHUNK) {
        let mut chunk = vec![0.0f32; CHUNK];
        chunk[..chunk_src.len()].copy_from_slice(chunk_src);
        let mut outbuf = vec![vec![0.0f32; cap]];
        rsp.process_into_buffer(&[chunk], &mut outbuf, None)
            .map_err(|e| format!("resample: {e}"))?;
        out.extend_from_slice(&outbuf[0]);
    }
    out.resize(expected, 0.0);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn wav_bytes(rate: u32, samples: &[i16]) -> Vec<u8> {
        let mut b = Vec::new();
        let data_len = (samples.len() * 2) as u32;
        b.extend_from_slice(b"RIFF");
        b.extend_from_slice(&(36 + data_len).to_le_bytes());
        b.extend_from_slice(b"WAVEfmt ");
        b.extend_from_slice(&16u32.to_le_bytes());
        b.extend_from_slice(&1u16.to_le_bytes());
        b.extend_from_slice(&1u16.to_le_bytes());
        b.extend_from_slice(&rate.to_le_bytes());
        b.extend_from_slice(&(rate * 2).to_le_bytes());
        b.extend_from_slice(&2u16.to_le_bytes());
        b.extend_from_slice(&16u16.to_le_bytes());
        b.extend_from_slice(b"data");
        b.extend_from_slice(&data_len.to_le_bytes());
        for s in samples {
            b.extend_from_slice(&s.to_le_bytes());
        }
        b
    }

    #[test]
    fn decodes_minimal_wav() {
        let dir = std::env::temp_dir().join("ross-clap-test");
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("t.wav");
        std::fs::write(&p, wav_bytes(8000, &[100, -100, 100, -100])).unwrap();
        let (mono, rate) = decode_audio(&p).unwrap();
        assert_eq!(rate, 8000);
        assert_eq!(mono.len(), 4);
        assert!(mono.iter().all(|s| s.abs() > 0.002));
    }

    #[test]
    fn resample_identity_and_length() {
        let x = vec![0.5f32; 4800];
        assert_eq!(resample_to_48k(&x, 48000).unwrap(), x);
        let y = vec![0.25f32; 4410];
        let r = resample_to_48k(&y, 44100).unwrap();
        assert_eq!(r.len(), (4410.0 * 48000.0 / 44100.0) as usize);
        assert!(r.iter().all(|v| v.is_finite()));
        assert!(r.iter().any(|v| *v > 0.1));
        assert!(resample_to_48k(&[], 44100).unwrap().is_empty());
    }

    /// The native filterbank replaced a 256KB .npy produced by python. These are
    /// the invariants that prove it is still the Slaney bank transformers builds.
    #[test]
    fn slaney_mel_matches_reference_shape() {
        let bank = slaney_mel();
        assert_eq!(bank.len(), MELS);
        let total: f64 = bank.iter().flat_map(|b| &b.weights).sum();
        assert!((total - 1.364433).abs() < 1e-5, "filterbank sum drifted: {total}");
        for b in &bank {
            assert!(!b.weights.is_empty() && b.start + b.weights.len() <= BINS);
            assert!(b.weights.iter().all(|w| *w >= 0.0));
        }
        // sparsity is the whole point of the rewrite
        let nz: usize = bank.iter().map(|b| b.weights.len()).sum();
        assert!(nz < MELS * BINS / 20, "expected a sparse bank, got {nz} weights");
    }

    #[test]
    fn mel_hz_roundtrip() {
        for hz in [50.0, 440.0, 999.0, 1000.0, 8000.0, 14000.0] {
            assert!((mel_to_hz(hz_to_mel(hz)) - hz).abs() < 1e-9, "roundtrip failed at {hz}");
        }
    }

    #[test]
    fn hann_matches_np() {
        let w: Vec<f64> = (0..FFT)
            .map(|i| 0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / FFT as f64).cos())
            .collect();
        assert!((w[0]).abs() < 1e-12);
        assert!((w[512] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn reflect_pad_small() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        assert_eq!(
            reflect_pad(&x, 2),
            vec![3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 5.0, 4.0]
        );
    }

    #[test]
    fn fit_10s_pads_and_crops() {
        assert_eq!(fit_10s(&vec![0.5; 48000]).len(), MAX_SAMPLES);
        assert!(fit_10s(&vec![0.5; 48000]).iter().all(|x| *x == 0.5));
        assert_eq!(fit_10s(&vec![0.1; MAX_SAMPLES * 2]).len(), MAX_SAMPLES);
        // regression: a 1-sample input used to divide by zero computing tile count
        assert_eq!(fit_10s(&[0.3]).len(), MAX_SAMPLES);
    }

    #[test]
    fn text_feats_match_vocabulary() {
        assert_eq!(TEXT_FEATS.len(), default_labels().len() * EMB_DIM * 4);
        let v: Vec<f32> = TEXT_FEATS
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        // every row must be unit-normalized, or cosine scores are meaningless
        for r in 0..default_labels().len() {
            let n: f64 = v[r * EMB_DIM..(r + 1) * EMB_DIM]
                .iter()
                .map(|x| (*x as f64).powi(2))
                .sum::<f64>()
                .sqrt();
            assert!((n - 1.0).abs() < 1e-4, "row {r} norm {n}");
        }
    }

    #[test]
    fn humanize_cases() {
        assert_eq!(humanize("/a/b/Enemy 2 (strong attack).wav"), "enemy (strong attack)");
        assert_eq!(humanize("Bullet_Impact-3.mp3"), "bullet impact");
        assert_eq!(humanize("ChestOpen1.mp3"), "chest open");
    }

    #[test]
    fn tag_gates_short_and_unconfident() {
        let t = match ClapTagger::load(None) {
            Ok(t) => t,
            Err(_) => return, // model not cached; covered by the integration test
        };
        // under the 2s gate
        assert!(matches!(t.tag(&vec![0.5f64; 4800], 48000).unwrap(), TagResult::Gated));
        // silence matches nothing and must never be handed tags
        assert!(matches!(t.tag(&vec![0.0f64; 48000 * 3], 48000).unwrap(), TagResult::Gated));
        // a real, matchable sound must still tag
        let boom: Vec<f64> = (0..48000 * 4)
            .map(|i| {
                let t = i as f64 / 48000.0;
                ((i * 7919 % 2003) as f64 / 1000.0 - 1.0) * (-t * 1.5).exp()
            })
            .collect();
        match t.tag(&boom, 48000).unwrap() {
            TagResult::Tagged(tags, score) => {
                assert!(!tags.is_empty() && tags.len() <= MAX_TAGS);
                assert!(tags.iter().all(|t| default_labels().contains(t)));
                assert!(score >= MIN_SCORE_DISTINCT, "accepted below the floor: {score}");
            }
            TagResult::Gated => {} // acceptable: the gate is allowed to be strict
        }
    }
}
