//! Zero-shot image tagging with MobileCLIP-S0, the visual twin of `clap_tag`:
//! embed the image, take the cosine against precomputed label embeddings, and
//! only emit what clears a confidence bar.
//!
//! Two things differ from the audio path and both are deliberate:
//!   * fp32 weights, not int8. The int8 export of this model scores 27% on a
//!     4-way task where chance is 25% — quantization destroys it. Measured, not
//!     assumed; see README.
//!   * the vendored label embeddings are mean-centered. Raw CLIP text vectors sit
//!     in a narrow cone (mean pairwise cosine 0.73), which squashes every score
//!     into the same range; centering drops that to -0.02 and makes the margin
//!     between labels mean something.

use ort::session::Session;
use ort::value::Tensor;
use std::path::{Path, PathBuf};

pub const MODEL_LABEL: &str = "clip:mobileclip_s0";
const MODEL_URL: &str =
    "https://huggingface.co/Xenova/mobileclip_s0/resolve/main/onnx/vision_model.onnx";
const MODEL_FILE: &str = "vision_model_fp32.onnx";
const MODEL_BYTES: u64 = 45_500_000;

const SIDE: u32 = 256; // preprocessor_config: shortest_edge 256 + center crop 256
const EMB_DIM: usize = 512;
const MAX_TAGS: usize = 3;

// Calibrated on real asset images (see README). Post-centering the scores are
// comparable across labels, so the same two-tier rule as the audio path applies:
// a strong score alone, or a weaker one that stands clearly apart from the field.
// See clap_tag: the absolute floor is vocabulary-size dependent, the margin is
// not. Override with ROSS_CLIP_MIN_SCORE / _DISTINCT / _MARGIN.
const MIN_SCORE: f64 = 0.055;
const MIN_SCORE_DISTINCT: f64 = 0.030;
const MIN_MARGIN: f64 = 0.012;
const REL_KEEP: f64 = 0.55;

/// Mean-centered, unit-norm CLIP text embeddings, [50][512] row-major f32,
/// in the same order as LABELS. Regenerate both together (see README).
static TEXT_FEATS: &[u8] = include_bytes!("clip_text_feats.bin");
static LABEL_LIST: &str = include_str!("clip_labels.txt");

pub fn labels() -> Vec<&'static str> {
    LABEL_LIST.lines().filter(|l| !l.trim().is_empty()).collect()
}

pub fn cache_dir() -> PathBuf {
    super::clap_tag::cache_dir()
}

pub fn model_path() -> PathBuf {
    cache_dir().join(MODEL_FILE)
}

pub struct ClipTagger {
    session: std::sync::Mutex<Session>,
    text: Vec<f32>,
    labels: Vec<String>,
}

pub enum TagResult {
    /// Nothing cleared the confidence bar.
    Gated,
    Tagged(Vec<String>, f64),
}

impl ClipTagger {
    /// `custom` replaces the built-in vocabulary; its embeddings are computed
    /// once by the text tower and cached (see labels.rs).
    pub fn load(custom: Option<Vec<String>>) -> Result<Self, String> {
        let onnx = model_path();
        super::clap_tag::download_model(&onnx, MODEL_URL, MODEL_BYTES, "CLIP image")?;
        let (labels, text) = match custom {
            Some(l) => {
                let f = crate::labels::embed(&crate::labels::CLIP, &l, EMB_DIM)?;
                (l, f)
            }
            None => {
                let l: Vec<String> = labels().into_iter().map(String::from).collect();
                if TEXT_FEATS.len() != l.len() * EMB_DIM * 4 {
                    return Err(format!(
                        "clip_text_feats.bin holds {} rows but clip_labels.txt has {}",
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
        let session = Session::builder()
            .map_err(|e| format!("ort builder: {e}"))?
            .commit_from_file(&onnx)
            .map_err(|e| format!("load onnx: {e}"))?;
        Ok(ClipTagger { session: std::sync::Mutex::new(session), text, labels })
    }

    pub fn tag_path(&self, path: &Path) -> Result<TagResult, String> {
        let img = image::open(path).map_err(|e| format!("decode image: {e}"))?;
        self.tag_image(img)
    }

    /// For video: frames arrive as encoded PNG bytes already in memory.
    pub fn tag_bytes(&self, bytes: &[u8]) -> Result<TagResult, String> {
        let img = image::load_from_memory(bytes).map_err(|e| format!("decode frame: {e}"))?;
        self.tag_image(img)
    }

    fn tag_image(&self, img: image::DynamicImage) -> Result<TagResult, String> {
        let pixels = preprocess(img);
        let input = Tensor::from_array((vec![1usize, 3, SIDE as usize, SIDE as usize], pixels))
            .map_err(|e| format!("tensor: {e}"))?;
        let emb: Vec<f32> = {
            let mut session = self.session.lock().map_err(|_| "session poisoned")?;
            let out = session
                .run(ort::inputs!["pixel_values" => input])
                .map_err(|e| format!("infer: {e}"))?;
            out["image_embeds"]
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
        if std::env::var("ROSS_CLIP_SCORES").is_ok() {
            let head: Vec<String> = sims
                .iter()
                .take(6)
                .map(|(i, s)| format!("{}={s:.3}", self.labels[*i]))
                .collect();
            eprintln!("clip-scores {}", head.join(" "));
        }
        let top = sims[0].1;
        let second = sims.get(1).map(|(_, s)| *s).unwrap_or(0.0);
        let hi = super::clap_tag::env_f64("ROSS_CLIP_MIN_SCORE", MIN_SCORE);
        let lo = super::clap_tag::env_f64("ROSS_CLIP_MIN_SCORE_DISTINCT", MIN_SCORE_DISTINCT);
        let margin = super::clap_tag::env_f64("ROSS_CLIP_MIN_MARGIN", MIN_MARGIN);
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

/// Resize shortest edge to 256, center crop, scale to 0..1, CHW.
/// `do_normalize` is false for this model — no ImageNet mean/std.
fn preprocess(img: image::DynamicImage) -> Vec<f32> {
    use image::imageops::FilterType;
    let rgb = img.to_rgb8();
    let (w, h) = (rgb.width().max(1), rgb.height().max(1));
    let scale = SIDE as f32 / w.min(h) as f32;
    let (nw, nh) = ((w as f32 * scale).round().max(SIDE as f32) as u32,
                    (h as f32 * scale).round().max(SIDE as f32) as u32);
    let resized = image::imageops::resize(&rgb, nw, nh, FilterType::CatmullRom);
    let (ox, oy) = ((nw - SIDE) / 2, (nh - SIDE) / 2);
    let mut out = vec![0f32; 3 * (SIDE * SIDE) as usize];
    let plane = (SIDE * SIDE) as usize;
    for y in 0..SIDE {
        for x in 0..SIDE {
            let p = resized.get_pixel(ox + x, oy + y).0;
            let i = (y * SIDE + x) as usize;
            out[i] = p[0] as f32 / 255.0;
            out[plane + i] = p[1] as f32 / 255.0;
            out[2 * plane + i] = p[2] as f32 / 255.0;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vendored_labels_and_embeddings_agree() {
        let l = labels();
        assert_eq!(l.len(), 50);
        assert_eq!(TEXT_FEATS.len(), l.len() * EMB_DIM * 4);
        let v: Vec<f32> = TEXT_FEATS
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        for r in 0..l.len() {
            let n: f64 = v[r * EMB_DIM..(r + 1) * EMB_DIM]
                .iter()
                .map(|x| (*x as f64).powi(2))
                .sum::<f64>()
                .sqrt();
            assert!((n - 1.0).abs() < 1e-4, "row {r} ({}) norm {n}", l[r]);
        }
    }

    /// Centering is what makes the scores comparable; if a regenerated blob ever
    /// loses it the labels collapse back into one cone and gating stops working.
    #[test]
    fn embeddings_are_mean_centered() {
        let l = labels();
        let v: Vec<f32> = TEXT_FEATS
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let mut total = 0.0;
        let mut pairs = 0;
        for a in 0..l.len() {
            for b in (a + 1)..l.len() {
                let d: f64 = (0..EMB_DIM)
                    .map(|c| v[a * EMB_DIM + c] as f64 * v[b * EMB_DIM + c] as f64)
                    .sum();
                total += d;
                pairs += 1;
            }
        }
        let mean = total / pairs as f64;
        assert!(mean.abs() < 0.15, "labels are not decorrelated: mean cosine {mean:.3}");
    }

    #[test]
    fn preprocess_shape_and_range() {
        let img = image::DynamicImage::new_rgb8(64, 300);
        let p = preprocess(img);
        assert_eq!(p.len(), 3 * 256 * 256);
        assert!(p.iter().all(|v| (0.0..=1.0).contains(v)));
    }
}
