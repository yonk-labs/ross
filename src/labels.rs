//! Runtime label embedding, so the tag vocabulary is not frozen at build time.
//!
//! The default label sets ship precomputed (`clip_text_feats.bin`,
//! `clap_text_feats.bin`) so the common path needs no text model at all. When
//! `--labels` / `--labels-file` supplies a different set, the matching text
//! tower is downloaded once and the resulting embeddings are cached on disk,
//! keyed by a hash of the exact label list — so only the first run pays.

use sha2::{Digest, Sha256};
use std::path::PathBuf;

pub struct TextTower {
    pub name: &'static str,
    pub model_url: &'static str,
    pub model_file: &'static str,
    pub model_bytes: u64,
    pub tokenizer_url: &'static str,
    pub tokenizer_file: &'static str,
    pub output: &'static str,
    /// CLIP's text encoder wants a fixed 77-token context, padded with id 0.
    pub pad_to: Option<usize>,
    /// CLIP text vectors sit in a narrow cone; centering makes scores comparable.
    pub center: bool,
}

pub const CLIP: TextTower = TextTower {
    name: "CLIP text",
    model_url:
        "https://huggingface.co/Xenova/clip-vit-base-patch32/resolve/main/onnx/text_model.onnx",
    model_file: "clip_text.onnx",
    model_bytes: 254_100_000,
    tokenizer_url:
        "https://huggingface.co/Xenova/clip-vit-base-patch32/resolve/main/tokenizer.json",
    tokenizer_file: "clip_tokenizer.json",
    output: "text_embeds",
    pad_to: Some(77),
    center: true,
};

pub const CLAP: TextTower = TextTower {
    name: "CLAP text",
    model_url:
        "https://huggingface.co/Xenova/larger_clap_general/resolve/main/onnx/text_model_quantized.onnx",
    model_file: "clap_text.onnx",
    model_bytes: 126_603_263,
    tokenizer_url: "https://huggingface.co/Xenova/larger_clap_general/resolve/main/tokenizer.json",
    tokenizer_file: "clap_tokenizer.json",
    output: "text_embeds",
    pad_to: None,
    center: false,
};

fn cache_key(tower: &TextTower, labels: &[String]) -> PathBuf {
    let mut h = Sha256::new();
    h.update(tower.model_file.as_bytes());
    for l in labels {
        h.update(l.as_bytes());
        h.update([0]);
    }
    let hex = format!("{:x}", h.finalize());
    crate::clap_tag::cache_dir().join(format!("labels-{}.bin", &hex[..16]))
}

/// Embed `labels` with `tower`, caching the result. Returns [n][512] row-major,
/// unit-norm, in the same space as the vendored defaults.
pub fn embed(tower: &TextTower, labels: &[String], dim: usize) -> Result<Vec<f32>, String> {
    let cache = cache_key(tower, labels);
    if let Ok(b) = std::fs::read(&cache) {
        if b.len() == labels.len() * dim * 4 {
            return Ok(from_le(&b));
        }
    }
    let dir = crate::clap_tag::cache_dir();
    let model = dir.join(tower.model_file);
    let tokj = dir.join(tower.tokenizer_file);
    crate::clap_tag::download_model(&model, tower.model_url, tower.model_bytes, tower.name)?;
    crate::clap_tag::download_model(&tokj, tower.tokenizer_url, 1_000_000, "tokenizer")?;

    let tk = tokenizers::Tokenizer::from_file(&tokj).map_err(|e| format!("tokenizer: {e}"))?;
    let encs = tk
        .encode_batch(labels.to_vec(), true)
        .map_err(|e| format!("tokenize: {e}"))?;
    let width = match tower.pad_to {
        Some(n) => n,
        None => encs.iter().map(|e| e.get_ids().len()).max().unwrap_or(1),
    };
    let mut ids = vec![0i64; labels.len() * width];
    for (r, e) in encs.iter().enumerate() {
        for (c, v) in e.get_ids().iter().take(width).enumerate() {
            ids[r * width + c] = *v as i64;
        }
    }
    let mut session = ort::session::Session::builder()
        .map_err(|e| format!("ort builder: {e}"))?
        .commit_from_file(&model)
        .map_err(|e| format!("load text onnx: {e}"))?;
    let input = ort::value::Tensor::from_array((vec![labels.len(), width], ids))
        .map_err(|e| format!("tensor: {e}"))?;
    let out = session
        .run(ort::inputs!["input_ids" => input])
        .map_err(|e| format!("text infer: {e}"))?;
    let raw = out[tower.output]
        .try_extract_tensor::<f32>()
        .map_err(|e| format!("extract: {e}"))?
        .1
        .to_vec();
    if raw.len() != labels.len() * dim {
        return Err(format!(
            "text tower returned {} floats, expected {}",
            raw.len(),
            labels.len() * dim
        ));
    }
    let mut feats = raw;
    normalize_rows(&mut feats, dim);
    if tower.center {
        let n = feats.len() / dim;
        for c in 0..dim {
            let mean: f32 = (0..n).map(|r| feats[r * dim + c]).sum::<f32>() / n as f32;
            for r in 0..n {
                feats[r * dim + c] -= mean;
            }
        }
        normalize_rows(&mut feats, dim);
    }
    let _ = std::fs::create_dir_all(&dir);
    let bytes: Vec<u8> = feats.iter().flat_map(|f| f.to_le_bytes()).collect();
    let _ = std::fs::write(&cache, &bytes);
    Ok(feats)
}

fn normalize_rows(v: &mut [f32], dim: usize) {
    for row in v.chunks_mut(dim) {
        let n: f32 = row.iter().map(|x| x * x).sum::<f32>().sqrt();
        if n > 0.0 {
            for x in row.iter_mut() {
                *x /= n;
            }
        }
    }
}

fn from_le(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// `--labels "a,b,c"` or `--labels-file path` (one per line, `#` comments ok).
pub fn parse(inline: Option<&str>, file: Option<&str>) -> Result<Option<Vec<String>>, String> {
    let raw = match (inline, file) {
        (Some(_), Some(_)) => return Err("use --labels or --labels-file, not both".into()),
        (Some(s), None) => s.split(',').map(|s| s.to_string()).collect::<Vec<_>>(),
        (None, Some(f)) => std::fs::read_to_string(f)
            .map_err(|e| format!("--labels-file {f}: {e}"))?
            .lines()
            .map(|l| l.split('#').next().unwrap_or("").to_string())
            .collect(),
        (None, None) => return Ok(None),
    };
    let out: Vec<String> = raw
        .into_iter()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();
    match out.len() {
        0 => Err("label list is empty".into()),
        1 => Err("need at least 2 labels to compare against".into()),
        _ => Ok(Some(out)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_inline_and_file() {
        assert!(parse(None, None).unwrap().is_none());
        let v = parse(Some("cat, dog , bird"), None).unwrap().unwrap();
        assert_eq!(v, vec!["cat", "dog", "bird"]);
        let p = std::env::temp_dir().join("ross-labels-test.txt");
        std::fs::write(&p, "sword\n# a comment\n\nshield   \npotion # trailing\n").unwrap();
        let v = parse(None, Some(p.to_str().unwrap())).unwrap().unwrap();
        assert_eq!(v, vec!["sword", "shield", "potion"]);
    }

    #[test]
    fn rejects_unusable_lists() {
        assert!(parse(Some("only-one"), None).is_err());
        assert!(parse(Some(" , , "), None).is_err());
        assert!(parse(Some("a,b"), Some("f")).is_err());
    }

    #[test]
    fn cache_key_tracks_label_content_and_order() {
        let a = vec!["x".to_string(), "y".to_string()];
        let b = vec!["y".to_string(), "x".to_string()];
        assert_ne!(cache_key(&CLIP, &a), cache_key(&CLIP, &b), "order must matter");
        assert_eq!(cache_key(&CLIP, &a), cache_key(&CLIP, &a));
        assert_ne!(cache_key(&CLIP, &a), cache_key(&CLAP, &a), "tower must matter");
    }

    #[test]
    fn normalize_rows_makes_unit_vectors() {
        let mut v = vec![3.0f32, 4.0, 0.0, 5.0];
        normalize_rows(&mut v, 2);
        assert!((v[0] - 0.6).abs() < 1e-6 && (v[1] - 0.8).abs() < 1e-6);
        assert!((v[3] - 1.0).abs() < 1e-6);
    }
}
