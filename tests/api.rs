//! Consumes ross the way an embedding caller does — only through the public
//! library surface, never the binary. If this stops compiling, the API broke.

use std::path::Path;

/// The deriver shape: load a tagger once, reuse it per file.
#[test]
fn image_tagging_through_the_public_api() {
    if !ross::clip_tag::model_path().exists() {
        eprintln!("skipping: CLIP model not downloaded");
        return;
    }
    let tagger = ross::clip_tag::ClipTagger::load(None).expect("load");

    let dir = std::env::temp_dir().join(format!("ross-api-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let png = dir.join("solid.png");
    // a plain generated image: we assert on the contract, not on which label wins
    let img = image::RgbImage::from_fn(320, 240, |x, y| {
        image::Rgb([(x % 256) as u8, (y % 256) as u8, 128])
    });
    img.save(&png).unwrap();

    match tagger.tag_path(&png).expect("tag_path") {
        ross::clip_tag::TagResult::Tagged(tags, conf) => {
            assert!(!tags.is_empty() && tags.len() <= 3);
            assert!((0.0..=1.0).contains(&conf));
        }
        ross::clip_tag::TagResult::Gated => {}
    }

    // in-memory bytes must work too — callers often already hold the image
    let bytes = std::fs::read(&png).unwrap();
    tagger.tag_bytes(&bytes).expect("tag_bytes");

    // and a non-image must be a clean Err, not a panic
    let junk = dir.join("junk.bin");
    std::fs::write(&junk, b"not an image").unwrap();
    assert!(tagger.tag_path(&junk).is_err());
}

/// A caller supplying its own taxonomy is the main reason to link rather than
/// shell out, so it has to be reachable without the CLI.
#[test]
fn custom_vocabulary_through_the_public_api() {
    let parsed = ross::labels::parse(Some("a cat,a dog,a bird"), None).expect("parse");
    assert_eq!(parsed.unwrap().len(), 3);
    assert!(ross::labels::parse(Some("just-one"), None).is_err());
}

/// Sniffing and metadata are useful on their own, with no model loaded and no
/// external binary present.
#[test]
fn metadata_helpers_need_no_model() {
    use ross::media::{sniff_bytes, Kind};
    assert!(matches!(
        sniff_bytes(&[0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A]),
        Some((Kind::Image, "image/png"))
    ));
    assert!(sniff_bytes(b"not media at all").is_none());

    let dir = std::env::temp_dir().join(format!("ross-api-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let p = dir.join("h.png");
    image::RgbImage::new(7, 11).save(&p).unwrap();
    let (meta, text) = ross::media::exif_metadata(&p).expect("exif_metadata");
    assert_eq!(meta["width"], 7, "native header parse should work without exiftool");
    assert_eq!(meta["height"], 11);
    assert!(!text.is_empty());
    assert_eq!(ross::media::sha256_hex(&p).unwrap().len(), 64);
    assert!(ross::media::sniff(Path::new(&p)).is_some());
}

/// Rendering is reusable too, so a caller can emit the same shapes the CLI does.
#[test]
fn output_rendering_is_reusable() {
    let v = serde_json::json!({"path": "/a.png", "type": "image", "tags": ["icon"]});
    assert!(ross::output::json_out(std::slice::from_ref(&v), true).contains("\"icon\""));
    assert!(ross::output::md_out(std::slice::from_ref(&v)).starts_with("## /a.png"));
    assert!(ross::output::text_out(&[v]).contains("== /a.png =="));
}

/// Compressed size does not predict decoded size: a small palette PNG can be
/// hundreds of megapixels. Unbounded, decoding it multiplies by the worker count
/// and exhausts memory, so oversized images are rejected from the header.
#[test]
fn oversized_images_are_rejected_from_the_header() {
    if !ross::clip_tag::model_path().exists() {
        eprintln!("skipping: CLIP model not downloaded");
        return;
    }
    let dir = std::env::temp_dir().join(format!("ross-big-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let tagger = ross::clip_tag::ClipTagger::load(None).expect("load");

    // a modest image must be unaffected
    let small = dir.join("small.png");
    image::RgbImage::new(64, 64).save(&small).unwrap();
    assert!(tagger.tag_path(&small).is_ok());

    // 20000x20000 = 400 MP, far over the default cap; a 1x1 grey PNG scaled up on
    // paper only — we write the header-sized image via a cheap luma buffer
    std::env::set_var("ROSS_MAX_PIXELS", "1000");
    let over = dir.join("over.png");
    image::GrayImage::new(64, 64).save(&over).unwrap(); // 4096 px > 1000 cap
    let e = tagger.tag_path(&over).unwrap_err();
    assert!(e.contains("decode cap") && e.contains("ROSS_MAX_PIXELS"),
            "error should name the cap and the escape hatch: {e}");
    std::env::remove_var("ROSS_MAX_PIXELS");
}
