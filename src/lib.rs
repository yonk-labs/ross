//! ross as a library: the pieces the CLI is built from, usable in-process.
//!
//! The binary wires these together into a batch pipeline; a caller embedding
//! ross usually wants one deriver, not the pipeline. Everything here is
//! independent — you can tag images without ever touching the audio path, and
//! neither requires an LLM endpoint.
//!
//! ```no_run
//! // one-time model load (fetches ~89 MB into ~/.cache/ross-clap on first use),
//! // then reuse the tagger across files — building the ONNX session is the
//! // expensive part, not inference.
//! let tagger = ross::clip_tag::ClipTagger::load(None)?;
//! match tagger.tag_path(std::path::Path::new("art.png"))? {
//!     ross::clip_tag::TagResult::Tagged(tags, confidence) => {
//!         println!("{tags:?} @ {confidence:.3}");
//!     }
//!     // nothing cleared the confidence bar: the caller decides what to do
//!     // instead, rather than being handed a guess
//!     ross::clip_tag::TagResult::Gated => println!("no confident match"),
//! }
//! # Ok::<(), String>(())
//! ```
//!
//! Pass `Some(vec![...])` to `load` to swap the built-in vocabulary for your
//! own; the label embeddings are computed once and cached on disk.
//!
//! # Threading
//!
//! `ClipTagger` and `ClapTagger` hold a `Mutex<Session>` because `ort` sessions
//! need `&mut` to run, so concurrent calls serialize on inference. Sharing one
//! tagger across threads is safe and correct, but does not parallelize; ONNX
//! also keeps process-global state, which is worth knowing before you decide
//! between linking this and shelling out to the binary.

pub mod clap_tag;
pub mod clip_tag;
pub mod labels;
pub mod media;
pub mod output;
pub mod semantic;
