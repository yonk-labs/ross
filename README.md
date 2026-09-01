# ross

Media in → metadata, tags, description, summary out. Single Rust binary, no daemon, no python — at runtime or setup.

```bash
ross photo.jpg                     # text output on a terminal
ross ~/media/ --json > out.json    # batch, JSON when piped
ross clip.mp4 --md                 # markdown report
ross sfx/ --clap --no-llm          # offline: local tags + descriptions, no network
ross art/ --clip --labels-file my-taxonomy.txt   # tag against your own vocabulary
```

## What it does

Per file (concurrent across files, default min(8, cpus)):

1. **Sniff** type by magic bytes (not extension) — image / audio / video; non-media skipped.
2. **Extract** deterministic facts: ffprobe (audio/video), exiftool (images; PNG/WebP header fallback when absent), sha256, size, mtime.
3. **Describe**, by either or both of:
   - `--clap` (audio) / `--clip` (images, video) — zero-shot tags from locally-run
     CLAP/CLIP models. Fully offline, and the tag vocabulary is yours to replace.
   - LLM pass — one call per file to any OpenAI-compatible endpoint. Omit with `--no-llm`.

Merge rule: the local tagger fills first, the LLM overwrites on success; if the LLM
fails after the local pass succeeded, the local result is kept and `llm_error` is set.
With `--no-llm --clap --clip` every file gets tags and a description with no network
at all.

## CLI

```
ross [PATHS]...
  --format json|md|text   default: text on tty, JSON when piped
  --json / --md / --text  shorthands
  --no-llm                deterministic fields only (fast, offline)
  --no-vision             never inline images/frames; model gets text only
  --audio                 send raw audio to an audio-capable model (transcodes to mp3)
  --clap                  local CLAP audio tags (see below)
  --clip                  local CLIP image/video tags (see below)
  --labels "A,B,C"        replace the tag vocabulary for --clap/--clip
  --labels-file FILE      same, one per line (# comments allowed)
  --ask "PROMPT" | --ask-file F
  --frames N              video frames sampled for the model (default 4)
  --concurrency N         worker threads (default min(8, cpus))
  --strict                exit 2 if any file errored (results are still printed)
  --quiet                 no per-file progress on stderr
  --doctor                check external binaries + endpoint config
  --url / --model                    global endpoint + model
  --vision-url / --vision-model      per-modality overrides
  --audio-url  / --audio-model
  --video-url  / --video-model
```

## Endpoints

Any OpenAI chat-completions base URL: OpenAI, vLLM `/v1`, Ollama `/v1`, LM Studio, etc.

One endpoint for everything:

```bash
export ROSS_URL=http://192.168.1.133:8000/v1
export ROSS_MODEL=qwen36-nvfp4
ross ~/media/ --json
```

Or a different server and model per modality — useful when vision, video and
audio-to-text live on different boxes:

```bash
export ROSS_VISION_URL=http://gpu1:8000/v1   ROSS_VISION_MODEL=qwen2.5-vl
export ROSS_AUDIO_URL=http://gpu2:8000/v1    ROSS_AUDIO_MODEL=qwen2-audio
export ROSS_VIDEO_URL=http://gpu1:8000/v1    ROSS_VIDEO_MODEL=qwen2.5-vl
export ROSS_AUDIO_API_KEY=sk-...             # optional, per modality
ross ~/media/ --json
```

Resolution runs most-specific-first, so any subset can be set:

```
--vision-url  >  ROSS_VISION_URL  >  --url  >  ROSS_URL
```

`--doctor` prints exactly what each modality resolved to. A modality with no
endpoint still gets deterministic fields (and CLAP tags, for audio) plus an
`llm_error` noting it was unset.

| Variable | Default | Meaning |
|---|---|---|
| `ROSS_URL` / `ROSS_MODEL` | — | global endpoint + model |
| `ROSS_{VISION,AUDIO,VIDEO}_URL` / `_MODEL` | falls back to global | per-modality override |
| `ROSS_API_KEY`, `ROSS_{MOD}_API_KEY` | none | optional bearer (never stored) |
| `ROSS_MAX_TOKENS`, `ROSS_{MOD}_MAX_TOKENS` | 4096 | reasoning models burn budget thinking; falls back to `message.reasoning` when `content` is empty |
| `ROSS_TIMEOUT_S` | 120 | per-request timeout |
| `ROSS_MAX_INLINE_MB` | 24 | refuse to inline media larger than this |

Failed requests retry up to 3 times with a short backoff. A 4xx other than 429 is
not retried — it will not fix itself.

## Local tagging (`--clap`, `--clip`)

Zero-shot classification against a label vocabulary, running ONNX models natively
in-process. This is the description path when no LLM is configured.

**Setup: none.** Models are fetched from Hugging Face on first use and cached in
`~/.cache/ross-clap/`. No python, no export step, no daemon.

| flag | covers | model | download |
|---|---|---|---|
| `--clap` | audio | `larger_clap_general` audio tower | 78 MB (int8) |
| `--clip` | images, video (one frame) | `mobileclip_s0` vision tower | 46 MB (fp32) |

```bash
ross sfx/ --clap --no-llm --json
ross art/ --clip --no-llm --json
```

### Your own vocabulary

The built-in lists (`src/clap_labels.txt`, `src/clip_labels.txt`) are a starting
point, not a limit. Supply your own and the matching text tower is downloaded once
to embed them; the result is cached by a hash of the exact list, so only the first
run pays for it.

```bash
ross art/  --clip --labels "a wrestler,a spaceship,a logo,a city street"
ross sfx/  --clap --labels-file audio-taxonomy.txt
```

Phrases work better than bare words — `"a door opening"` beats `"door"`. The text
towers are 170 MB (CLIP) and 127 MB (CLAP), downloaded only when you use custom
labels, and never again for the same list.

### Confidence

A label is emitted only if it scores high outright, or scores lower while standing
clearly apart from the runner-up. Anything else gets **no tags** and falls back to a
filename-derived description rather than confident nonsense. Tagged files carry
`tag_confidence`.

Absolute scores depend on how many labels compete, so a custom list of 5 scores
lower across the board than the built-in 50 — the margin is what carries the
decision. If your vocabulary is over- or under-eager, tune it:

| Variable | Default (clap / clip) | Meaning |
|---|---|---|
| `ROSS_{CLAP,CLIP}_MIN_SCORE` | 0.30 / 0.055 | accept on absolute score alone |
| `ROSS_{CLAP,CLIP}_MIN_SCORE_DISTINCT` | 0.15 / 0.030 | floor for the margin path |
| `ROSS_{CLAP,CLIP}_MIN_MARGIN` | 0.08 / 0.012 | required lead over the runner-up |
| `ROSS_CLAP_SCORES`, `ROSS_CLIP_SCORES` | unset | print every label's score to stderr |

### Notes

- Audio under 0.35s is too brief for a usable spectrogram and is gated on length
  alone (`ROSS_CLAP_MIN_SECONDS`). Over 10s, a deterministic centre crop is used.
- **The CLIP model is fp32 on purpose.** Its int8 export is 12 MB instead of 46 MB,
  but scores 27% on a 4-way task where chance is 25%, and agrees with its own fp32
  weights on 0 of 16 images — quantization destroys this model. The audio model
  survives int8 fine; `ROSS_CLAP_PRECISION=fp32` trades a 281 MB download for a
  0.3s rather than 1.3s startup.
- Startup is ~1.3s (CLAP) / ~0.4s (CLIP) to build the ONNX session, then ~110ms and
  ~35ms per file. Batch runs amortize it.
- Because `--clap`/`--clip` were asked for explicitly, ross exits 5 rather than
  silently continuing without them.
- Cache location: `ROSS_CLAP_CACHE`.

Editing `src/*_labels.txt` requires regenerating the matching
`src/*_text_feats.bin` against the same text tower — or just use `--labels`, which
does it at runtime.

## External binaries

| Binary | Used for | Required |
|---|---|---|
| ffprobe | audio/video metadata | yes for a/v files |
| ffmpeg | video frames, audio transcode | yes for video+LLM, `--audio` |
| exiftool | image metadata | no — PNG/WebP dims parsed natively |

## Exit codes

`0` ok (per-file errors don't fail the batch) · `2` `--strict` and at least one
file errored · `3` no endpoint configured · `4` bad input or usage · `5`
`--clap`/`--clip` requested but unavailable

## Non-goals

Storage/database, dedup, embeddings search, chat, templates, plugins, thumbnails output, watching.

## Development

```bash
cargo test                    # 32 unit + 7 integration
ROSS_TEST_STRICT=1 cargo test # fail instead of skipping when a dep is missing
cargo build --release
```

Integration tests generate fixtures with ffmpeg, run the real binary, and assert
exit codes, JSON shape, `--strict` result-preservation, per-modality endpoint
resolution, CLAP/CLIP gating, custom-label validation, and clean exit on a
closed pipe.

Layout: `main.rs` (CLI/walk/thread pool/exit codes) · `media.rs` (sniff, sha256,
ffprobe/exif, frames) · `semantic.rs` (endpoint resolution + chat-completions +
JSON extraction) · `clap_tag.rs` (native CLAP: symphonia decode → rubato resample →
realfft STFT → native Slaney mel → ort inference) · `clip_tag.rs` (native CLIP:
decode → resize/crop → ort inference) · `labels.rs` (runtime label embedding +
cache) · `output.rs` (json/md/text).

## License

Apache 2.0 — see [LICENSE](LICENSE) and [NOTICE](NOTICE).

The CLAP and CLIP model weights are **not** distributed with this source; they are
downloaded from Hugging Face on first use of `--clap` / `--clip` and carry their own
upstream licenses. The vendored label embeddings (`src/*_text_feats.bin`) are derived
from those models' text towers. NOTICE lists both.
