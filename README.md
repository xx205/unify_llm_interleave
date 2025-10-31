
# unify-llm-interleave

LLM‑driven PDF layout extraction (figures + text blocks) and Markdown interleaving. Turn complex scientific PDFs into clean, column‑aware Markdown with images and captions in the right order.

- `unify_llm_interleave/` package
  - `layout.py`: page→LLM layout (figures + text_blocks) → optional absorption/linking/NMS → JSONL + visualizations.
  - `markdown.py`: converts the JSONL into Markdown with image–text interleaving.
  - `common.py`: shared helpers (JSON/LaTeX normalization, geometry, config).
  - CLI entry points: `unify-llm-layout`, `unify-llm-markdown`.

## Backend Support

- OpenAI-compatible Chat Completions only.
- Must support: `response_format={"type":"json_object"}` and vision `image_url` data URI.
- Example model names:
  - `doubao-seed-1-6-vision-250815` (tested via an OpenAI-compatible endpoint)

> Tip: Set `OPENAI_BASE_URL` to your provider's endpoint. The scripts do not use the OpenAI SDK directly; they send HTTP requests to a compatible `/chat/completions` API.

## Install

- Python >= 3.9
- Dependencies

```bash
pip install requests pymupdf
# optional visualization overlays
pip install opencv-python-headless Pillow numpy
```

## Environment

- `OPENAI_API_KEY` (required)
- `OPENAI_BASE_URL` (default: `https://ark.cn-beijing.volces.com/api/v3`)
- `OPENAI_MODEL` (optional; default: `doubao-seed-1-6-vision-250815`, can be overridden by `--llm-model`)

Examples (bash)

```bash
export OPENAI_API_KEY=...
export OPENAI_BASE_URL=https://ark.cn-beijing.volces.com/api/v3
export OPENAI_MODEL=doubao-seed-1-6-vision-250815
```

## Quick Start

1) Layout + JSONL (with overlays)

```bash
python3 -m unify_llm_interleave.cli_layout your.pdf --viz --include-figures --jobs 4
```

2) Markdown generation (column-aware by default)

```bash
python3 -m unify_llm_interleave.cli_markdown your.pdf
# To disable column detection:
# python3 -m unify_llm_interleave.cli_markdown your.pdf --no-md-col-detect
```

3) LLM-only (skip linking/absorption/NMS/order-fallback)

```bash
unify-llm-layout your.pdf --llm-only
unify-llm-markdown your.pdf --llm-only
```

## Selected CLI (layout)

- `--jobs`: page-level concurrency
- `--llm-image-format` jpeg|png (default jpeg)
- `--max-llm-side` / `--jpeg-quality`: single-pass raster and encoding control
- `--strict-figure-capture`: tighter caption/label absorption conditions
- `--fig-nms-iou`: figure NMS IoU (0 to disable)
- `--ambiguous-gap-frac`: fraction of page diagonal for unique-assignment ambiguity (default 0.01)
- `--link-min-overlap-ratio` / `--link-min-iou`: caption→figure gate thresholds
- `--order-inversion-ratio-thr`: trigger ratio for reading-order fallback
- `--llm-only`: trust LLM output only (no geometry postprocess)
- `--log-content`: persist raw LLM content (privacy risk)

Outputs (under `out_interleave_llm/<doc_stem>/`)

- `structured_layout.jsonl`: per-page records
  - `figures`, `text_blocks`, `page_size`, `engine`, `violations` {`missing_caption_ref`, `equation_overlap`}
- `interleaved.jsonl`: legacy-like interleaving sequence
- `page_XXX_layout_llm.png`: overlays
- `page_XXX_absorb_debug.png`: absorption decisions (if enabled)

## Selected CLI (markdown)

- `--md-col-detect` / `--no-md-col-detect` (default: enabled)
- `--md-col-gap-frac`: column split gap (x-fraction of page width; default 0.08)
- `--md-max-cols`: at most columns to detect (default 3)
- `--md-col-min-blocks`: min text blocks to attempt column detection (default 6)
- `--repair-math-macros`: whitelist-based repair of missing backslashes inside math (`off|moderate|aggressive`, default `moderate`). `aggressive` 还会包含希腊字母、更多宏名。

Column detection uses x-center clustering with a gap threshold (`--md-col-gap-frac`) and reorders blocks column-wise (left→right) then top→bottom. Tune `--md-col-min-blocks`, `--md-col-gap-frac`, and `--md-max-cols` as needed.

## Notes

- Vision input uses lossy JPEG by default for the LLM; visualizations/crops are lossless PNG.
- The scripts do not mutate your proxy env; they use a thread-local HTTP session and can bypass proxies for LLM requests when desired.
- Reading-order fallback: if the pairwise inversion ratio of text blocks (by y0,x0) exceeds `--order-inversion-ratio-thr`, the page is resorted by (y0,x0) for more stable reading order. A log `order_fallback_page_XXX.txt` is emitted.
- For two-column pages where the left column begins with a large figure (few text blocks at the top-left), consider reducing `--md-col-min-blocks` for more stable column detection.
- Backslash handling: the parser tolerates and repairs common issues instead of relying on prompt conventions. It (1) auto-repairs illegal JSON escapes (turns `\x` that are not valid JSON escapes into `\\x` so JSON parses); (2) in math segments, collapses over-produced backslashes and restores macros accidentally broken by JSON control characters (e.g., `\t` → TAB making `text` become `\text`); and (3) normalizes residual control characters (TAB→space, CR removed) after macro repair. All changes are logged per page for auditing.

## Migration note

This repo used to expose two standalone scripts (`unify_llm_interleave.py`, `unify_llm_interleave_md.py`).
They are now organized into a package with console scripts. Existing Python imports should switch to
`import unify_llm_interleave as mod` (unchanged name) and refer to helpers such as `mod._extract_json_obj`.

## License

MIT License (see `LICENSE`).

## Troubleshooting

- Empty or invalid JSON from the LLM
  - Ensure your backend supports `response_format={"type":"json_object"}`.
  - Try `--replay-from-logs` to debug parsing/normalization without incurring API calls.
- Captions mis‑linked to figures
  - Lower `--link-min-overlap-ratio` or `--link-min-iou`.
  - Increase `--absorb-marg-x`/`--absorb-marg-y` or enable `--strict-figure-capture`.
- Two‑column PDFs mis‑ordered
  - Enable column detection (`--md-col-detect`, default on) and tune `--md-col-gap-frac`.
  - When inversion ratio is high, the reading‑order fallback sorts by (y0, x0). Adjust `--order-inversion-ratio-thr`.

## Roadmap

- Built‑in providers module (OpenAI SDK, DashScope, etc.).
- Better math normalization coverage (tables/arrays).
- Optional HTML export with figure anchors.
