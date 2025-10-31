from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List

from .layout import process_pdf, NO_PROXY_FOR_LLM
from .common import LayoutConfig, MAX_LLM_SIDE, DEFAULT_JPEG_QUALITY


def main():
    parser = argparse.ArgumentParser(description='LLM layout (figures + text_blocks) → JSONL + overlays')
    parser.add_argument('inputs', nargs='*')
    parser.add_argument('--out', default='out_interleave_llm')
    parser.add_argument('--zoom', type=float, default=2.0)
    parser.add_argument('--include-figures', action='store_true')
    parser.add_argument('--viz', action='store_true')
    parser.add_argument('--viz-text-blocks', action='store_true')
    parser.add_argument('--viz-absorb-debug', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--pages', default=None)
    parser.add_argument('--strict-figure-capture', action='store_true')
    parser.add_argument('--log-content', action='store_true')
    parser.add_argument('--no-small-label-absorb', action='store_true')
    parser.add_argument('--llm-image-format', choices=['jpeg','png'], default='jpeg')
    parser.add_argument('--max-llm-side', type=int, default=MAX_LLM_SIDE)
    parser.add_argument('--jpeg-quality', type=int, default=DEFAULT_JPEG_QUALITY)
    parser.add_argument('--jobs', type=int, default=1)
    parser.add_argument('--fig-nms-iou', type=float, default=0.0)
    parser.add_argument('--ambiguous-gap-frac', type=float, default=0.01)
    parser.add_argument('--link-min-overlap-ratio', type=float, default=0.15)
    parser.add_argument('--link-min-iou', type=float, default=0.02)
    parser.add_argument('--order-inversion-ratio-thr', type=float, default=0.30)
    parser.add_argument('--absorb-marg-x', type=float, default=0.03)
    parser.add_argument('--absorb-marg-y', type=float, default=0.05)
    parser.add_argument('--absorb-final-expand', type=float, default=0.015)
    parser.add_argument('--small-label-w-ratio', type=float, default=0.30)
    parser.add_argument('--small-label-w-ratio-strict', type=float, default=0.20)
    parser.add_argument('--small-label-h-ratio', type=float, default=0.022)
    parser.add_argument('--small-label-h-ratio-strict', type=float, default=0.018)
    parser.add_argument('--min-figure-area-ratio', type=float, default=0.0025)
    parser.add_argument('--no-respect-llm-absorb-hints', action='store_true')
    parser.add_argument('--llm-timeout', type=int, default=600)
    parser.add_argument('--llm-temperature', type=float, default=0.0)
    parser.add_argument('--llm-model', default=None)
    parser.add_argument('--llm-allow-proxy', action='store_true')
    parser.add_argument('--llm-seed', type=int, default=None)
    parser.add_argument('--llm-only', action='store_true')
    parser.add_argument('--repair-math-macros', choices=['off','moderate','aggressive'], default='moderate',
                        help='Repair missing backslashes for LaTeX macros inside math ($$...$$/$...$) using a whitelist')
    parser.add_argument('--replay-from-logs', action='store_true', help='Replay parsing/normalization from existing I_content_page_XXX.txt logs instead of calling LLM')

    args=parser.parse_args()

    if getattr(args, 'verbose', False):
        print('[info] OpenAI-compatible Chat Completions backend; requires response_format={"type":"json_object"} and image_url.', file=sys.stderr)

    base_url = os.environ.get('OPENAI_BASE_URL','https://ark.cn-beijing.volces.com/api/v3')
    api_key = os.environ.get('OPENAI_API_KEY')
    model = args.llm_model or os.environ.get('OPENAI_MODEL','doubao-seed-1-6-vision-250815')
    if not api_key and not bool(args.replay_from_logs):
        print('ERROR: OPENAI_API_KEY not set', file=sys.stderr); return

    # update proxy behavior
    import unify_llm_interleave.layout as lay
    lay.NO_PROXY_FOR_LLM = not bool(args.llm_allow_proxy)

    out_root = Path(args.out); out_root.mkdir(exist_ok=True)
    pdfs: List[Path]=[]
    if not args.inputs:
        print('No inputs.'); return
    for ip in args.inputs:
        p=Path(ip)
        if p.is_file() and p.suffix.lower()=='.pdf': pdfs.append(p)
        elif p.is_dir(): pdfs.extend([q for q in p.rglob('*.pdf')])
    if not pdfs:
        print('No PDF found.'); return

    cfg = LayoutConfig(
        absorb_marg_x=float(args.absorb_marg_x),
        absorb_marg_y=float(args.absorb_marg_y),
        absorb_final_expand=float(args.absorb_final_expand),
        small_label_w_ratio=float(args.small_label_w_ratio),
        small_label_w_ratio_strict=float(args.small_label_w_ratio_strict),
        small_label_h_ratio=float(args.small_label_h_ratio),
        small_label_h_ratio_strict=float(args.small_label_h_ratio_strict),
        min_figure_area_ratio=float(args.min_figure_area_ratio),
        ambiguous_gap_frac=float(args.ambiguous_gap_frac),
        link_min_overlap_ratio=float(args.link_min_overlap_ratio),
        link_min_iou=float(args.link_min_iou),
        order_inversion_ratio_thr=float(args.order_inversion_ratio_thr),
        macro_repair_mode=str(args.repair_math_macros),
    )

    for pdf in pdfs:
        doc_out = out_root / pdf.stem
        inter, layout = process_pdf(
            pdf, doc_out,
            zoom=args.zoom,
            include_figures=args.include_figures,
            viz=bool(args.viz),
            viz_text_blocks=bool(args.viz_text_blocks),
            base_url=base_url,
            api_key=api_key,
            model=model,
            temperature=args.llm_temperature,
            timeout=args.llm_timeout,
            strict_capture=bool(args.strict_figure_capture),
            image_format=str(args.llm_image_format),
            no_small_label_absorb=bool(args.no_small_label_absorb),
            respect_llm_absorb_hints=(not bool(args.no_respect_llm_absorb_hints)),
            viz_absorb_debug=bool(args.viz_absorb_debug),
            pages_expr=args.pages,
            jobs=int(args.jobs),
            max_llm_side=int(args.max_llm_side),
            jpeg_quality=int(args.jpeg_quality),
            config=cfg,
            fig_nms_iou=(0.0 if args.llm_only else float(args.fig_nms_iou)),
            llm_only=bool(args.llm_only),
            log_content=bool(args.log_content),
            llm_seed=args.llm_seed,
            replay_from_logs=bool(args.replay_from_logs)
        )
        print(f'Processed {pdf.name} → {inter} ; layout → {layout}')


if __name__=='__main__':
    main()
