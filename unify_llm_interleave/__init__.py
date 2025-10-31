"""
unify_llm_interleave package

Exports a stable surface compatible with the original single-file module,
while organizing code into submodules:

- common: pure helpers, JSON/LaTeX utilities, geometry helpers, config
- layout: LLM call + PDF processing pipeline
- markdown: Markdown generation from layout JSONL
"""

from .common import (
    LayoutConfig,
    MAX_LLM_SIDE,
    DEFAULT_JPEG_QUALITY,
    _extract_json_obj,
    _normalize_latex_backslashes,
    _purge_ctrl,
    _scale_kilo_to_px,
    _parse_pages_expr,
    _iou,
    _score_tb_to_fig,
    get_last_extract_meta,
)

from .layout import (
    NO_PROXY_FOR_LLM,
    llm_call,
    process_pdf,
    _absorb_into_figures,
)

__all__ = [
    'LayoutConfig', 'MAX_LLM_SIDE', 'DEFAULT_JPEG_QUALITY',
    '_extract_json_obj', '_normalize_latex_backslashes', '_purge_ctrl',
    '_scale_kilo_to_px', '_parse_pages_expr', '_iou', '_score_tb_to_fig', 'get_last_extract_meta',
    'NO_PROXY_FOR_LLM', 'llm_call', 'process_pdf', '_absorb_into_figures'
]
