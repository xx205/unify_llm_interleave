from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Iterator, Optional, List, Dict, Any
import concurrent.futures as _fut

try:
    import fitz  # PyMuPDF
    _HAS_FITZ = True
except ImportError:
    fitz = None
    _HAS_FITZ = False

try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False

from .pipeline import PageContext, PipelineStage
from .common import LayoutConfig, MAX_LLM_SIDE, DEFAULT_JPEG_QUALITY, _parse_pages_expr, _pix_to_encoded, _bytes_to_data_url, _iou
from .llm_client import llm_call
from .refinement import link_captions, demote_equation_only_figures, nms_figures, absorb_into_figures, apply_reading_order_fallback

class PdfSource:
    def __init__(self, pdf_path: Path, zoom: float = 2.0, pages_expr: Optional[str] = None, max_llm_side: int = MAX_LLM_SIDE):
        if not _HAS_FITZ:
            raise RuntimeError("PyMuPDF (fitz) is required for PdfSource")
        self.pdf_path = pdf_path
        self.zoom = zoom
        self.pages_expr = pages_expr
        self.max_llm_side = max_llm_side

    def __iter__(self) -> Iterator[PageContext]:
        doc = fitz.open(self.pdf_path.as_posix())
        try:
            page_indices = _parse_pages_expr(self.pages_expr, doc.page_count)
            for page_idx in page_indices:
                page = doc[page_idx]
                pw, ph = page.rect.width, page.rect.height
                base_side = max(pw, ph)
                side_cap = float(self.max_llm_side)
                scale = min(self.zoom, max(0.5, side_cap / float(base_side)))
                pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
                
                # Convert to numpy for OpenCV if needed, or keep as bytes for LLM
                # For PageContext, we store the raw pixmap or convert to image as needed.
                # Here we'll store basic info.
                
                ctx = PageContext(
                    doc_id=self.pdf_path.name,
                    page_index=page_idx,
                    width=pix.width,
                    height=pix.height,
                    image_path=None # Will be set if saved
                )
                # Attach pixmap to metadata for the Engine to use (avoiding premature save)
                ctx.metadata['pixmap'] = pix
                yield ctx
        finally:
            doc.close()

class LlmLayoutEngine(PipelineStage):
    def __init__(self, api_key: str, base_url: str, model: str, 
                 out_dir: Path,
                 temperature: float = 0.0, timeout: int = 600,
                 image_format: str = 'jpeg', jpeg_quality: int = DEFAULT_JPEG_QUALITY,
                 strict_capture: bool = False,
                 llm_only: bool = False,
                 seed: Optional[int] = None,
                 config: Optional[LayoutConfig] = None):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.out_dir = out_dir
        self.temperature = temperature
        self.timeout = timeout
        self.image_format = image_format
        self.jpeg_quality = jpeg_quality
        self.strict_capture = strict_capture
        self.llm_only = llm_only
        self.seed = seed
        self.config = config or LayoutConfig.from_globals()

    def process(self, ctx: PageContext) -> PageContext:
        pix = ctx.metadata.get('pixmap')
        if not pix:
            # Fallback or error?
            return ctx

        # Save image for debug/viz
        img_name = f'page_{ctx.page_index:03d}.png'
        img_path = self.out_dir / img_name
        if not img_path.exists():
            pix.save(img_path.as_posix())
        ctx.image_path = img_path

        # Prepare image for LLM
        enc_bytes, enc_mime = _pix_to_encoded(pix, fmt=self.image_format, jpeg_quality=self.jpeg_quality)
        data_url = _bytes_to_data_url(enc_bytes, enc_mime)

        # Call LLM
        # Note: llm_call handles logging internally if log_content is True. 
        # We might want to expose log_content param.
        raw_result = llm_call(
            image_data_url=data_url,
            page_w=ctx.width,
            page_h=ctx.height,
            out_dir=self.out_dir,
            base_url=self.base_url,
            api_key=self.api_key,
            model=self.model,
            temperature=self.temperature,
            timeout=self.timeout,
            strict_capture=self.strict_capture,
            image_format=self.image_format,
            log_id=f'page_{ctx.page_index:03d}',
            config=self.config,
            llm_only=self.llm_only,
            seed=self.seed
        )

        if raw_result:
            ctx.raw_layout = raw_result
            ctx.figures = raw_result.get('figures', [])
            ctx.text_blocks = raw_result.get('text_blocks', [])
            ctx.page_text = raw_result.get('page_text', '')
        else:
            ctx.metadata['engine'] = 'llm_fail'
            
        return ctx

class LayoutRefiner(PipelineStage):
    def __init__(self, config: LayoutConfig, 
                 strict_capture: bool = False,
                 no_small_label_absorb: bool = False,
                 respect_llm_absorb_hints: bool = True,
                 fig_nms_iou: float = 0.0,
                 llm_only: bool = False):
        self.config = config
        self.strict_capture = strict_capture
        self.no_small_label_absorb = no_small_label_absorb
        self.respect_llm_absorb_hints = respect_llm_absorb_hints
        self.fig_nms_iou = fig_nms_iou
        self.llm_only = llm_only

    def process(self, ctx: PageContext) -> PageContext:
        if ctx.metadata.get('engine') == 'llm_fail':
            return ctx
            
        if self.llm_only:
            # Minimal processing for LLM only mode
             # Check for missing refs and equation overlaps just for stats
            miss_ref = sum(1 for tb in ctx.text_blocks if (tb.get('role')=='caption' and not tb.get('ref')))
            ctx.violations['missing_caption_ref'] = miss_ref
            return ctx

        # 1. Link Captions
        link_captions(ctx.figures, ctx.text_blocks, self.config)

        # 2. Demote Equation-Only Figures
        demote_equation_only_figures(ctx.figures, ctx.text_blocks, ctx.width, ctx.height, 
                                     min_area_ratio=self.config.min_figure_area_ratio)

        # 3. NMS
        nms_thr = float(self.fig_nms_iou)
        if nms_thr > 0.0:
            new_figs, remap = nms_figures(ctx.figures, nms_thr)
            # Remap references
            kept_ids = {f.get('id') for f in new_figs if f.get('id')}
            if remap or kept_ids:
                for tb in ctx.text_blocks:
                    r = tb.get('ref')
                    if not r: continue
                    if r in remap:
                        tb['ref'] = remap[r]
                    elif r not in kept_ids:
                        # Try to re-link to nearest figure
                        best = None; best_score = 1e9
                        bb = tb.get('bbox') or [0,0,0,0]
                        role = tb.get('role','paragraph')
                        for f in new_figs:
                            sc = _score_tb_to_fig(bb, f['bbox'], role)
                            if sc < best_score:
                                best_score, best = sc, f
                        if best is not None and best.get('id'):
                            tb['ref'] = best.get('id')
            ctx.figures = new_figs

        # 4. Absorption
        ctx.absorb_events = absorb_into_figures(
            ctx.figures, ctx.text_blocks, ctx.width, ctx.height, self.config,
            strict=self.strict_capture,
            allow_small_labels=(not self.no_small_label_absorb),
            respect_hints=self.respect_llm_absorb_hints
        )

        # 5. Reading Order Fallback
        # We pass None for out_dir here to avoid side effects in pure logic, 
        # but we could pass it if we want logging.
        apply_reading_order_fallback(ctx.text_blocks, self.config, ctx.page_index)

        # 6. Calc Violations
        miss_ref = sum(1 for tb in ctx.text_blocks if (tb.get('role')=='caption' and not tb.get('ref')))
        eq_overlap = 0
        for tb in ctx.text_blocks:
            if tb.get('role')!='equation': continue
            for f in ctx.figures:
                if _iou(tb['bbox'], f['bbox']) > 0:
                    eq_overlap += 1
                    break
        ctx.violations = {'missing_caption_ref': miss_ref, 'equation_overlap': eq_overlap}

        return ctx

class JsonlWriter:
    def __init__(self, out_dir: Path, include_figures: bool = False):
        self.out_dir = out_dir
        self.include_figures = include_figures
        self.layout_path = out_dir / 'structured_layout.jsonl'
        self.inter_path = out_dir / 'interleaved.jsonl'
        # Ensure directories exist
        self.out_dir.mkdir(parents=True, exist_ok=True)
        # We append to files, so we assume they are cleared or managed by the caller/coordinator
        # Or we can open them in append mode.
        
    def write(self, ctx: PageContext):
        # Write structured layout
        rec = ctx.to_layout_record()
        with self.layout_path.open('a', encoding='utf-8') as f:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')

        # Write interleaved sequence
        seq: List[Dict] = []
        img_name = ctx.image_path.name if ctx.image_path else f'page_{ctx.page_index:03d}.png'
        
        seq.append({'kind':'image_page', 'source': img_name, 'bbox': [0, 0, ctx.width, ctx.height]})
        
        for f in ctx.figures:
            if self.include_figures:
                seq.append({'kind':'image_region', 'role':'figure', 'source': img_name, 'bbox': f['bbox'], 'id': f.get('id')})
                
        for tb in ctx.text_blocks:
            item = {'kind':'text', 'type': tb.get('role','paragraph'), 'bbox': tb['bbox'], 'text': tb.get('text','')}
            if tb.get('role') == 'caption' and tb.get('ref'):
                item['ref'] = tb['ref']
            seq.append(item)
            
        with self.inter_path.open('a', encoding='utf-8') as f:
            f.write(json.dumps({'doc_id': ctx.doc_id, 'page_index': ctx.page_index, 'interleaved': seq}, ensure_ascii=False) + '\n')

class LayoutVisualizer:
    def __init__(self, out_dir: Path, viz_text_blocks: bool = False, viz_absorb_debug: bool = False):
        self.out_dir = out_dir
        self.viz_text_blocks = viz_text_blocks
        self.viz_absorb_debug = viz_absorb_debug

    def process(self, ctx: PageContext):
        if not _HAS_CV2 or not ctx.image_path:
            return

        img_bgr = cv2.imread(str(ctx.image_path))
        if img_bgr is None:
            return

        # Main layout viz
        dbg = img_bgr.copy()
        for f in ctx.figures:
            x0, y0, x1, y1 = map(int, f['bbox'])
            cv2.rectangle(dbg, (x0, y0), (x1, y1), (0, 0, 255), 2, cv2.LINE_AA)
            if f.get('id'):
                cv2.putText(dbg, f['id'], (x0 + 3, y0 + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        
        if self.viz_text_blocks:
            for tb in ctx.text_blocks:
                x0, y0, x1, y1 = map(int, tb['bbox'])
                role = tb.get('role', 'paragraph')
                color = (180, 180, 180)
                if role == 'caption': color = (0, 165, 255)
                elif role == 'equation': color = (255, 0, 0)
                elif role == 'heading': color = (255, 0, 255)
                cv2.rectangle(dbg, (x0, y0), (x1, y1), color, 1, cv2.LINE_AA)
                
        cv2.imwrite(str(self.out_dir / f'page_{ctx.page_index:03d}_layout_llm.png'), dbg)

        # Absorb debug viz
        if self.viz_absorb_debug and ctx.absorb_events:
            dbg2 = img_bgr.copy()
            # Draw figures
            for f in ctx.figures:
                x0, y0, x1, y1 = map(int, f['bbox'])
                cv2.rectangle(dbg2, (x0, y0), (x1, y1), (0, 0, 255), 2, cv2.LINE_AA)
            
            color_map = {
                'linked_caption': (0, 140, 255),
                'linked_heading': (255, 0, 255),
                'caption_band': (255, 0, 0),
                'unique_caption_band': (200, 120, 0),
                'small_label': (0, 255, 0),
                'unique_small_label': (0, 200, 80),
                'llm_hint_hard': (255, 255, 0),
                'llm_hint_soft': (180, 255, 100),
                'final_pad': (80, 80, 80)
            }
            
            for ev in ctx.absorb_events:
                reason = ev.get('reason')
                if reason == 'final_pad': continue
                
                tb = ev.get('tb_bbox') or []
                fig_after = ev.get('fig_bbox_after') or []
                c = color_map.get(reason, (255, 255, 255))
                
                if len(tb) >= 4:
                    cv2.rectangle(dbg2, (tb[0], tb[1]), (tb[2], tb[3]), c, 1, cv2.LINE_AA)
                if len(tb) >= 4 and len(fig_after) >= 4:
                    cx = (tb[0]+tb[2])//2; cy=(tb[1]+tb[3])//2
                    fx = (fig_after[0]+fig_after[2])//2; fy=(fig_after[1]+fig_after[3])//2
                    cv2.arrowedLine(dbg2, (cx, cy), (fx, fy), c, 1, cv2.LINE_AA, tipLength=0.2)
                    
            cv2.imwrite(str(self.out_dir / f'page_{ctx.page_index:03d}_absorb_debug.png'), dbg2)
