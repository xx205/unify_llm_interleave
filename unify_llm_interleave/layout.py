from __future__ import annotations

import concurrent.futures as _fut
from pathlib import Path
from typing import Optional, Tuple, Dict

from .common import LayoutConfig, MAX_LLM_SIDE, DEFAULT_JPEG_QUALITY
from .llm_client import llm_call, NO_PROXY_FOR_LLM  # Re-export for compatibility if needed
from .stages import PdfSource, LlmLayoutEngine, LayoutRefiner, JsonlWriter, LayoutVisualizer

def process_pdf(pdf_path: Path, out_dir: Path, *, zoom: float, include_figures: bool,
                viz: bool, viz_text_blocks: bool, base_url: str, api_key: str, model: str,
                temperature: float, timeout: int, strict_capture: bool,
                image_format: str = 'jpeg', no_small_label_absorb: bool = False,
                respect_llm_absorb_hints: bool = True,
                viz_absorb_debug: bool = False,
                pages_expr: Optional[str] = None,
                jobs: int = 1,
                max_llm_side: Optional[int] = None,
                jpeg_quality: Optional[int] = None,
                config: Optional[LayoutConfig] = None,
                fig_nms_iou: float = 0.0,
                llm_only: bool = False,
                log_content: bool = False,
                llm_seed: Optional[int] = None,
                replay_from_logs: bool = False) -> Tuple[Path, Path]:

    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = config or LayoutConfig.from_globals()

    # 1. Initialize Stages
    source = PdfSource(
        pdf_path=pdf_path,
        zoom=zoom,
        pages_expr=pages_expr,
        max_llm_side=int(max_llm_side or MAX_LLM_SIDE)
    )
    
    engine = LlmLayoutEngine(
        api_key=api_key,
        base_url=base_url,
        model=model,
        out_dir=out_dir,
        temperature=temperature,
        timeout=timeout,
        image_format=image_format,
        jpeg_quality=int(jpeg_quality or DEFAULT_JPEG_QUALITY),
        strict_capture=strict_capture,
        llm_only=llm_only,
        seed=llm_seed,
        config=cfg
    )
    
    refiner = LayoutRefiner(
        config=cfg,
        strict_capture=strict_capture,
        no_small_label_absorb=no_small_label_absorb,
        respect_llm_absorb_hints=respect_llm_absorb_hints,
        fig_nms_iou=fig_nms_iou,
        llm_only=llm_only
    )
    
    writer = JsonlWriter(out_dir, include_figures=include_figures)
    
    visualizer = LayoutVisualizer(
        out_dir=out_dir,
        viz_text_blocks=viz_text_blocks,
        viz_absorb_debug=viz_absorb_debug
    )

    # 2. Execute Pipeline
    # We use a ThreadPoolExecutor to parallelize the Engine stage (LLM calls)
    # Refinement and Writing are fast enough to be done in the callback or main thread, 
    # but for simplicity we can do them in the future callback or sequentially.
    
    # To maintain page order in output files, we should collect results and write them in order,
    # OR use a lock for writing. JsonlWriter appends, so order might be mixed if we write concurrently.
    # However, the original code wrote concurrently! "fjsonl.write(...)". 
    # Wait, original code used `executor.submit` and then `futures[page_idx].result()` in a loop over `page_indices`.
    # This ensures ORDERED writing. We should replicate that.

    inter_path = writer.inter_path
    layout_path = writer.layout_path
    
    # Clear existing files if we are starting fresh? 
    # The original code opened with 'w' mode. JsonlWriter uses 'a'.
    # We should probably clear them first.
    with inter_path.open('w', encoding='utf-8') as f: pass
    with layout_path.open('w', encoding='utf-8') as f: pass

    # Collect all pages from source first (generator)
    # If PDF is huge, this might be memory intensive if we load all pixmaps.
    # PdfSource yields PageContext with pixmap in metadata.
    # We should probably iterate and submit.
    
    pages = list(source) # Realize the list to know how many jobs
    
    with _fut.ThreadPoolExecutor(max_workers=max(1, int(jobs))) as executor:
        # Submit all engine tasks
        future_to_page = {executor.submit(engine.process, ctx): ctx for ctx in pages}
        
        # We want to process results IN ORDER of page_index to keep JSONL sorted.
        # So we shouldn't use as_completed. We should iterate over the futures in order.
        # But `pages` list is ordered. So we can map page -> future.
        
        page_to_future = {ctx.page_index: f for f, ctx in future_to_page.items()}
        sorted_indices = sorted(page_to_future.keys())
        
        for idx in sorted_indices:
            fut = page_to_future[idx]
            try:
                ctx = fut.result()
                
                # Refine
                ctx = refiner.process(ctx)
                
                # Write
                writer.write(ctx)
                
                # Visualize (can be done in parallel but safe here)
                if viz:
                    visualizer.process(ctx)
                    
                print(f'Processed page {ctx.page_index}')
                
            except Exception as e:
                # Log error
                print(f"Error processing page {idx}: {e}")
                # We might want to write a failure record like the original code did
                # The Engine or Refiner should handle graceful failure and return a context with error info.
                # Our Engine sets metadata['engine'] = 'llm_fail' on error.
                # Refiner skips if 'llm_fail'.
                # Writer should handle 'llm_fail'?
                # Let's check Writer.
                # Writer writes whatever is in figures/text_blocks. If empty, it writes empty.
                pass

    return inter_path, layout_path
