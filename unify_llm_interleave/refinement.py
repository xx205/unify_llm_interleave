from __future__ import annotations

import math
from typing import List, Dict, Optional, Tuple, Set
from .common import LayoutConfig, _iou, _score_tb_to_fig

def link_captions(figures: List[Dict], text_blocks: List[Dict], config: LayoutConfig) -> None:
    """Link caption text blocks to figures based on proximity and overlap."""
    import re
    cap_pat = re.compile(r'^(?:fig(?:ure)?\.?\s*\d+|图\s*\d+|图表\s*\d+)', re.I)
    
    for tb in text_blocks:
        if tb.get('role') != 'caption' or tb.get('ref'):
            continue
        bb = tb['bbox']
        best = None
        best_score = 1e9
        
        for fig in figures:
            fb = fig['bbox']
            horiz_overlap = max(0, min(bb[2], fb[2]) - max(bb[0], fb[0]))
            overlap_ratio = horiz_overlap / max(1.0, bb[2] - bb[0])
            
            if overlap_ratio < float(config.link_min_overlap_ratio) and _iou(bb, fb) < float(config.link_min_iou):
                continue
                
            score = _score_tb_to_fig(bb, fb, 'caption')
            if cap_pat.search((tb.get('text') or '').strip()):
                score -= 8.0
                
            if score < best_score:
                best_score = score
                best = fig
                
        if best:
            tb['ref'] = best.get('id')

def demote_equation_only_figures(figures: List[Dict], text_blocks: List[Dict], 
                               page_w: int, page_h: int, min_area_ratio: float) -> None:
    """Remove figures that only contain equations or are too small."""
    by_id = {f.get('id'): f for f in figures if f.get('id')}
    to_remove: List[str] = []
    
    for fid, f in list(by_id.items()):
        bb = f.get('bbox') or [0,0,0,0]
        w = max(1, int(bb[2]-bb[0]))
        h = max(1, int(bb[3]-bb[1]))
        h_ratio = h / float(max(1, page_h))
        w_ratio = w / float(max(1, page_w))
        area_ratio = (w_ratio * h_ratio)
        
        refs = [tb for tb in text_blocks if (tb.get('ref') == fid)]
        
        if refs and all((tb.get('role') == 'equation') for tb in refs):
            has_caption = any((tb.get('role') == 'caption') for tb in text_blocks if tb.get('ref') == fid)
            aspect = (max(w, h) / max(1, min(w, h)))
            if not has_caption:
                if not (area_ratio >= 0.01 or aspect >= 1.6):
                    to_remove.append(fid)
        elif not refs and area_ratio < float(min_area_ratio):
            to_remove.append(fid)
            
    if not to_remove:
        return
        
    figures[:] = [f for f in figures if f.get('id') not in to_remove]
    for tb in text_blocks:
        if tb.get('ref') in to_remove:
            tb['ref'] = None
            if tb.get('role') == 'equation':
                tb['absorb'] = 'no'

def nms_figures(figures: List[Dict], iou_thr: float) -> Tuple[List[Dict], Dict[str, str]]:
    """Perform Non-Maximum Suppression on figures."""
    if iou_thr <= 0:
        return figures[:], {}
        
    order = sorted(figures, key=lambda f: (f['bbox'][2]-f['bbox'][0])*(f['bbox'][3]-f['bbox'][1]), reverse=True)
    kept: List[Dict] = []
    remap: Dict[str, str] = {}
    
    for f in order:
        fb = f['bbox']
        found = None
        for k in kept:
            if _iou(fb, k['bbox']) >= iou_thr:
                found = k
                break
        if found is None:
            kept.append(f)
        else:
            if f.get('id') and found.get('id'):
                remap[str(f['id'])] = str(found['id'])
                
    return kept, remap

def absorb_into_figures(figures: List[Dict], text_blocks: List[Dict], 
                       page_w: int, page_h: int, config: LayoutConfig,
                       strict: bool = False, 
                       allow_small_labels: bool = True,
                       respect_hints: bool = True) -> List[Dict]:
    """Absorb small text blocks into figures."""
    events: List[Dict] = []
    mx = max(2, int(page_w * config.absorb_marg_x))
    my = max(2, int(page_h * config.absorb_marg_y))
    my_hc = max(2, int(page_h * (0.02 if strict else 0.03)))

    def overlap_h(b, f):
        return not (b[2] < f[0]-mx or b[0] > f[2]+mx)
    def overlap_h_strict(b, f):
        return (min(b[2], f[2]) - max(b[0], f[0])) > 0
    def near_v(b, f):
        return (b[1] <= f[3]+my and b[3] >= f[1]-my)
    def near_v_hc(b, f):
        return (b[1] <= f[3]+my_hc and b[3] >= f[1]-my_hc)
    def expand(fb, bb):
        fb[0] = min(fb[0], bb[0])
        fb[1] = min(fb[1], bb[1])
        fb[2] = max(fb[2], bb[2])
        fb[3] = max(fb[3], bb[3])

    touched: Set[int] = set()
    by_id = {f.get('id'): f for f in figures if f.get('id')}

    # 1. Respect LLM hints
    if respect_hints:
        for tb_idx, tb in enumerate(text_blocks):
            hint = (tb.get('absorb') or '').lower().strip()
            ref = tb.get('ref')
            if hint in ('hard', 'soft') and ref and ref in by_id:
                fobj = by_id[ref]
                bb = tb['bbox']
                fb = fobj['bbox']
                fb0 = list(fb)
                
                def near_v_relaxed(b, f):
                    my_r = max(2, int(page_h * 0.05))
                    return (b[1] <= f[3]+my_r and b[3] >= f[1]-my_r)
                    
                cond = True if hint == 'hard' else (overlap_h_strict(bb, fb) or near_v_relaxed(bb, fb))
                if cond:
                    expand(fb, bb)
                    touched.add(id(fobj))
                    events.append({
                        'tb_index': tb_idx, 'tb_role': tb.get('role'), 'tb_bbox': list(bb),
                        'fig_id': ref, 'fig_bbox_before': fb0, 'fig_bbox_after': list(fb),
                        'reason': f'llm_hint_{hint}', 'metrics': {}
                    })

    # 2. Absorb linked captions/headings
    for tb_idx, tb in enumerate(text_blocks):
        role = tb.get('role', 'paragraph')
        bb = tb['bbox']
        ref = tb.get('ref')
        if ref and ref in by_id and role in ('caption', 'heading'):
            fb = by_id[ref]['bbox']
            cond_h = overlap_h_strict(bb, fb)
            cond_v = near_v_hc(bb, fb)
            if cond_h and cond_v:
                fb0 = list(fb)
                expand(fb, bb)
                touched.add(id(by_id[ref]))
                events.append({
                    'tb_index': tb_idx, 'tb_role': role, 'tb_bbox': list(bb),
                    'fig_id': ref, 'fig_bbox_before': fb0, 'fig_bbox_after': list(fb),
                    'reason': f'linked_{role}', 'metrics': {'overlap_h_strict': True, 'near_v_hc': True}
                })

    # 3. Absorb small labels
    for tb_idx, tb in enumerate(text_blocks):
        role = tb.get('role', 'paragraph')
        bb = tb['bbox']
        if role not in ('caption', 'heading', 'paragraph'):
            continue
        if tb.get('ref') and role in ('caption', 'heading'):
            continue
            
        h = bb[3] - bb[1]
        w = bb[2] - bb[0]
        w_ratio = config.small_label_w_ratio_strict if strict else config.small_label_w_ratio
        h_ratio = config.small_label_h_ratio_strict if strict else config.small_label_h_ratio
        w_thr = int(page_w * float(max(0.01, min(1.0, w_ratio))))
        w_thr = min(w_thr, 320) if strict else max(w_thr, 240)
        h_thr = max(24, int(page_h * float(max(0.005, min(1.0, h_ratio)))))
        
        is_small = (role == 'paragraph' and allow_small_labels and h <= h_thr and w <= w_thr)

        candidates = []
        for f in figures:
            fb = f['bbox']
            if role in ('caption', 'heading'):
                cond_h = overlap_h_strict(bb, fb)
                cond_v = near_v_hc(bb, fb)
                if not (cond_h and cond_v):
                    continue
            elif is_small:
                if strict:
                    cond_h = overlap_h_strict(bb, fb)
                    my_tight = max(1, int(page_h * 0.01))
                    cond_v = (bb[1] <= fb[3]+my_tight and bb[3] >= fb[1]-my_tight)
                    if not (cond_h and cond_v):
                        continue
                else:
                    if not (overlap_h(bb, fb) and near_v(bb, fb)):
                        continue
            else:
                continue
                
            bx0, by0, bx1, by1 = bb
            bcx = (bx0 + bx1) / 2.0
            bcy = (by0 + by1) / 2.0
            fx0, fy0, fx1, fy1 = fb
            fcx = (fx0 + fx1) / 2.0
            fcy = (fy0 + fy1) / 2.0
            vert_gap = max(0.0, by0 - fy1, fy0 - by1)
            center_dist = abs(bcx - fcx) + 0.5 * abs(bcy - fcy)
            score = vert_gap + 0.3 * center_dist - 20.0 * _iou(bb, fb)
            candidates.append((score, f))

        if not candidates:
            continue
            
        candidates.sort(key=lambda x: x[0])
        best_score, best_f = candidates[0]
        
        if len(candidates) >= 2:
            second_score = candidates[1][0]
            diag = math.hypot(page_w, page_h)
            if (second_score - best_score) <= float(config.ambiguous_gap_frac) * diag:
                continue
                
        fb = best_f['bbox']
        fb0 = list(fb)
        expand(fb, bb)
        events.append({
            'tb_index': tb_idx, 'tb_role': role, 'tb_bbox': list(bb), 'fig_id': best_f.get('id'),
            'fig_bbox_before': fb0, 'fig_bbox_after': list(fb),
            'reason': ('unique_caption_band' if role in ('caption', 'heading') else 'unique_small_label'), 
            'metrics': {}
        })

    # 4. Final padding
    final_pad = 0 if strict else int(max(1, round(page_w * config.absorb_final_expand)))
    if final_pad > 0:
        for f in figures:
            fb0 = list(f['bbox'])
            f['bbox'] = [max(0, fb0[0]-final_pad), max(0, fb0[1]-final_pad), fb0[2]+final_pad, fb0[3]+final_pad]
            events.append({
                'tb_index': None, 'tb_role': None, 'tb_bbox': None, 'fig_id': f.get('id'),
                'fig_bbox_before': fb0, 'fig_bbox_after': list(f['bbox']),
                'reason': 'final_pad', 'metrics': {'pad_x': final_pad, 'pad_y': final_pad}
            })

    return events

def apply_reading_order_fallback(text_blocks: List[Dict], config: LayoutConfig, page_idx: int, out_dir: Optional[Path] = None) -> None:
    """Sort text blocks by reading order if inversion ratio is high."""
    try:
        inv_thr = float(getattr(config, 'order_inversion_ratio_thr', 0.30))
        if len(text_blocks) >= 3 and inv_thr > 0:
            def _key(tb):
                bb = tb.get('bbox') or [0,0,0,0]
                return (int(bb[1]), int(bb[0]))
                
            n = len(text_blocks)
            total = n * (n - 1) // 2
            inv = 0
            for i in range(n):
                yi, xi = _key(text_blocks[i])
                for j in range(i+1, n):
                    yj, xj = _key(text_blocks[j])
                    if (yi, xi) > (yj, xj):
                        inv += 1
            ratio = (inv / total) if total > 0 else 0.0
            
            if ratio >= inv_thr:
                text_blocks.sort(key=_key)
                if out_dir:
                    try:
                        logdir = out_dir / 'logs'
                        logdir.mkdir(exist_ok=True, parents=True)
                        (logdir / f'order_fallback_page_{page_idx:03d}.txt').write_text(
                            f'pair_inversions={inv} total_pairs={total} ratio={ratio:.3f} thr={inv_thr}', 'utf-8')
                    except Exception:
                        pass
    except Exception:
        pass
