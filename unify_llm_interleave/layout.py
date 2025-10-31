from __future__ import annotations

import base64
import concurrent.futures as _fut
import json
import math
import os
import re
import sys
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import fitz  # PyMuPDF
    _HAS_FITZ = True
except Exception:
    fitz = None  # type: ignore
    _HAS_FITZ = False

try:
    import cv2  # type: ignore
    _HAS_CV2 = True
except Exception:
    _HAS_CV2 = False

import requests

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


# -------------- HTTP / Proxy --------------
NO_PROXY_FOR_LLM = True
_thread_local = threading.local()


def _get_session() -> requests.Session:
    s = getattr(_thread_local, 'session', None)
    if s is None:
        s = requests.Session()
        s.trust_env = (not NO_PROXY_FOR_LLM)
        setattr(_thread_local, 'session', s)
    return s


def _http_post_json(url: str, headers: Dict, payload: Dict, timeout: Tuple[float, float] | float):
    s = _get_session()
    if NO_PROXY_FOR_LLM:
        return s.post(url, headers=headers, json=payload, timeout=timeout, proxies={})
    return s.post(url, headers=headers, json=payload, timeout=timeout)


def _post_with_retry(url: str, headers: Dict, payload: Dict, timeout: Tuple[float, float] | float,
                     max_tries: int = 5, base_delay: float = 1.0, max_total_wait: float = 120.0) -> requests.Response:
    import time
    last_exc = None
    delay = base_delay
    t0 = time.time()
    for i in range(max_tries):
        try:
            resp = _http_post_json(url, headers, payload, timeout)
            if resp is not None and resp.status_code not in (429, 408) and resp.status_code < 500:
                try:
                    resp._retry_meta = {'tries': i+1, 'last_status': resp.status_code, 'total_wait': time.time()-t0, 'retry_after_used': None}
                except Exception:
                    pass
                return resp
            last_exc = resp
        except (requests.Timeout, requests.ConnectionError) as e:
            last_exc = e
        except requests.RequestException as e:
            last_exc = e
        try:
            import random
            ra = None
            if isinstance(last_exc, requests.Response):
                try:
                    ra_hdr = last_exc.headers.get('Retry-After')
                    if ra_hdr:
                        try:
                            ra = float(ra_hdr)
                        except ValueError:
                            from email.utils import parsedate_to_datetime
                            dt = parsedate_to_datetime(ra_hdr)
                            if dt is not None:
                                ra = max(0.0, (dt.timestamp() - time.time()))
                except Exception:
                    ra = None
            wait_s = (ra if ra is not None else delay) + random.uniform(0, 0.4)
            if (time.time() - t0 + wait_s) > max_total_wait:
                wait_s = max(0.0, max_total_wait - (time.time() - t0))
            time.sleep(wait_s)
        except Exception:
            pass
        delay = min(delay * 2.0, 16.0)
    if isinstance(last_exc, requests.Response):
        try:
            last_exc._retry_meta = {'tries': max_tries, 'last_status': last_exc.status_code, 'total_wait': time.time()-t0, 'retry_after_used': True}
        except Exception:
            pass
        return last_exc
    if last_exc is not None:
        raise last_exc  # type: ignore
    return _http_post_json(url, headers, payload, timeout)


def _bytes_to_data_url(data: bytes, mime: str) -> str:
    b64 = base64.b64encode(data).decode('ascii')
    return f'data:{mime};base64,{b64}'


def _pix_to_encoded(pix: fitz.Pixmap, fmt: str = 'jpeg', jpeg_quality: int = DEFAULT_JPEG_QUALITY) -> Tuple[bytes, str]:
    f = (fmt or 'jpeg').lower()
    if f in ('jpg', 'jpeg'):
        try:
            p = pix
            if getattr(p, 'alpha', 0):
                p = fitz.Pixmap(fitz.csRGB, p)
            data = p.tobytes('jpg', quality=int(jpeg_quality))
            return data, 'image/jpeg'
        except Exception:
            pass
    try:
        data = pix.tobytes('png')
        return data, 'image/png'
    except Exception:
        if _HAS_CV2:
            try:
                import numpy as _np
                arr = _np.frombuffer(pix.samples, dtype=_np.uint8)
                img = arr.reshape((pix.height, pix.width, pix.n))
                if pix.n == 4:
                    img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA)
                elif pix.n == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                ok, buf = cv2.imencode('.png', img)
                if ok:
                    return buf.tobytes(), 'image/png'
            except Exception:
                pass
        return b'', 'application/octet-stream'


def _coerce_llm_layout(obj: dict) -> Optional[dict]:
    if not isinstance(obj, dict):
        return None
    out = {'figures': [], 'text_blocks': [], 'page_text': obj.get('page_text','')}
    figs = obj.get('figures') or obj.get('figure_regions') or []
    if isinstance(figs, list):
        for i, it in enumerate(figs):
            if not isinstance(it, dict):
                continue
            bb = it.get('bbox'); fid = it.get('id') or f'fig_{i}'
            if isinstance(bb, list) and len(bb) >= 4:
                nums = []
                for v in bb:
                    if isinstance(v,(int,float)):
                        nums.append(int(round(v)))
                        if len(nums) >= 4: break
                if len(nums) == 4:
                    out['figures'].append({'id': str(fid), 'bbox': nums})
    tbs = obj.get('text_blocks') or []
    if isinstance(tbs, list):
        for it in tbs:
            if not isinstance(it, dict): continue
            bb = it.get('bbox'); role = (it.get('role') or 'paragraph').lower(); txt = it.get('text',''); ref = it.get('ref')
            absorb = (it.get('absorb') or it.get('absorb_policy') or '').lower().strip() or None
            if absorb not in (None, 'hard', 'soft', 'no'):
                absorb = None
            if isinstance(bb, list) and len(bb) >= 4:
                nums = []
                for v in bb:
                    if isinstance(v,(int,float)):
                        nums.append(int(round(v)))
                        if len(nums) >= 4: break
                if len(nums) == 4:
                    out['text_blocks'].append({'bbox': nums, 'role': role, 'text': str(txt), 'ref': (str(ref) if ref else None), 'absorb': absorb})
    if not isinstance(out.get('page_text'), str):
        out['page_text'] = ''
    return out


def _link_captions(figs: List[Dict], tbs: List[Dict], *, config: Optional[LayoutConfig] = None) -> None:
    cfg = config or LayoutConfig.from_globals()
    cap_pat = re.compile(r'^(?:fig(?:ure)?\.?\s*\d+|图\s*\d+|图表\s*\d+)', re.I)
    for tb in tbs:
        if tb.get('role')!='caption' or tb.get('ref'): continue
        bb = tb['bbox']
        best=None; best_score=1e9
        for fig in figs:
            fb = fig['bbox']
            horiz_overlap = max(0, min(bb[2],fb[2]) - max(bb[0],fb[0]))
            overlap_ratio = horiz_overlap / max(1.0, bb[2]-bb[0])
            if overlap_ratio < float(cfg.link_min_overlap_ratio) and _iou(bb, fb) < float(cfg.link_min_iou):
                continue
            score = _score_tb_to_fig(bb, fb, 'caption')
            if cap_pat.search((tb.get('text') or '').strip()):
                score -= 8.0
            if score < best_score:
                best_score=score; best=fig
        if best:
            tb['ref']=best.get('id')


def _absorb_into_figures(figs: List[Dict], tbs: List[Dict], page_w:int, page_h:int,
                         *, strict: bool = False, return_events: bool = False,
                         allow_small_labels: Optional[bool] = None,
                         respect_hints: bool = True,
                         config: Optional[LayoutConfig] = None) -> Optional[List[Dict]]:
    events: List[Dict] = []
    if True:
        pass
    cfg = config or LayoutConfig.from_globals()
    mx=max(2,int(page_w*cfg.absorb_marg_x)); my=max(2,int(page_h*cfg.absorb_marg_y))
    my_hc=max(2,int(page_h*(0.02 if strict else 0.03)))

    def overlap_h(b,f):
        return not (b[2] < f[0]-mx or b[0] > f[2]+mx)
    def overlap_h_strict(b,f):
        return (min(b[2],f[2]) - max(b[0],f[0])) > 0
    def near_v(b,f):
        return (b[1] <= f[3]+my and b[3] >= f[1]-my)
    def near_v_hc(b,f):
        return (b[1] <= f[3]+my_hc and b[3] >= f[1]-my_hc)
    def expand(fb,bb):
        fb[0]=min(fb[0],bb[0]); fb[1]=min(fb[1],bb[1]); fb[2]=max(fb[2],bb[2]); fb[3]=max(fb[3],bb[3])

    # Track figures that were explicitly expanded; reserved for future heuristics/logs
    touched: set[int] = set()

    if respect_hints:
        by_id={f.get('id'): f for f in figs if f.get('id')}
        for tb_idx, tb in enumerate(tbs):
            hint = (tb.get('absorb') or '').lower().strip()
            ref = tb.get('ref')
            if hint in ('hard','soft') and ref and ref in by_id:
                fobj = by_id[ref]
                bb = tb['bbox']; fb = fobj['bbox']
                fb0=list(fb)
                def near_v_relaxed(b,f):
                    my_r = max(2,int(page_h*0.05))
                    return (b[1] <= f[3]+my_r and b[3] >= f[1]-my_r)
                cond = True if hint=='hard' else (overlap_h_strict(bb,fb) or near_v_relaxed(bb,fb))
                if cond:
                    expand(fb, bb)
                    touched.add(id(fobj))
                    if return_events:
                        events.append({'tb_index': tb_idx, 'tb_role': tb.get('role'), 'tb_bbox': list(bb),
                                       'fig_id': ref, 'fig_bbox_before': fb0, 'fig_bbox_after': list(fb),
                                       'reason': f'llm_hint_{hint}', 'metrics': {}})

    by_id={f.get('id'): f for f in figs if f.get('id')}
    for tb_idx, tb in enumerate(tbs):
        role=tb.get('role','paragraph'); bb=tb['bbox']; ref=tb.get('ref')
        if ref and ref in by_id and role in ('caption','heading'):
            fb = by_id[ref]['bbox']
            cond_h = overlap_h_strict(bb, fb)
            cond_v = near_v_hc(bb, fb)
            if cond_h and cond_v:
                fb0 = list(fb)
                expand(fb, bb)
                touched.add(id(by_id[ref]))
                if return_events:
                    events.append({'tb_index': tb_idx, 'tb_role': role, 'tb_bbox': list(bb),
                                   'fig_id': ref, 'fig_bbox_before': fb0, 'fig_bbox_after': list(fb),
                                   'reason': f'linked_{role}', 'metrics': {'overlap_h_strict': True, 'near_v_hc': True}})

    for tb_idx, tb in enumerate(tbs):
        role=tb.get('role','paragraph'); bb=tb['bbox']
        if role not in ('caption','heading','paragraph'):
            continue
        if tb.get('ref') and role in ('caption','heading'):
            continue
        h=bb[3]-bb[1]; w=bb[2]-bb[0]
        w_ratio = cfg.small_label_w_ratio_strict if strict else cfg.small_label_w_ratio
        h_ratio = cfg.small_label_h_ratio_strict if strict else cfg.small_label_h_ratio
        w_thr = int(page_w * float(max(0.01, min(1.0, w_ratio))))
        w_thr = min(w_thr, 320) if strict else max(w_thr, 240)
        h_thr = max(24, int(page_h * float(max(0.005, min(1.0, h_ratio)))))
        allow_small = (not strict) if (allow_small_labels is None) else bool(allow_small_labels)
        is_small = (role=='paragraph' and allow_small and h<=h_thr and w<=w_thr)

        candidates = []
        for f in figs:
            fb = f['bbox']
            if role in ('caption','heading'):
                cond_h = overlap_h_strict(bb, fb)
                cond_v = near_v_hc(bb, fb)
                if not (cond_h and cond_v):
                    continue
            elif is_small:
                if strict:
                    cond_h = overlap_h_strict(bb, fb)
                    my_tight = max(1, int(page_h*0.01))
                    cond_v = (bb[1] <= fb[3]+my_tight and bb[3] >= fb[1]-my_tight)
                    if not (cond_h and cond_v):
                        continue
                else:
                    if not (overlap_h(bb,fb) and near_v(bb,fb)):
                        continue
            else:
                continue
            bx0,by0,bx1,by1 = bb; bcx=(bx0+bx1)/2.0; bcy=(by0+by1)/2.0
            fx0,fy0,fx1,fy1 = fb; fcx=(fx0+fx1)/2.0; fcy=(fy0+fy1)/2.0
            vert_gap = max(0.0, by0 - fy1, fy0 - by1)
            center_dist = abs(bcx-fcx) + 0.5*abs(bcy-fcy)
            score = vert_gap + 0.3*center_dist - 20.0*_iou(bb, fb)
            candidates.append((score, f))

        if not candidates:
            continue
        candidates.sort(key=lambda x: x[0])
        best_score, best_f = candidates[0]
        if len(candidates) >= 2:
            second_score = candidates[1][0]
            diag = math.hypot(page_w, page_h)
            if (second_score - best_score) <= float(cfg.ambiguous_gap_frac) * diag:
                continue
        fb = best_f['bbox']; fb0 = list(fb)
        fb[0]=min(fb[0],bb[0]); fb[1]=min(fb[1],bb[1]); fb[2]=max(fb[2],bb[2]); fb[3]=max(fb[3],bb[3])
        if return_events:
            events.append({'tb_index': tb_idx, 'tb_role': role, 'tb_bbox': list(bb), 'fig_id': best_f.get('id'),
                           'fig_bbox_before': fb0, 'fig_bbox_after': list(fb),
                           'reason': ('unique_caption_band' if role in ('caption','heading') else 'unique_small_label'), 'metrics': {}})

    final_pad = 0 if strict else int(max(1, round(page_w*cfg.absorb_final_expand)))
    if final_pad > 0:
        for f in figs:
            fb0 = list(f['bbox'])
            f['bbox'] = [max(0, fb0[0]-final_pad), max(0, fb0[1]-final_pad), fb0[2]+final_pad, fb0[3]+final_pad]
            if return_events:
                events.append({'tb_index': None, 'tb_role': None, 'tb_bbox': None, 'fig_id': f.get('id'),
                               'fig_bbox_before': fb0, 'fig_bbox_after': list(f['bbox']),
                               'reason': 'final_pad', 'metrics': {'pad_x': final_pad, 'pad_y': final_pad}})

    return events if return_events else None


def llm_call(*,
             image_path: Optional[Path] = None,
             image_data_url: Optional[str] = None,
             page_w: int, page_h: int,
             out_dir: Path,
             base_url: str, api_key: str, model: str, temperature: float, timeout: int,
             strict_capture: bool, log_content: bool=True, image_format: str = 'jpeg',
             log_id: Optional[str] = None,
             config: Optional[LayoutConfig] = None,
             llm_only: bool = False,
             seed: Optional[int] = None) -> Optional[dict]:

    headers={
        'Authorization': f'Bearer {api_key}'
    }
    sys_prompt = (
        'You are a document layout analyzer. Follow ALL rules:\n'
        '- Output JSON only (no prose).\n'
        '- BBoxes are integers on a 0..1000 grid.\n'
        '- Figures must include nearby descriptive text (captions/labels/axes/legends) or link them via a caption text_block with ref.\n'
        "- Give absorb hint per text_block: 'hard' | 'soft' | 'no'.\n"
        '- Equations are text_blocks (never figures).\n'
        '- Display math uses "$$ ... $$" with math only; inline math uses "$ ... $" inside paragraphs. Do not put numbering or prose inside "$$ ... $$".\n'
        '- Use standard KaTeX/MathJax macros; do not invent macros (\\sqrt, \\exp, \\sum, \\ instead of \\bigsqrt, \\bigexp, \\bigsum, \\backslash) or use spaced control sequences (no "\\text m"; if needed, "\\text{m}"). Prefer canonical, styling-light forms.\n'
    )
    strict_note = (" Do NOT output caption-like text outside figures. If uncertain whether a text is a caption/legend of a graphic, add it as a caption text_block with correct 'ref' instead of creating a new figure. "
                   " Never treat equations as figures. Each figure must have a stable 'id', e.g., 'fig_0', 'fig_1'.") if strict_capture else ""
    user_text = (
        'Return strictly this JSON (no commentary), coordinates are integers in 0..1000:\n'
        '{\n'
        '  "figures": [ { "id": "fig_0", "bbox": [x0,y0,x1,y1] }, ... ],\n'
        '  "text_blocks": [ { "bbox": [x0,y0,x1,y1], "role": "paragraph|equation|heading|caption", "text": "...", "ref": "fig_i"?, "absorb": "hard|soft|no"? } ]\n'
        '}\n'
        'Notes: figure ids must be stable across the page ("fig_0", "fig_1", ...). Use standard KaTeX/MathJax macros; do not invent macros (\\sqrt, \\exp, \\sum, \\ instead of \\bigsqrt, \\bigexp, \\bigsum, \\backslash). Prefer canonical, styling-light forms.\n'
    )

    if image_data_url is None and (not image_path or not Path(image_path).exists()):
        return None
    if image_data_url is not None:
        data_url = image_data_url
    else:
        try:
            data = Path(image_path).read_bytes() if image_path else b''
            mime = 'image/png' if (image_path and image_path.suffix.lower()=='.png') else 'image/jpeg'
            data_url = _bytes_to_data_url(data, mime)
        except Exception:
            return None
    payload = {
        'model': model,
        'messages': [
            {'role': 'system', 'content': sys_prompt + strict_note},
            {'role': 'user', 'content': [
                {'type': 'text', 'text': user_text},
                {'type': 'image_url', 'image_url': {'url': data_url}}
            ]}
        ],
        'thinking': {
            'type': 'disabled',
        },
        'response_format': {'type': 'json_object'},
        'temperature': float(temperature),
        'top_p': 1,
        'presence_penalty': 0.1,
        'frequency_penalty': 0,
        'max_tokens': 16384
    }
    if seed is not None:
        try:
            payload['seed'] = int(seed)
        except Exception:
            pass
    url = f'{base_url}/chat/completions'
    logdir = out_dir / 'logs'; logdir.mkdir(exist_ok=True, parents=True)
    _tag = (log_id or (image_path.stem if image_path else 'page'))
    (logdir / f'I_pre_{_tag}.json').write_text(json.dumps({'endpoint':url,'model':model}, ensure_ascii=False, indent=2), 'utf-8')
    try:
        print(payload['messages'][0]['content'], file=sys.stderr, flush=True)
        print(payload['messages'][1]['content'][0]['text'], file=sys.stderr, flush=True)
        resp = _post_with_retry(url, headers, payload, timeout)
    except Exception as e:
        (logdir / f'I_err_{_tag}.txt').write_text(f'EXC: {e}', 'utf-8')
        return None
    st = {'status': resp.status_code}
    meta = getattr(resp, '_retry_meta', None)
    if isinstance(meta, dict):
        st.update(meta)
    (logdir / f'I_status_{_tag}.json').write_text(json.dumps(st, ensure_ascii=False, indent=2), 'utf-8')
    if resp.status_code != 200:
        (logdir / f'I_err_{_tag}.txt').write_text(resp.text, 'utf-8')
        try:
            slim = {
                'endpoint': url,
                'model': payload.get('model'),
                'temperature': payload.get('temperature'),
                'max_tokens': payload.get('max_tokens'),
                'response_format': payload.get('response_format'),
                'messages': []
            }
            for m in payload.get('messages', []):
                if isinstance(m, dict) and m.get('role') == 'user' and isinstance(m.get('content'), list):
                    cont = []
                    for it in m['content']:
                        if isinstance(it, dict) and it.get('type') == 'image_url':
                            cont.append({'type': 'image_url', 'image_url': {'url': '<omitted-data-url>'}})
                        elif isinstance(it, dict) and it.get('type') == 'text':
                            cont.append({'type': 'text', 'text': it.get('text', '')})
                    slim['messages'].append({'role': 'user', 'content': cont})
                elif isinstance(m, dict) and m.get('role') == 'system':
                    slim['messages'].append({'role': 'system', 'content': m.get('content', '')})
                else:
                    slim['messages'].append(m)
            (logdir / f'I_err_payload_{_tag}.json').write_text(json.dumps(slim, ensure_ascii=False, indent=2), 'utf-8')
        except Exception:
            pass
        return None
    j = resp.json()
    if log_content:
        (logdir / f'I_resp_{_tag}.json').write_text(json.dumps(j, ensure_ascii=False, indent=2), 'utf-8')
    else:
        jr = dict(j)
        if isinstance(jr.get('choices'), list):
            jr['choices'] = [
                {k: v for k, v in c.items() if k != 'message'} if isinstance(c, dict) else c
                for c in jr['choices']
            ]
        (logdir / f'I_resp_{_tag}.json').write_text(json.dumps(jr, ensure_ascii=False, indent=2), 'utf-8')
    txt = j.get('choices', [{}])[0].get('message', {}).get('content')
    if not txt: return None
    if log_content:
        try:
            (logdir / f'I_content_{_tag}.txt').write_text(txt, 'utf-8')
        except Exception:
            pass
    txt_src = txt

    def _shield_inner_json(s: str) -> str:
        try:
            import re as _re
            s = _re.sub(r'\\([bfnrt])', r'\\\\\1', s)
            s = _re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', s)
            return s
        except Exception:
            return s
    obj = _extract_json_obj(txt_src)
    extract_meta = get_last_extract_meta(clear=True)
    if obj is None:
        txt_shield = _shield_inner_json(txt_src)
        if log_content:
            try:
                (logdir / f'I_content_sanitized_{_tag}.txt').write_text(txt_shield, 'utf-8')
            except Exception:
                pass
        obj = _extract_json_obj(txt_shield)
        # log a unified diff for shielding stage
        try:
            import difflib
            diff = difflib.unified_diff(
                txt_src.splitlines(keepends=True),
                txt_shield.splitlines(keepends=True),
                fromfile='content.orig', tofile='content.shielded')
            (logdir / f'I_escape_diff_{_tag}.txt').write_text(''.join(diff), 'utf-8')
        except Exception:
            pass
        extract_meta = get_last_extract_meta(clear=True)
    if obj is None: return None
    # Persist extract meta if available (tracks internal fix_invalid_escapes)
    try:
        if extract_meta:
            (logdir / f'I_escape_meta_{_tag}.json').write_text(json.dumps(extract_meta, ensure_ascii=False, indent=2), 'utf-8')
            if extract_meta.get('used_fix_invalid_escapes') and extract_meta.get('original_fragment') is not None:
                import difflib
                diff = difflib.unified_diff(
                    str(extract_meta.get('original_fragment')).splitlines(keepends=True),
                    str(extract_meta.get('sanitized_fragment')).splitlines(keepends=True),
                    fromfile='fragment.orig', tofile='fragment.sanitized')
                (logdir / f'I_escape_meta_diff_{_tag}.txt').write_text(''.join(diff), 'utf-8')
    except Exception:
        pass
    norm = _coerce_llm_layout(obj)
    if norm is None: return None

    figures = [{'id': f['id'], 'bbox': _scale_kilo_to_px(f['bbox'], page_w, page_h)} for f in norm.get('figures', [])]
    text_blocks = []
    # track pre/post text to log escape changes
    _tb_changes = []
    for idx, tb in enumerate(norm.get('text_blocks', [])):
        role = tb.get('role','paragraph')
        before = tb.get('text','')
        # apply LaTeX normalization + optional macro repair inside math
        after = _normalize_latex_backslashes(before, role, macro_mode=getattr((config or LayoutConfig()), 'macro_repair_mode', 'off'))
        after2 = _purge_ctrl(after)
        if before != after2:
            try:
                import difflib
                diff = ''.join(difflib.unified_diff(
                    str(before).splitlines(keepends=True),
                    str(after2).splitlines(keepends=True),
                    fromfile=f'tb{idx}.orig', tofile=f'tb{idx}.normalized'))
            except Exception:
                diff = ''
            _tb_changes.append({
                'index': idx, 'role': role, 'ref': tb.get('ref'),
                'before': before, 'after': after2,
                'diff': diff
            })
        text_blocks.append({
            'bbox': _scale_kilo_to_px(tb['bbox'], page_w, page_h),
            'role': role,
            'text': after2,
            'ref': tb.get('ref'),
            'absorb': tb.get('absorb')
        })
    # write per-page text escape change logs
    try:
        if _tb_changes:
            (logdir / f'I_text_escape_changes_{_tag}.json').write_text(json.dumps(_tb_changes, ensure_ascii=False, indent=2), 'utf-8')
            # and a human-readable diff bundle
            txt = []
            for ch in _tb_changes:
                hdr = f"# text_block[{ch['index']}] role={ch['role']} ref={ch.get('ref')}\n"
                txt.append(hdr)
                if ch.get('diff'):
                    txt.append(ch['diff'])
                else:
                    txt.append('--- before\n')
                    txt.append(str(ch['before']) + '\n')
                    txt.append('+++ after\n')
                    txt.append(str(ch['after']) + '\n')
                txt.append('\n')
            (logdir / f'I_text_escape_changes_{_tag}.txt').write_text(''.join(txt), 'utf-8')
    except Exception:
        pass
    cfg = config or LayoutConfig.from_globals()
    # Note: Linking and absorption are now centralized in process_pdf so that CLI flags
    # (strict/no-small-label-absorb/no-respect-llm-absorb-hints) uniformly apply once.
    # llm_call returns only the normalized LLM output scaled to pixels.
    return {'figures': figures, 'text_blocks': text_blocks, 'page_text': norm.get('page_text','')}


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

    if not _HAS_FITZ and not replay_from_logs:
        raise RuntimeError('PyMuPDF (fitz) not installed. Please install: pip install pymupdf or use --replay-from-logs')

    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = config or LayoutConfig.from_globals()
    doc = None
    if _HAS_FITZ:
        doc = fitz.open(pdf_path.as_posix())
    inter_path = out_dir / 'interleaved.jsonl'
    layout_path = out_dir / 'structured_layout.jsonl'
    try:
        # determine pages to process BEFORE opening outputs (avoid truncation during replay)
        if doc is not None:
            page_indices = _parse_pages_expr(pages_expr, doc.page_count)
        else:
            page_indices = []
            # prefer logs/I_content_page_*.txt for replay page discovery
            try:
                logdir = out_dir / 'logs'
                if logdir.exists():
                    import re as _re
                    for pth in sorted(logdir.glob('I_content_page_*.txt')):
                        m = _re.search(r'I_content_page_(\d{3})', pth.name)
                        if m:
                            page_indices.append(int(m.group(1)))
                if not page_indices and (out_dir / 'structured_layout.jsonl').exists():
                    # secondary fallback: old layout file
                    recs = []
                    with (out_dir / 'structured_layout.jsonl').open('r', encoding='utf-8') as fr:
                        for line in fr:
                            j = json.loads(line)
                            recs.append(j)
                    page_indices = sorted({int(r.get('page_index', 0)) for r in recs})
                if pages_expr and page_indices:
                    # unify with _parse_pages_expr semantics
                    try:
                        from .common import _parse_pages_expr as _ppe
                        sel = set(_ppe(pages_expr, max(page_indices) + 1))
                        page_indices = [i for i in page_indices if i in sel]
                    except Exception:
                        pass
            except Exception:
                page_indices = []

        with inter_path.open('w', encoding='utf-8') as fjsonl, layout_path.open('w', encoding='utf-8') as flayout:
            executor = _fut.ThreadPoolExecutor(max_workers=max(1, int(jobs)))
            futures: Dict[int, _fut.Future] = {}
            page_meta: Dict[int, Dict] = {}
            for page_idx in page_indices:
                if doc is not None:
                    page = doc[page_idx]
                    pw, ph = page.rect.width, page.rect.height
                    base_side = max(pw, ph)
                    side_cap = float(max_llm_side or MAX_LLM_SIDE)
                    scale = min(zoom, max(0.5, side_cap/float(base_side)))
                    pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
                    img_name = f'page_{page_idx:03d}.png'
                    img_path = out_dir / img_name
                    pix.save(img_path.as_posix())
                    w,h = pix.width, pix.height
                else:
                    # derive from existing layout if present
                    img_name = f'page_{page_idx:03d}.png'
                    img_path = out_dir / img_name
                    # default size
                    w = h = 1000
                    # prefer existing page image for size if present (replay scenario)
                    try:
                        if img_path.exists():
                            if _HAS_CV2:
                                _img = cv2.imread(str(img_path))
                                if _img is not None:
                                    h, w = _img.shape[:2]
                            else:
                                from PIL import Image as _PIL_Image  # type: ignore
                                with _PIL_Image.open(str(img_path)) as _im:
                                    w, h = _im.size
                    except Exception:
                        pass
                    try:
                        if layout_path.exists():
                            # find matching record
                            with layout_path.open('r', encoding='utf-8') as fr:
                                for line in fr:
                                    j = json.loads(line)
                                    if int(j.get('page_index', -1)) == int(page_idx):
                                        sz = j.get('page_size')
                                        if isinstance(sz, list) and len(sz) == 2:
                                            w, h = int(sz[0]), int(sz[1])
                                        break
                    except Exception:
                        pass

                if not replay_from_logs:
                    enc_bytes, enc_mime = _pix_to_encoded(pix, fmt=image_format or 'jpeg', jpeg_quality=int(jpeg_quality or DEFAULT_JPEG_QUALITY))
                    data_url = _bytes_to_data_url(enc_bytes, enc_mime) if enc_bytes else None
                    image_arg = {'image_data_url': data_url} if enc_bytes else {'image_path': img_path}

                    fut = executor.submit(
                        llm_call,
                        **image_arg,
                        page_w=w,
                        page_h=h,
                        out_dir=out_dir,
                        base_url=base_url,
                        api_key=api_key,
                        model=model,
                        temperature=temperature,
                        timeout=timeout,
                        strict_capture=strict_capture,
                        log_content=bool(log_content),
                        image_format=image_format,
                        log_id=f'page_{page_idx:03d}',
                        config=cfg,
                        llm_only=bool(llm_only),
                        seed=llm_seed
                    )
                    futures[page_idx] = fut
                page_meta[page_idx] = {'w': w, 'h': h, 'img_name': img_name, 'img_path': img_path}

            for page_idx in page_indices:
                meta = page_meta[page_idx]
                w = meta['w']; h = meta['h']; img_name = meta['img_name']; img_path = meta['img_path']
                if not replay_from_logs:
                    try:
                        llm = futures[page_idx].result()
                    except Exception as e:
                        try:
                            logdir = out_dir / 'logs'; logdir.mkdir(exist_ok=True, parents=True)
                            (logdir / f'I_err_page_{page_idx:03d}.txt').write_text(str(e), 'utf-8')
                        except Exception:
                            pass
                        llm = None
                else:
                    # Replay from existing content snapshot
                    logdir = out_dir / 'logs'; logdir.mkdir(exist_ok=True, parents=True)
                    tag = f'page_{page_idx:03d}'
                    content_fp = logdir / f'I_content_{tag}.txt'
                    if content_fp.exists():
                        try:
                            txt_src = content_fp.read_text('utf-8')
                            # Reuse the inner stages of llm_call to build figures/text_blocks and write escape logs
                            obj = _extract_json_obj(txt_src)
                            extract_meta = get_last_extract_meta(clear=True)
                            if obj is None:
                                # Try sanitized copy if available
                                try:
                                    san = (logdir / f'I_content_sanitized_{tag}.txt')
                                    if san.exists():
                                        txt_src = san.read_text('utf-8')
                                        obj = _extract_json_obj(txt_src)
                                        extract_meta = get_last_extract_meta(clear=True)
                                except Exception:
                                    pass
                            if obj is None:
                                # Final salvage: coarse regex extraction to avoid dropping page entirely
                                try:
                                    import re as _re
                                    figures = []
                                    tblocks = []
                                    for m in _re.finditer(r'\{\s*"id"\s*:\s*"(?P<id>[^"]+)"\s*,\s*"bbox"\s*:\s*\[(?P<b>[^\]]+)\]\s*\}', txt_src):
                                        try:
                                            bb = [int(float(x.strip())) for x in m.group('b').split(',')[:4]]
                                            figures.append({'id': m.group('id'), 'bbox': _scale_kilo_to_px(bb, w, h) if max(bb)>10 else bb})
                                        except Exception:
                                            pass
                                    for m in _re.finditer(r'\{\s*"bbox"\s*:\s*\[(?P<b>[^\]]+)\]\s*,\s*"role"\s*:\s*"(?P<role>[^"]+)"', txt_src):
                                        try:
                                            bb = [int(float(x.strip())) for x in m.group('b').split(',')[:4]]
                                            role = m.group('role').lower()
                                            tblocks.append({'bbox': _scale_kilo_to_px(bb, w, h) if max(bb)>10 else bb, 'role': role, 'text': ''})
                                        except Exception:
                                            pass
                                    if figures or tblocks:
                                        llm = {'figures': figures, 'text_blocks': tblocks, 'page_text': ''}
                                    else:
                                        llm = None
                                except Exception:
                                    llm = None
                            else:
                                norm = _coerce_llm_layout(obj)
                                if norm is None:
                                    llm = None
                                else:
                                    figures = [{'id': f['id'], 'bbox': _scale_kilo_to_px(f['bbox'], w, h)} for f in norm.get('figures', [])]
                                    text_blocks = []
                                    _tb_changes = []
                                    for idx2, tb in enumerate(norm.get('text_blocks', [])):
                                        role = tb.get('role','paragraph')
                                        before = tb.get('text','')
                                        after = _normalize_latex_backslashes(before, role, macro_mode=getattr((cfg), 'macro_repair_mode', 'off'))
                                        after2 = _purge_ctrl(after)
                                        if before != after2:
                                            try:
                                                import difflib
                                                diff = ''.join(difflib.unified_diff(
                                                    str(before).splitlines(keepends=True),
                                                    str(after2).splitlines(keepends=True),
                                                    fromfile=f'tb{idx2}.orig', tofile=f'tb{idx2}.normalized'))
                                            except Exception:
                                                diff = ''
                                            _tb_changes.append({'index': idx2, 'role': role, 'ref': tb.get('ref'), 'before': before, 'after': after2, 'diff': diff})
                                        text_blocks.append({'bbox': _scale_kilo_to_px(tb['bbox'], w, h), 'role': role, 'text': after2, 'ref': tb.get('ref'), 'absorb': tb.get('absorb')})
                                    if _tb_changes:
                                        try:
                                            (logdir / f'I_text_escape_changes_{tag}.json').write_text(json.dumps(_tb_changes, ensure_ascii=False, indent=2), 'utf-8')
                                            txt = []
                                            for ch in _tb_changes:
                                                hdr = f"# text_block[{ch['index']}] role={ch['role']} ref={ch.get('ref')}\n"
                                                txt.append(hdr)
                                                if ch.get('diff'):
                                                    txt.append(ch['diff'])
                                                    txt.append('\n')
                                            (logdir / f'I_text_escape_changes_{tag}.txt').write_text(''.join(txt), 'utf-8')
                                        except Exception:
                                            pass
                                    llm = {'figures': figures, 'text_blocks': text_blocks, 'page_text': norm.get('page_text','')}
                        except Exception:
                            llm = None
                    else:
                        llm = None
                if llm is None:
                    flayout.write(json.dumps({'doc_id': pdf_path.name, 'page_index': page_idx, 'page_size':[w,h],
                                              'engine':'llm_fail','figures':[], 'text_blocks':[], 'page_text':''}, ensure_ascii=False) + '\n')
                    seq = [ {'kind':'image_page','source':img_name,'bbox':[0,0,w,h]},
                            {'kind':'text','type':'page_text','bbox':[0,0,w,h],'text':''} ]
                    fjsonl.write(json.dumps({'doc_id': pdf_path.name, 'page_index': page_idx, 'interleaved': seq}, ensure_ascii=False) + '\n')
                    continue

                figures = llm.get('figures', [])
                tbs = llm.get('text_blocks', [])

                def _demote_equation_only_figures(figs: List[Dict], tbs: List[Dict], pw: int, ph: int, *, min_area_ratio: float) -> None:
                    by_id = {f.get('id'): f for f in figs if f.get('id')}
                    to_remove: List[str] = []
                    for fid, f in list(by_id.items()):
                        bb = f.get('bbox') or [0,0,0,0]
                        w = max(1, int(bb[2]-bb[0])); h = max(1, int(bb[3]-bb[1]))
                        h_ratio = h/float(max(1, ph)); w_ratio = w/float(max(1, pw))
                        area_ratio = (w_ratio * h_ratio)
                        refs = [tb for tb in tbs if (tb.get('ref') == fid)]
                        if refs and all((tb.get('role')=='equation') for tb in refs):
                            has_caption = any((tb.get('role')=='caption') for tb in tbs if tb.get('ref')==fid)
                            aspect = (max(w,h)/max(1,min(w,h)))
                            if not has_caption:
                                if not (area_ratio >= 0.01 or aspect >= 1.6):
                                    to_remove.append(fid)
                        elif not refs and area_ratio < float(min_area_ratio):
                            to_remove.append(fid)
                    if not to_remove:
                        return
                    figs[:] = [f for f in figs if f.get('id') not in to_remove]
                    for tb in tbs:
                        if tb.get('ref') in to_remove:
                            tb['ref'] = None
                            if (tb.get('role')=='equation'):
                                tb['absorb'] = 'no'

                if not llm_only:
                    # Step 0: link captions prior to any NMS/remap so refs exist
                    _link_captions(figures, tbs, config=cfg)
                    # Step 1: demote figures that are equations-only/tiny
                    _demote_equation_only_figures(figures, tbs, w, h, min_area_ratio=cfg.min_figure_area_ratio)

                def _nms_figures(figs: List[Dict], iou_thr: float) -> Tuple[List[Dict], Dict[str, str]]:
                    if iou_thr <= 0: return figs[:], {}
                    order = sorted(figs, key=lambda f: (f['bbox'][2]-f['bbox'][0])*(f['bbox'][3]-f['bbox'][1]), reverse=True)
                    kept: List[Dict] = []
                    remap: Dict[str,str] = {}
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

                nms_thr = 0.0 if llm_only else float(fig_nms_iou or 0.0)
                if nms_thr > 0.0:
                    new_figs, remap = _nms_figures(figures, nms_thr)
                    kept_ids = {f.get('id') for f in new_figs if f.get('id')}
                    if remap or kept_ids:
                        for tb in tbs:
                            r = tb.get('ref')
                            if not r:
                                continue
                            if r in remap:
                                tb['ref'] = remap[r]
                            elif r not in kept_ids:
                                best = None; best_score = 1e9
                                bb = tb.get('bbox') or [0,0,0,0]
                                role = tb.get('role','paragraph')
                                for f in new_figs:
                                    sc = _score_tb_to_fig(bb, f['bbox'], role)
                                    if sc < best_score:
                                        best_score, best = sc, f
                                if best is not None and best.get('id'):
                                    tb['ref'] = best.get('id')
                                else:
                                    try:
                                        logdir = out_dir / 'logs'; logdir.mkdir(exist_ok=True, parents=True)
                                        (logdir / f'NMS_warn_page_{page_idx:03d}.txt').write_text(
                                            f'Unmapped ref after NMS: {r}', 'utf-8')
                                    except Exception:
                                        pass
                    figures = new_figs

                orig_figures = [{'id': f.get('id'), 'bbox': list(f['bbox'])} for f in figures]
                absorb_events: List[Dict] = []
                if not llm_only:
                    # Step 2: absorption (single pass, centralized here)
                    absorb_events = _absorb_into_figures(
                        figures, tbs, w, h,
                        strict=bool(strict_capture),
                        return_events=True,
                        allow_small_labels=(not no_small_label_absorb),
                        respect_hints=bool(respect_llm_absorb_hints),
                        config=cfg,
                    ) or []

                # Step 3: optional reading-order fallback before writing results
                try:
                    inv_thr = float(getattr(cfg, 'order_inversion_ratio_thr', 0.30))
                    if not llm_only and len(tbs) >= 3 and inv_thr > 0:
                        def _key(tb):
                            bb = tb.get('bbox') or [0,0,0,0]
                            return (int(bb[1]), int(bb[0]))
                        n = len(tbs)
                        total = n * (n - 1) // 2
                        inv = 0
                        for i in range(n):
                            yi, xi = _key(tbs[i])
                            for j in range(i+1, n):
                                yj, xj = _key(tbs[j])
                                if (yi, xi) > (yj, xj):
                                    inv += 1
                        ratio = (inv / total) if total > 0 else 0.0
                        if ratio >= inv_thr:
                            tbs.sort(key=_key)
                            try:
                                logdir = out_dir / 'logs'; logdir.mkdir(exist_ok=True, parents=True)
                                (logdir / f'order_fallback_page_{page_idx:03d}.txt').write_text(
                                    f'pair_inversions={inv} total_pairs={total} ratio={ratio:.3f} thr={inv_thr}', 'utf-8')
                            except Exception:
                                pass
                except Exception:
                    pass

                # include_figures branch is handled below while building interleaved sequence

                if not llm_only:
                    miss_ref = sum(1 for tb in tbs if (tb.get('role')=='caption' and not tb.get('ref')))
                    eq_overlap = 0
                    for tb in tbs:
                        if tb.get('role')!='equation':
                            continue
                        for f in figures:
                            if _iou(tb['bbox'], f['bbox']) > 0:
                                eq_overlap += 1
                                break
                else:
                    miss_ref = sum(1 for tb in tbs if (tb.get('role')=='caption' and not tb.get('ref')))
                    eq_overlap = 0

                rec = {'doc_id': pdf_path.name, 'page_index': page_idx, 'page_size':[w,h], 'engine':'llm',
                       'figures': figures, 'text_blocks': tbs, 'page_text': llm.get('page_text',''),
                       'violations': {'missing_caption_ref': miss_ref, 'equation_overlap': eq_overlap}}
                flayout.write(json.dumps(rec, ensure_ascii=False) + '\n')

                seq: List[Dict] = []
                seq.append({'kind':'image_page','source':img_name,'bbox':[0,0,w,h]})
                for f in figures:
                    if include_figures:
                        seq.append({'kind':'image_region','role':'figure','source':img_name,'bbox': f['bbox'], 'id': f.get('id')})
                for tb in tbs:
                    item={'kind':'text','type':tb.get('role','paragraph'),'bbox':tb['bbox'],'text':tb.get('text','')}
                    if tb.get('role')=='caption' and tb.get('ref'):
                        item['ref']=tb['ref']
                    seq.append(item)
                fjsonl.write(json.dumps({'doc_id': pdf_path.name, 'page_index': page_idx, 'interleaved': seq}, ensure_ascii=False) + '\n')

                if viz and _HAS_CV2:
                    try:
                        img_bgr = cv2.imread(str(img_path))
                        if img_bgr is not None:
                            dbg = img_bgr.copy()
                            for f in orig_figures:
                                x0, y0, x1, y1 = map(int, f['bbox'])
                                cv2.rectangle(dbg, (x0, y0), (x1, y1), (0, 255, 255), 1, cv2.LINE_AA)
                                if f.get('id'):
                                    cv2.putText(dbg, f"orig:{f['id']}", (x0 + 3, y0 + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 200), 1, cv2.LINE_AA)
                            for f in figures:
                                x0, y0, x1, y1 = map(int, f['bbox'])
                                cv2.rectangle(dbg, (x0, y0), (x1, y1), (0, 0, 255), 2, cv2.LINE_AA)
                                if f.get('id'):
                                    cv2.putText(dbg, f['id'], (x0 + 3, y0 + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                            if viz_text_blocks:
                                for tb in tbs:
                                    x0, y0, x1, y1 = map(int, tb['bbox'])
                                    role = tb.get('role', 'paragraph')
                                    if role == 'caption':
                                        color = (0, 165, 255)
                                    elif role == 'equation':
                                        color = (255, 0, 0)
                                    elif role == 'heading':
                                        color = (255, 0, 255)
                                    else:
                                        color = (180, 180, 180)
                                    cv2.rectangle(dbg, (x0, y0), (x1, y1), color, 1, cv2.LINE_AA)
                            cv2.imwrite(str(out_dir / f'page_{page_idx:03d}_layout_llm.png'), dbg)

                            if (viz_absorb_debug and not llm_only):
                                dbg2 = img_bgr.copy()
                                for f in orig_figures:
                                    x0, y0, x1, y1 = map(int, f['bbox'])
                                    cv2.rectangle(dbg2, (x0, y0), (x1, y1), (0, 255, 255), 1, cv2.LINE_AA)
                                for f in figures:
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
                                for ev in absorb_events:
                                    reason = ev.get('reason')
                                    if reason == 'final_pad':
                                        continue
                                    tb = ev.get('tb_bbox') or []
                                    fig_after = ev.get('fig_bbox_after') or []
                                    c = color_map.get(reason, (255, 255, 255))
                                    if len(tb) >= 4:
                                        cv2.rectangle(dbg2, (tb[0], tb[1]), (tb[2], tb[3]), c, 1, cv2.LINE_AA)
                                    if len(tb) >= 4 and len(fig_after) >= 4:
                                        cx = (tb[0]+tb[2])//2; cy=(tb[1]+tb[3])//2
                                        fx = (fig_after[0]+fig_after[2])//2; fy=(fig_after[1]+fig_after[3])//2
                                        cv2.arrowedLine(dbg2, (cx,cy), (fx,fy), c, 1, cv2.LINE_AA, tipLength=0.2)
                                cv2.imwrite(str(out_dir / f'page_{page_idx:03d}_absorb_debug.png'), dbg2)
                    except Exception:
                        pass
    finally:
        try:
            doc.close()
        except Exception:
            pass
    return inter_path, layout_path
