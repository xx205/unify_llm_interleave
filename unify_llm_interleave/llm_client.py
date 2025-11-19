from __future__ import annotations

import base64
import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests

from .common import (
    LayoutConfig,
    MAX_LLM_SIDE,
    DEFAULT_JPEG_QUALITY,
    _extract_json_obj,
    _normalize_latex_backslashes,
    _purge_ctrl,
    _scale_kilo_to_px,
    get_last_extract_meta,
    _bytes_to_data_url,
    _pix_to_encoded
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
    return {'figures': figures, 'text_blocks': text_blocks, 'page_text': norm.get('page_text','')}
