from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Dict, List, Optional

try:
    import cv2  # type: ignore
    _HAS_CV2 = True
except Exception:
    _HAS_CV2 = False

try:
    from PIL import Image  # type: ignore
    from io import BytesIO
    _HAS_PIL = True
except Exception:
    _HAS_PIL = False

from .common import _scale_kilo_to_px, _extract_json_obj, _normalize_latex_backslashes, _parse_pages_expr


def _png_to_data_uri(img) -> str:
    if _HAS_CV2 and 'numpy' in str(type(img)):
        ok, buf = cv2.imencode('.png', img)
        if not ok:
            return ''
        b = buf.tobytes()
    elif _HAS_PIL and isinstance(img, Image.Image):
        bio = BytesIO(); img.save(bio, format='PNG'); b = bio.getvalue()
    else:
        return ''
    b64 = base64.b64encode(b).decode('ascii')
    return f'data:image/png;base64,{b64}'


def _read_img(path: Path):
    if _HAS_CV2:
        return cv2.imread(str(path))
    if _HAS_PIL:
        try:
            return Image.open(str(path)).convert('RGB')
        except Exception:
            return None
    return None


def _crop(img, bbox: List[int]):
    x0,y0,x1,y1 = map(int, bbox)
    if _HAS_CV2 and 'numpy' in str(type(img)):
        h, w = img.shape[:2]
        x0 = max(0, min(x0, w-1)); x1 = max(0, min(x1, w-1))
        y0 = max(0, min(y0, h-1)); y1 = max(0, min(y1, h-1))
        if x1 <= x0 or y1 <= y0:
            return None
        return img[y0:y1, x0:x1]
    if _HAS_PIL and isinstance(img, Image.Image):
        w, h = img.size
        x0 = max(0, min(x0, w-1)); x1 = max(0, min(x1, w-1))
        y0 = max(0, min(y0, h-1)); y1 = max(0, min(y1, h-1))
        if x1 <= x0 or y1 <= y0:
            return None
        return img.crop((x0, y0, x1, y1))
    return None


def generate_markdown(pdf_path: Path, out_dir: Path, *, embed_full_page: bool=False, pages_expr: Optional[str]=None,
                      md_col_detect: bool = True, md_col_gap_frac: float = 0.08, md_max_cols: int = 3,
                      md_col_min_blocks: int = 6, macro_repair_mode: str = 'off') -> Path:
    layout_path = out_dir / 'structured_layout.jsonl'
    md_path = out_dir / f'{pdf_path.stem}.md'
    if not layout_path.exists():
        raise FileNotFoundError(f'Missing layout file: {layout_path}')

    lines: List[str] = []
    lines.append(f'# {pdf_path.name}')

    recs: List[dict] = []
    with layout_path.open('r', encoding='utf-8') as f:
        for line in f:
            rec = json.loads(line)
            recs.append(rec)

    max_idx = max((int(r.get('page_index', 0)) for r in recs), default=-1)
    page_count = max_idx + 1 if max_idx >= 0 else 0
    sel = set(_parse_pages_expr(pages_expr, page_count)) if pages_expr else None

    for rec in recs:
        if sel is not None and int(rec.get('page_index', 0)) not in sel:
            continue
        w, h = rec['page_size']
        engine = rec.get('engine','')
        lines.append(f"\n## Page {rec['page_index']} ({w}×{h}, {engine})\n")

        page_png = out_dir / f"page_{int(rec['page_index']):03d}.png"
        img = _read_img(page_png) if page_png.exists() else None

        if embed_full_page and img is not None:
            uri = _png_to_data_uri(img)
            if uri:
                lines.append(f'![page {rec["page_index"]}]({uri})\n')

        fig_map: Dict[str, List[int]] = {}
        figures = rec.get('figures') or rec.get('figure_regions') or []
        for idx, f in enumerate(figures):
            bb = f.get('bbox') if isinstance(f, dict) else f
            fid = f.get('id') if isinstance(f, dict) else f'fig_{idx}'
            if isinstance(bb, list) and len(bb) >= 4:
                fig_map[str(fid)] = [int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3])]

        tbs = rec.get('text_blocks') or []

        # Optional: simple column detection by x-center clustering
        if md_col_detect and isinstance(w, int) and len(tbs) >= int(md_col_min_blocks):
            try:
                gap_px = float(md_col_gap_frac) * float(max(1, w))
                # Collect centers of non-caption text blocks to estimate columns
                cents = []
                for tb in tbs:
                    bb = tb.get('bbox') or [0,0,0,0]
                    cx = (int(bb[0]) + int(bb[2])) / 2.0
                    cents.append((cx, tb))
                cents.sort(key=lambda x: x[0])
                # Split where consecutive centers are far apart
                cols: List[List[dict]] = []
                cur: List[dict] = []
                last_cx = None
                for cx, tb in cents:
                    if last_cx is not None and (cx - last_cx) > gap_px and len(cols) + 1 < int(md_max_cols):
                        cols.append(cur); cur = []
                    cur.append(tb)
                    last_cx = cx
                if cur:
                    cols.append(cur)
                if len(cols) >= 2:
                    # Reorder: columns left→right; within each, top→bottom by y0,x0
                    def yx(tb):
                        b = tb.get('bbox') or [0,0,0,0]
                        return (int(b[1]), int(b[0]))
                    ordered: List[dict] = []
                    for col in cols:
                        col.sort(key=yx)
                        ordered.extend(col)
                    # Keep stable relative order by creating a map and reassembling
                    tbs = ordered
            except Exception:
                pass
        # track any additional normalization changes at markdown stage (all roles)
        md_changes: List[dict] = []
        for tb in tbs:
            role = tb.get('role','paragraph')
            text = tb.get('text','')
            before = text
            # normalize/repair also for inline math inside paragraphs/headings
            if role == 'equation':
                text = _normalize_latex_backslashes(text, 'equation', macro_mode=macro_repair_mode)
            else:
                text = _normalize_latex_backslashes(text, 'paragraph', macro_mode=macro_repair_mode)
            if before != text:
                md_changes.append({'index': None, 'role': role, 'before': before, 'after': text})
            lines.append('')
            if role == 'heading':
                lines.append(f'### {text}')
            elif role == 'equation':
                lines.append(text)
            elif role == 'caption' and tb.get('ref') in fig_map and img is not None:
                crop = _crop(img, fig_map[tb.get('ref')])
                if crop is not None:
                    uri = _png_to_data_uri(crop)
                    if uri:
                        lines.append(f'![{tb.get("ref")}]({uri})')
                lines.append(text)
            else:
                lines.append(text)

        # write page-level markdown escape change logs if any
        try:
            if md_changes:
                logdir = out_dir / 'logs'; logdir.mkdir(exist_ok=True, parents=True)
                (logdir / f'MD_text_escape_changes_page_{int(rec["page_index"]):03d}.json').write_text(
                    json.dumps(md_changes, ensure_ascii=False, indent=2), 'utf-8')
                # also write unified diffs for readability
                try:
                    import difflib
                    diffs = []
                    for ch in md_changes:
                        before = str(ch.get('before',''))
                        after = str(ch.get('after',''))
                        diff = difflib.unified_diff(
                            before.splitlines(keepends=True),
                            after.splitlines(keepends=True),
                            fromfile='md.before', tofile='md.after'
                        )
                        diffs.append('# role=' + str(ch.get('role')) + '\n')
                        diffs.append(''.join(diff))
                        diffs.append('\n')
                    (logdir / f'MD_text_escape_changes_page_{int(rec["page_index"]):03d}.txt').write_text(''.join(diffs), 'utf-8')
                except Exception:
                    pass
        except Exception:
            pass

    md_path.write_text('\n'.join(lines), 'utf-8')
    return md_path
