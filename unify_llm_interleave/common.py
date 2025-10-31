from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Set
import threading

# -------------- Tunables / Defaults --------------
MAX_LLM_SIDE = 1800
DEFAULT_JPEG_QUALITY = 85

# Precompiled patterns for LaTeX normalization
_LATEX_MULTILINE_MARKERS = (
    '\\begin{', 'aligned', 'align', 'cases', 'matrix', 'pmatrix', 'bmatrix', 'vmatrix', 'Vmatrix',
    'array', 'tabular', 'eqnarray', 'split', 'gather', 'multline', 'flalign'
)
_LATEX_COLLAPSE_RE = re.compile(r'\\{2,}(?=[^ \*\[\n%\\])')


@dataclass
class LayoutConfig:
    absorb_marg_x: float = 0.03
    absorb_marg_y: float = 0.05
    absorb_final_expand: float = 0.015
    small_label_w_ratio: float = 0.30
    small_label_w_ratio_strict: float = 0.20
    small_label_h_ratio: float = 0.022
    small_label_h_ratio_strict: float = 0.018
    min_figure_area_ratio: float = 0.0025
    # caption/link thresholds
    link_min_overlap_ratio: float = 0.15
    link_min_iou: float = 0.02
    # reading-order fallback
    order_inversion_ratio_thr: float = 0.30
    # ambiguous assignment gap threshold as fraction of page diagonal
    ambiguous_gap_frac: float = 0.01
    # math macro repair inside math environments: 'off' | 'moderate' | 'aggressive'
    macro_repair_mode: str = 'off'

    @staticmethod
    def from_globals() -> 'LayoutConfig':
        # preserve previous behavior of reading module-level constants if mutated
        return LayoutConfig()


_EXTRACT_TL = threading.local()

def _set_last_extract_meta(meta: Dict) -> None:
    try:
        setattr(_EXTRACT_TL, 'last_meta', dict(meta))
    except Exception:
        pass

def get_last_extract_meta(clear: bool = True) -> Optional[Dict]:
    """Return metadata recorded during the most recent _extract_json_obj call.
    Includes which strategy succeeded and whether invalid escapes were fixed.
    { 'case': 'direct'|'fenced'|'substring', 'used_fix_invalid_escapes': bool,
      'original_fragment': str|None, 'sanitized_fragment': str|None }
    """
    m = getattr(_EXTRACT_TL, 'last_meta', None)
    if clear:
        try:
            delattr(_EXTRACT_TL, 'last_meta')
        except Exception:
            pass
    return m


def _extract_json_obj(text: str) -> Optional[dict]:
    """Best-effort JSON extraction with selective backslash repair (LaTeX tolerant)."""
    def try_load(s: str) -> Optional[dict]:
        try:
            return json.loads(s)
        except Exception:
            return None

    def fix_invalid_escapes(s: str) -> str:
        import re as _re
        # Only double backslashes that are not starting a valid JSON escape
        return _re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', s)

    def escape_literal_ctrls_in_strings(s: str) -> str:
        # Escape raw control chars inside JSON string literals: \n, \r, \t, \b, \f
        out = []
        in_str = False
        esc = False
        for ch in s:
            if esc:
                out.append(ch)
                esc = False
                continue
            if ch == '\\':
                out.append(ch)
                esc = True
                continue
            if ch == '"':
                out.append(ch)
                in_str = not in_str
                continue
            if in_str:
                if ch == '\n':
                    out.append('\\n'); continue
                if ch == '\r':
                    out.append('\\r'); continue
                if ch == '\t':
                    out.append('\\t'); continue
                if ch == '\x08':
                    out.append('\\b'); continue
                if ch == '\x0c':
                    out.append('\\f'); continue
            out.append(ch)
        return ''.join(out)

    def escape_unescaped_quotes_in_text_fields(s: str) -> str:
        import re as _re
        token = '"text": "'
        i = 0
        L = len(s)
        out = []
        # heuristic: what can follow the closing quote of text field
        tail_pat = _re.compile(r'^\s*,\s*"(?:absorb|ref|bbox|role|id|type|kind|page_text|engine|figures|text_blocks|violations|page_size)"')
        while i < L:
            j = s.find(token, i)
            if j < 0:
                out.append(s[i:]); break
            out.append(s[i:j])
            out.append(token)
            k = j + len(token)
            # scan until the closing quote we believe ends the field
            while k < L:
                ch = s[k]
                if ch == '\\':
                    # keep escape + next char as-is
                    out.append(s[k:k+2]); k += 2; continue
                if ch == '"':
                    # lookahead to decide if this ends the field
                    tail = s[k+1:k+1+160]
                    if tail_pat.search(tail) is not None:
                        out.append('"'); k += 1
                        break
                    # otherwise, quote is inside payload → escape it
                    out.append('\\"'); k += 1; continue
                # escape raw controls as we go (safety)
                if ch == '\n': out.append('\\n'); k += 1; continue
                if ch == '\r': out.append('\\r'); k += 1; continue
                if ch == '\t': out.append('\\t'); k += 1; continue
                if ch == '\x08': out.append('\\b'); k += 1; continue
                if ch == '\x0c': out.append('\\f'); k += 1; continue
                out.append(ch); k += 1
            i = k
        return ''.join(out)

    # direct
    o = try_load(text)
    if o is not None:
        _set_last_extract_meta({'case': 'direct', 'used_fix_invalid_escapes': False,
                                'original_fragment': None, 'sanitized_fragment': None})
        return o
    # direct + escape string controls
    t_esc = escape_literal_ctrls_in_strings(text)
    o = try_load(t_esc)
    if o is not None:
        _set_last_extract_meta({'case': 'direct+esc', 'used_fix_invalid_escapes': False,
                                'original_fragment': text, 'sanitized_fragment': t_esc})
        return o
    # fenced (support ```json ... ``` and plain ``` ... ```)
    if '```' in text:
        import re as _re
        for m in _re.finditer(r'```[A-Za-z]*\s*([\s\S]*?)```', text):
            s = (m.group(1) or '').strip()
            if s.startswith('{') and s.endswith('}'):
                o0 = try_load(s)
                if o0 is not None:
                    _set_last_extract_meta({'case': 'fenced', 'used_fix_invalid_escapes': False,
                                            'original_fragment': s, 'sanitized_fragment': None})
                    return o0
                s_esc = escape_literal_ctrls_in_strings(s)
                o_esc = try_load(s_esc)
                if o_esc is not None:
                    _set_last_extract_meta({'case': 'fenced+esc', 'used_fix_invalid_escapes': False,
                                            'original_fragment': s, 'sanitized_fragment': s_esc})
                    return o_esc
                s2 = fix_invalid_escapes(s)
                o1 = try_load(s2)
                if o1 is not None:
                    _set_last_extract_meta({'case': 'fenced', 'used_fix_invalid_escapes': True,
                                            'original_fragment': s, 'sanitized_fragment': s2})
                    return o1
                s3 = escape_unescaped_quotes_in_text_fields(s)
                o3 = try_load(s3)
                if o3 is not None:
                    _set_last_extract_meta({'case': 'fenced+qesc', 'used_fix_invalid_escapes': False,
                                            'original_fragment': s, 'sanitized_fragment': s3})
                    return o3
                s4 = escape_unescaped_quotes_in_text_fields(s2)
                o4 = try_load(s4)
                if o4 is not None:
                    _set_last_extract_meta({'case': 'fenced+qesc', 'used_fix_invalid_escapes': True,
                                            'original_fragment': s, 'sanitized_fragment': s4})
                    return o4
                s2_esc = escape_literal_ctrls_in_strings(s2)
                o2 = try_load(s2_esc)
                if o2 is not None:
                    _set_last_extract_meta({'case': 'fenced+esc', 'used_fix_invalid_escapes': True,
                                            'original_fragment': s, 'sanitized_fragment': s2_esc})
                    return o2
    # substring
    i = text.find('{'); j = text.rfind('}')
    if i != -1 and j != -1 and j > i:
        s = text[i:j+1]
        o0 = try_load(s)
        if o0 is not None:
            _set_last_extract_meta({'case': 'substring', 'used_fix_invalid_escapes': False,
                                    'original_fragment': s, 'sanitized_fragment': None})
            return o0
        s_esc = escape_literal_ctrls_in_strings(s)
        o_esc = try_load(s_esc)
        if o_esc is not None:
            _set_last_extract_meta({'case': 'substring+esc', 'used_fix_invalid_escapes': False,
                                    'original_fragment': s, 'sanitized_fragment': s_esc})
            return o_esc
        s2 = fix_invalid_escapes(s)
        o1 = try_load(s2)
        if o1 is not None:
            _set_last_extract_meta({'case': 'substring', 'used_fix_invalid_escapes': True,
                                    'original_fragment': s, 'sanitized_fragment': s2})
            return o1
        s3 = escape_unescaped_quotes_in_text_fields(s)
        o3 = try_load(s3)
        if o3 is not None:
            _set_last_extract_meta({'case': 'substring+qesc', 'used_fix_invalid_escapes': False,
                                    'original_fragment': s, 'sanitized_fragment': s3})
            return o3
        s4 = escape_unescaped_quotes_in_text_fields(s2)
        o4 = try_load(s4)
        if o4 is not None:
            _set_last_extract_meta({'case': 'substring+qesc', 'used_fix_invalid_escapes': True,
                                    'original_fragment': s, 'sanitized_fragment': s4})
            return o4
        s2_esc = escape_literal_ctrls_in_strings(s2)
        o2 = try_load(s2_esc)
        if o2 is not None:
            _set_last_extract_meta({'case': 'substring+esc', 'used_fix_invalid_escapes': True,
                                    'original_fragment': s, 'sanitized_fragment': s2_esc})
            return o2
    return None


def _normalize_latex_backslashes(text: str, role: str, *, macro_mode: str = 'off') -> str:
    """Collapse over-produced backslashes in LaTeX where safe.
    Rule: In math, multiple backslashes not starting a valid linebreak sequence (\\, \\*, \\[len], \\ +space/\n/%/\\) can be collapsed to a single backslash before macros/letters.
    Apply conservatively:
      - For display 'equation' blocks: operate inside $$...$$ if present; skip if multi-line environments are detected.
      - For paragraphs/headings: operate only inside inline math $...$ spans.
    """
    if not isinstance(text, str) or '\\' not in text:
        return text

    # Heuristic: if multi-line/math environments likely present, avoid touching \\\ newlines
    has_multiline_env = any(m in text for m in _LATEX_MULTILINE_MARKERS)

    def collapse_segment(seg: str) -> str:
        if not seg or '\\' not in seg:
            return seg
        if has_multiline_env:
            # be extra conservative inside multiline envs — do not alter
            return seg
        return _LATEX_COLLAPSE_RE.sub(r'\\', seg)

    def repair_macros(seg: str, mode: str) -> str:
        if not seg or mode == 'off':
            return seg
        # Build whitelist per mode
        SYMBOLS: Set[str] = {
            'times','cdot','pm','mp','leq','geq','neq','approx','sim','cong','to','infty','partial','nabla',
            'circ','star','oplus','otimes','cup','cap','setminus','subset','supset','subseteq','supseteq',
            'in','notin','exists','forall','land','lor','neg','wedge','vee',
            'uparrow','downarrow','leftarrow','rightarrow','Leftarrow','Rightarrow','leftrightarrow','Leftrightarrow',
            'mapsto','implies','iff','triangle','triangleq','perp','angle','prod','sum'
        }
        FUNCTIONS: Set[str] = {
            'sin','cos','tan','cot','sec','csc','arcsin','arccos','arctan','sinh','cosh','tanh','log','ln','exp',
            'max','min','sup','inf','lim','det','dim','gcd','lcm','Pr','arg','argmax','argmin','mod','bmod'
        }
        FRACTIONS_FONTS: Set[str] = {
            'frac','dfrac','tfrac','sqrt','root','boldsymbol','mathbf','mathrm','mathsf','mathtt','mathbb','mathcal','mathfrak',
            'text','textrm','textbf','textit','texttt','textsc','textsf','operatorname','overline','underline','vec','tilde','widehat','hat','bar','dot','ddot','overbrace','underbrace','color','phantom','vphantom','hphantom',
            'left','right','big','Big','bigg','Bigg'
        }
        GREEK: Set[str] = {
            # lowercase
            'alpha','beta','gamma','delta','epsilon','zeta','eta','theta','iota','kappa','lambda','mu','nu','xi','pi','rho','sigma','tau','upsilon','phi','chi','psi','omega',
            'varepsilon','vartheta','varpi','varphi','varrho','varsigma',
            # uppercase
            'Gamma','Delta','Theta','Lambda','Xi','Pi','Sigma','Upsilon','Phi','Psi','Omega'
        }
        WH: Set[str] = set()
        WH |= SYMBOLS | FUNCTIONS | FRACTIONS_FONTS
        if mode == 'aggressive':
            WH |= GREEK
        if not WH:
            return seg
        # Extend macro whitelist with additional common macros
        FRACTIONS_FONTS |= {
            'binom', 'dbinom', 'tbinom', 'mathscr', 'widebar', 'operatorname*'
        }
        # Sort by length desc to avoid partial matches (e.g., 'sin' within 'sinh')
        names = sorted(WH, key=len, reverse=True)
        import re as _re
        # word-ish macros: ensure not preceded by backslash or letter; allow following '_'/'^' for subscripts/superscripts
        pat = r'(?<!\\)(?<![A-Za-z])(?P<name>' + '|'.join(_re.escape(n) for n in names if n not in {'left','right','big','Big','bigg','Bigg'}) + r')(?![A-Za-z0-9])'
        def repl(m: 're.Match[str]') -> str:
            return '\\' + m.group('name')
        seg2 = _re.sub(pat, repl, seg)
        # handle left/right/big* which should be followed by delimiters or '.'; add only if plausible
        pat_lr = r'(?<!\\)(?<![A-Za-z])(?P<name>left|right)\s*(?=[\(\)\[\]\{\}\|<>\.]|\\langle|\\rangle)'
        seg3 = _re.sub(pat_lr, lambda m: '\\' + m.group('name'), seg2)
        pat_big = r'(?<!\\)(?<![A-Za-z])(?P<name>big|Big|bigg|Bigg)\s*(?=[\(\)\[\]\{\}\|<>])'
        seg4 = _re.sub(pat_big, lambda m: '\\' + m.group('name'), seg3)
        # Additionally repair cases where JSON ate the first letter via control codes (\b,\f,\n,\r,\t)
        CTRL = r'[\x08\x0c\x09\x0a\x0d]'
        def _suffix_group(st: Set[str], initial: str) -> str:
            items = [n[1:] for n in st if n.startswith(initial) and len(n) > 1]
            if not items:
                return ''
            return '(?:' + '|'.join(_re.escape(s) for s in sorted(items, key=len, reverse=True)) + ')'
        def _initials(st: Set[str]) -> Set[str]:
            return {n[0] for n in st if n}
        # FRACTIONS_FONTS require a brace following
        def _strip_ctrl_prefix(s: str) -> str:
            # remove leading control + optional stray hyphens/zero-width spaces
            return _re.sub(r'^[\x08\x0c\x09\x0a\x0d][\s\-\u00ad\u200b\u200c\u200d]*', '', s)
        for init in sorted(_initials(FRACTIONS_FONTS)):
            suf = _suffix_group(FRACTIONS_FONTS, init)
            if suf:
                pat = _re.compile(CTRL + r'[\s\-\u00ad\u200b\u200c\u200d]*' + suf + r'(?=\s*\{)')
                seg4 = pat.sub(lambda m, i=init: '\\' + i + _strip_ctrl_prefix(m.group(0)), seg4)
        # FUNCTIONS require next '('
        for init in sorted(_initials(FUNCTIONS)):
            suf = _suffix_group(FUNCTIONS, init)
            if suf:
                pat = _re.compile(CTRL + r'[\s\-\u00ad\u200b\u200c\u200d]*' + suf + r'(?=\s*\()')
                seg4 = pat.sub(lambda m, i=init: '\\' + i + _strip_ctrl_prefix(m.group(0)), seg4)
        # SYMBOLS: word boundary
        for init in sorted(_initials(SYMBOLS)):
            suf = _suffix_group(SYMBOLS, init)
            if suf:
                pat = _re.compile(CTRL + r'[\s\-\u00ad\u200b\u200c\u200d]*' + suf + r'(?![A-Za-z0-9_])')
                seg4 = pat.sub(lambda m, i=init: '\\' + i + _strip_ctrl_prefix(m.group(0)), seg4)
        # GREEK (aggressive only): word boundary
        if mode == 'aggressive':
            for init in sorted(_initials(GREEK)):
                suf = _suffix_group(GREEK, init)
                if suf:
                    pat = _re.compile(CTRL + r'\s*' + suf + r'(?![A-Za-z0-9_])')
                    seg4 = pat.sub(lambda m, i=init: '\\' + i + _strip_ctrl_prefix(m.group(0)), seg4)
        # Canonicalize a few invented macros occasionally produced by LLMs
        try:
            seg5 = seg4
            seg5 = _re.sub(r'(?<!\\)bigsqrt\s*(?=\{)', r'\\sqrt', seg5)
            seg5 = _re.sub(r'(?<!\\)bigunderline\s*(?=\{)', r'\\underline', seg5)
            seg5 = _re.sub(r'(?<!\\)bigquad\b', r'\\quad', seg5)
            seg5 = _re.sub(r'(?<!\\)bigspace\b', r'\\quad', seg5)
            # Repair common cases where the first letter vanished without leaving a control code
            seg5 = _re.sub(r'(?<![A-Za-z\\])imes(?![A-Za-z0-9_])', r'\\times', seg5)  # times without leading 't'
            seg5 = _re.sub(r'(?<![A-Za-z\\])rac(?=\s*\{)', r'\\frac', seg5)         # frac without leading 'f'
            seg5 = _re.sub(r'(?<![A-Za-z\\])ext(?=\s*\{)', r'\\text', seg5)         # text without leading 't'
            return seg5
        except Exception:
            return seg4

    # Helper to process delimited spans
    def collapse_in_delims(s: str, open_delim: str, close_delim: str) -> str:
        out = []
        i = 0
        L = len(s)
        while i < L:
            j = s.find(open_delim, i)
            if j < 0:
                out.append(s[i:])
                break
            # keep prefix
            out.append(s[i:j])
            k = s.find(close_delim, j + len(open_delim))
            if k < 0:
                # no closing; bail out
                out.append(s[j:])
                break
            # inside math
            inner = s[j + len(open_delim):k]
            out.append(open_delim)
            collapsed = collapse_segment(inner)
            repaired = repair_macros(collapsed, macro_mode)
            out.append(repaired)
            out.append(close_delim)
            i = k + len(close_delim)
        return ''.join(out)

    if role == 'equation':
        # operate inside $$...$$ if present; otherwise whole text
        if '$$' in text:
            s = collapse_in_delims(text, '$$', '$$')
            return s
        s0 = collapse_segment(text)
        return repair_macros(s0, macro_mode)
    else:
        # collapse inside inline $...$ spans only
        if '$' in text:
            # first process $$...$$ to avoid interfering with inline
            s = collapse_in_delims(text, '$$', '$$') if '$$' in text else text
            s2 = collapse_in_delims(s, '$', '$')
            return s2
        return text


_CTRL_MAP = {
    "\x08": "\\b",  # backspace → visible escape
    "\x0c": "\\f",  # formfeed → visible escape
}


def _purge_ctrl(s: str) -> str:
    if not isinstance(s, str):
        return s
    # First, map rare controls to visible escapes
    for k, v in _CTRL_MAP.items():
        if k in s:
            s = s.replace(k, v)
    # Normalize common controls after macro repair completed upstream
    # TAB → single space; CR → removed; LF 保留
    if "\x09" in s:  # TAB
        s = s.replace("\x09", " ")
    if "\x0d" in s:  # CR
        s = s.replace("\x0d", "")
    return s


def _scale_kilo_to_px(bb: List[int], w: int, h: int) -> List[int]:
    x0, y0, x1, y1 = bb
    X0 = max(0, min(w - 1, int(round(x0 / 1000.0 * w))))
    X1 = max(0, min(w - 1, int(round(x1 / 1000.0 * w))))
    Y0 = max(0, min(h - 1, int(round(y0 / 1000.0 * h))))
    Y1 = max(0, min(h - 1, int(round(y1 / 1000.0 * h))))
    x0, x1 = min(X0, X1), max(X0, X1)
    y0, y1 = min(Y0, Y1), max(Y0, Y1)
    return [x0, y0, x1, y1]


def _parse_pages_expr(expr: Optional[str], page_count: int) -> List[int]:
    """Parse a pages expression like "0,5-8" into a sorted unique list of 0-based page indices."""
    if page_count <= 0:
        return []
    if not expr:
        return list(range(page_count))
    expr = str(expr).strip()
    if expr.lower() in ('all', '*'):
        return list(range(page_count))
    # normalize separators (support Chinese comma)
    expr = expr.replace('，', ',')
    parts = [p.strip() for p in expr.split(',') if p.strip()]
    out: List[int] = []

    def _add_range(a: int, b: int):
        lo = max(0, min(a, b))
        hi = min(page_count - 1, max(a, b))
        out.extend(range(lo, hi + 1))

    for tok in parts:
        if '-' in tok:
            a_str, b_str = tok.split('-', 1)
            a = 0 if a_str == '' else int(a_str)
            b = (page_count - 1) if b_str == '' else int(b_str)
            _add_range(a, b)
        else:
            try:
                idx = int(tok)
            except ValueError:
                continue
            idx = max(0, min(page_count - 1, idx))
            out.append(idx)
    # unique & sorted
    return sorted(dict.fromkeys(out))


def _iou(a: List[int], b: List[int]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    iw = max(0, ix1 - ix0)
    ih = max(0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    aa = max(1, (ax1 - ax0) * (ay1 - ay0))
    ba = max(1, (bx1 - bx0) * (by1 - by0))
    return inter / float(aa + ba - inter)


def _score_tb_to_fig(tb_bb: List[int], fig_bb: List[int], role: str) -> float:
    """Lower is better. Combines vertical gap, center distance, and IoU reward."""
    bx0, by0, bx1, by1 = tb_bb
    fx0, fy0, fx1, fy1 = fig_bb
    bcx = (bx0 + bx1) / 2.0
    bcy = (by0 + by1) / 2.0
    fcx = (fx0 + fx1) / 2.0
    fcy = (fy0 + fy1) / 2.0
    vert_gap = max(0.0, by0 - fy1, fy0 - by1)
    center_dist = abs(bcx - fcx) + 0.5 * abs(bcy - fcy)
    w_vert = 1.0 if role in ('caption', 'heading') else 0.8
    w_center = 0.3
    from .common import _iou as _iou_local  # avoid circular names when imported *
    return w_vert * vert_gap + w_center * center_dist - 20.0 * _iou_local(tb_bb, fig_bb)
