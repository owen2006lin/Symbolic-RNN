"""
latex_cleaner.py

Cleans raw LaTeX equation records from TeXExpressions JSONL files into
forms parseable by sympy.parsing.latex.parse_latex.

Pipeline per candidate string (Stages 5-9 run by clean_to_expr):
  Stage 1  – Strip outer environment wrapper
  Stage 2  – Remove metadata markers (\\label, \\tag, \\nonumber, ...)
  Stage 3  – Split multi-line environments on \\\\ into individual lines
  Stage 4  – Extract RHS (split on bare =, depth-aware; take last segment)
  Stage 5  – Pre-filter banned constructs (integrals, sums, \\cdot, ...)
  Stage 6  – Substitute / normalise (strip decorators, \\left/\\right, spacing)
  Stage 6.5– Strip ambiguous Greek whole-wrappers (e.g. \\sigma(x+y) → x+y)
  Stage 7  – Reject unknown \\commands not in whitelist
  Stage 8  – Try sympy.parsing.latex.parse_latex
  Stage 9  – Post-filter trivially degenerate results (bare symbol / number)

Greek-wrapper rule (Stage 6.5)
  After normalisation, if the entire cleaned string is of the form
  \\greek(...) where the parentheses span the whole string, treat the Greek
  letter as a function-like wrapper and keep only the inside.  This is a
  conservative salvage rule, NOT semantic preservation.  Applied in a loop
  until no more outermost wrappers remain.  The ambiguous Greek set is
  _AMBIGUOUS_GREEK (edit that frozenset to extend/restrict the rule).

Operator rules (from treebuilding.ipynb):
  Allowed unary : sin cos tan cot sec csc arcsin arccos arctan
                  sinh cosh tanh  exp log ln  sqrt (→ x^½)  abs (|...|)
  Allowed binary: + - * /   (\\frac handled natively by parse_latex)
  Subtraction   : treated as +(a, -b) at tree-build time; kept as-is here
  Derivatives   : treated as variables (\\partial filtered out)
  Filtered ops  : \\int \\sum \\prod \\partial \\nabla \\cdot \\times
                  \\min \\max \\lim \\arg  and any custom macro

Public API
----------
clean_to_expr(candidate) -> (ok, cleaned, expr_or_None, reason)
    Returns a live SymPy expression object.  Preferred for in-memory pipelines.
clean_candidate(candidate) -> (ok, cleaned, sympy_str, reason)
    Backward-compatible wrapper that stringifies the SymPy expression.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

from sympy.parsing.latex import parse_latex
from sympy import Number, Symbol, Derivative


# ── Stage 1: Strip outer environment wrapper ─────────────────────────────────

_ENV_RE = re.compile(r'\\begin\{[^}]+\}(.*?)\\end\{[^}]+\}', re.DOTALL)


def strip_environment(s: str) -> str:
    """Remove \\begin{...}...\\end{...}, \\[...\\], and $$...$$ wrappers."""
    m = _ENV_RE.search(s)
    if m:
        return m.group(1)
    # \[...\]
    s = re.sub(r'^\s*\\\[', '', s)
    s = re.sub(r'\\\]\s*$', '', s)
    # $$...$$
    s = re.sub(r'^\s*\$\$', '', s)
    s = re.sub(r'\$\$\s*$', '', s)
    return s.strip()


# ── Stage 2: Remove metadata markers ─────────────────────────────────────────

_METADATA_RES = [
    re.compile(r'\\label\{[^}]*\}'),
    re.compile(r'\\tag\*?\{[^}]*\}'),
    re.compile(r'\\nonumber\b'),
    re.compile(r'\\notag\b'),
    re.compile(r'\\eqnr\b'),
    re.compile(r'\\eqnumber\b'),
]


def remove_metadata(s: str) -> str:
    for pat in _METADATA_RES:
        s = pat.sub('', s)
    return s


# ── Stage 3: Split multi-line body into individual lines ─────────────────────

def split_lines(s: str) -> list[str]:
    """Split on \\\\\\\\ (LaTeX line break), but not \\\\[ (spacing command)."""
    parts = re.split(r'\\\\(?!\[)', s)
    return [p.strip() for p in parts if p.strip()]


# ── Stage 4: Extract RHS ──────────────────────────────────────────────────────

def _split_on_bare_equals(s: str) -> list[str]:
    """
    Split s on '=' characters that are NOT inside braces {}.
    Handles continuation lines like '&= expr' after & is removed.
    """
    parts: list[str] = []
    depth = 0
    buf: list[str] = []
    for ch in s:
        if ch == '{':
            depth += 1
            buf.append(ch)
        elif ch == '}':
            depth = max(0, depth - 1)
            buf.append(ch)
        elif ch == '=' and depth == 0:
            parts.append(''.join(buf))
            buf = []
        else:
            buf.append(ch)
    if buf:
        parts.append(''.join(buf))
    return parts


def extract_rhs(line: str) -> Optional[str]:
    """
    Strip & alignment markers, split on bare =, return the last (RHS) segment.
    Returns None if no = is found (e.g. pure inequality lines).
    """
    line = line.replace('&', '')
    parts = _split_on_bare_equals(line)
    if len(parts) < 2:
        return None
    rhs = parts[-1].strip()
    return rhs if rhs else None


# ── Stage 5: Pre-filter banned constructs ─────────────────────────────────────

# Any match → reject the candidate immediately (unless it is the sole construct,
# in which case _strip_single_banned attempts to extract the inner expression).
# NOTE: \partial is intentionally NOT banned — derivatives are treated as
# variables per the treebuilding spec.
_BANNED_RE = re.compile(
    r'\\(?:'
    r'int[oi]?'                          # \int, \into, \inti
    r'|oint|iint|iiint'                  # multiple integrals
    r'|nabla'                            # gradient / divergence (vector op)
    r'|sum'                              # summation
    r'|prod|coprod'                      # (co)product
    r'|bigcup|bigcap|bigvee|bigwedge'   # big set / logic ops
    r'|lim(?:sup|inf)?(?=[^a-z]|$)'     # \lim, \limsup, \liminf
    r'|inf(?!ty)(?=[^a-z]|$)'           # \inf (infimum) but NOT \infty
    r'|sup(?=[^a-z]|$)'                 # \sup (supremum)
    r'|arg(?:min|max)?(?=[^a-z]|$)'     # \arg, \argmin, \argmax
    r'|min(?=[^a-z]|$)'                 # \min
    r'|max(?=[^a-z]|$)'                 # \max
    r'|cdot'                             # dot product (filtered per user)
    r'|times'                            # cross product
    r'|otimes|oplus|wedge|vee'          # tensor / logic ops
    r'|begin|end'                        # nested environments
    r'|text|mbox|textrm|textit|textbf|textsc|intertext'
    r'|hline|cline|multicolumn|multirow'
    r'|forall|exists'                    # quantifiers
    r'|cdots|ldots|vdots|ddots'         # ellipsis (typically in sums)
    r'|overset|underset|stackrel'       # over/under decorators
    r'|overbrace|underbrace'
    r'|operatorname'                     # custom operators
    r'|pmod|bmod'                        # modulo
    r'|[|]'                              # \| = norm delimiter
    r')',
)


def _skip_scripts(s: str, i: int) -> int:
    """
    Starting at position i in s, skip any run of _ or ^ followed by a
    brace-enclosed group {…} or a single character. Returns the new position.
    Used to consume the sub/superscript decorators attached to a banned command
    so we can isolate the body that follows.
    """
    n = len(s)
    while i < n:
        # skip whitespace
        while i < n and s[i] == ' ':
            i += 1
        if i >= n or s[i] not in ('_', '^'):
            break
        i += 1  # consume _ or ^
        # skip whitespace
        while i < n and s[i] == ' ':
            i += 1
        if i < n and s[i] == '{':
            # consume brace-enclosed group
            depth = 0
            while i < n:
                if s[i] == '{':
                    depth += 1
                elif s[i] == '}':
                    depth -= 1
                    if depth == 0:
                        i += 1
                        break
                i += 1
        elif i < n:
            # single-character script (e.g. ^2 or _n)
            i += 1
    return i


def _strip_single_banned(s: str) -> Optional[str]:
    """
    If s consists of exactly ONE banned construct (possibly with sub/sup
    scripts) wrapping a subexpression, return that subexpression.
    Returns None if there are 0 or 2+ banned constructs, or if there is
    any content before the banned token.

    Examples:
      '\\sum_{i=1}^{n} (a+b)'  →  '(a+b)'
      '\\nabla^2 \\phi'         →  '\\phi'
      '\\sum(a+b) + \\nabla(c)' →  None   (two banned constructs)
      'f + \\sum(g)'            →  None   (content before the construct)
    """
    matches = list(_BANNED_RE.finditer(s))
    if len(matches) != 1:
        return None

    m = matches[0]
    # Reject if there is any non-whitespace content before the banned token
    if s[:m.start()].strip():
        return None

    # Skip past subscripts/superscripts that belong to the banned command
    body_start = _skip_scripts(s, m.end())
    body = s[body_start:].strip()
    return body if body else None


def has_banned_construct(s: str) -> tuple[bool, str]:
    """Returns (True, matched_token) if s contains a banned construct."""
    m = _BANNED_RE.search(s)
    if m:
        return True, m.group(0)
    return False, ''


# ── Stage 6: Normalisation / substitution ─────────────────────────────────────

# Decorator commands: \cmd{content} → content  (strip the wrapper, keep the symbol)
_DECORATOR_CMDS = (
    'mathbf', 'mathcal', 'mathrm', 'mathit', 'mathbb', 'mathsf', 'mathtt',
    'boldsymbol', 'bm',
    'hat', 'tilde', 'bar', 'vec', 'dot', 'ddot',
    'widehat', 'widetilde', 'overline', 'underline',
    'acute', 'grave', 'breve', 'check', 'ring',
)


def _strip_decorators(s: str) -> str:
    """
    Replace \\cmd{content} → content for all decorator commands.
    Applies three passes to handle nested decorators like \\hat{\\mathbf{x}}.
    """
    # Build one big alternation pattern for efficiency
    pattern = re.compile(
        r'\\(?:' + '|'.join(_DECORATOR_CMDS) + r')\{([^{}]*)\}'
    )
    for _ in range(3):
        s_new = pattern.sub(r'\1', s)
        if s_new == s:
            break
        s = s_new

    # Also handle old-style font groups: {\bf content} → content
    s = re.sub(r'\{\\(?:bf|it|rm|sf|tt|cal)\s+([^{}]*)\}', r'\1', s)
    return s


def substitute_latex(s: str) -> str:
    """
    Apply safe normalisation substitutions:
      - strip decorators
      - remove spacing commands
      - expand \\left/\\right delimiters
      - strip display-style commands
      - remove trailing punctuation
    """
    s = _strip_decorators(s)

    # Spacing → single space (then collapsed below)
    s = re.sub(r'\\[,;:!]', ' ', s)
    s = re.sub(r'\\(?:quad|qquad)\b', ' ', s)
    s = re.sub(r'\\(?:hspace|vspace)\{[^}]*\}', ' ', s)

    # Display-style annotations → remove
    s = re.sub(r'\\(?:displaystyle|textstyle|scriptstyle|scriptscriptstyle)\b', '', s)

    # \left( → (   \right) → )
    s = re.sub(r'\\left\s*\(', '(', s)
    s = re.sub(r'\\right\s*\)', ')', s)
    # \left[ → (   \right] → )
    s = re.sub(r'\\left\s*\[', '(', s)
    s = re.sub(r'\\right\s*\]', ')', s)
    # \left| → |   \right| → |
    s = re.sub(r'\\left\s*\|', '|', s)
    s = re.sub(r'\\right\s*\|', '|', s)
    s = re.sub(r'\\left\s*\\lvert\b', '|', s)
    s = re.sub(r'\\right\s*\\rvert\b', '|', s)
    # \left. and \right. are null delimiters → remove
    s = re.sub(r'\\(?:left|right)\s*\.', '', s)

    # Sizing modifiers (\big, \Big, \bigl, \bigr, ...) → strip
    s = re.sub(r'\\[Bb]ig[glrLR]?\b\s*', '', s)

    # \langle → (   \rangle → )
    s = re.sub(r'\\langle\b', '(', s)
    s = re.sub(r'\\rangle\b', ')', s)
    # \lvert \rvert → |
    s = re.sub(r'\\lvert\b', '|', s)
    s = re.sub(r'\\rvert\b', '|', s)

    # Trailing sentence punctuation (., ;) that physics papers append to equations
    s = re.sub(r'\s*[.,;]\s*$', '', s.strip())

    # Collapse extra whitespace
    s = re.sub(r'\s+', ' ', s).strip()

    return s


# ── Stage 7: Check for unknown \\commands ─────────────────────────────────────

_KNOWN_COMMANDS: frozenset[str] = frozenset({
    # Trig
    'sin', 'cos', 'tan', 'cot', 'sec', 'csc',
    'arcsin', 'arccos', 'arctan', 'arccot',
    'sinh', 'cosh', 'tanh', 'coth',
    # Exp / log (log/ln explicitly allowed per user)
    'exp', 'log', 'ln',
    # Other allowed functions
    'sqrt', 'abs',
    # Structure (parse_latex handles \frac natively)
    'frac', 'dfrac', 'tfrac',
    # Greek lowercase
    'alpha', 'beta', 'gamma', 'delta', 'epsilon', 'varepsilon',
    'zeta', 'eta', 'theta', 'vartheta', 'iota', 'kappa', 'lambda',
    'mu', 'nu', 'xi', 'pi', 'varpi', 'rho', 'varrho', 'sigma', 'varsigma',
    'tau', 'upsilon', 'phi', 'varphi', 'chi', 'psi', 'omega',
    # Greek uppercase
    'Gamma', 'Delta', 'Theta', 'Lambda', 'Xi', 'Pi', 'Sigma', 'Upsilon',
    'Phi', 'Psi', 'Omega',
    # Derivatives (treated as variables, not banned)
    'partial',
    # Common symbols
    'infty', 'pm', 'mp', 'prime', 'ell',
    # Relations that may appear after stripping (won't hurt parse_latex)
    'approx', 'equiv', 'propto', 'leq', 'geq', 'neq', 'll', 'gg',
    'not',
    # Delimiters (after substitution these should be gone, but defensive)
    'lvert', 'rvert', 'lfloor', 'rfloor', 'lceil', 'rceil', 'vert', 'Vert',
    'langle', 'rangle',
    # Misc
    'limits', 'nolimits',
})


def find_unknown_commands(s: str) -> list[str]:
    """Return any \\command tokens not in _KNOWN_COMMANDS."""
    commands = re.findall(r'\\([a-zA-Z]+)', s)
    return [c for c in commands if c not in _KNOWN_COMMANDS]


# ── Stage 6.5: Ambiguous Greek whole-wrapper stripping ────────────────────────

# Greek letters that are commonly used as function-like wrappers in ML/physics
# papers (e.g. \sigma(x), \mu(a/b)) but are also legitimate variable names.
# Edit this set to extend or restrict the stripping rule.
_AMBIGUOUS_GREEK: frozenset[str] = frozenset({
    'alpha', 'beta', 'gamma', 'delta', 'epsilon', 'varepsilon',
    'zeta', 'eta', 'theta', 'vartheta', 'kappa', 'lambda',
    'mu', 'nu', 'xi', 'pi', 'varpi', 'rho', 'varrho',
    'sigma', 'varsigma', 'tau', 'upsilon', 'phi', 'varphi',
    'chi', 'psi', 'omega',
    # uppercase (less common as wrappers but included for completeness)
    'Gamma', 'Delta', 'Theta', 'Lambda', 'Xi', 'Pi',
    'Sigma', 'Phi', 'Psi', 'Omega',
})


def strip_greek_wrapper(s: str) -> Optional[str]:
    """
    If s is EXACTLY of the form \\greek(...) — where the Greek command is
    in _AMBIGUOUS_GREEK and the parentheses span the entire string — return
    the content inside the parentheses.  Returns None otherwise.

    Applies AFTER Stage 6 normalisation, so \\left( has already become (.

    Examples (post-normalisation):
        '\\sigma(x+y)'  →  'x+y'
        '\\mu(a/b)'     →  'a/b'
        'a + \\sigma(x)'→  None  (\\sigma is internal, not the sole wrapper)
        '\\sigma_i(x)'  →  None  (subscript present before the paren)
        '\\sin(x)'      →  None  (sin not in _AMBIGUOUS_GREEK)
    """
    m = re.match(r'^\\([A-Za-z]+)\s*\(', s)
    if m is None:
        return None

    greek = m.group(1)
    if greek not in _AMBIGUOUS_GREEK:
        return None

    # Trace the matching close-paren from the opening '(' at m.end()-1
    open_pos = m.end() - 1  # position of the '('
    depth = 0
    for i in range(open_pos, len(s)):
        if s[i] == '(':
            depth += 1
        elif s[i] == ')':
            depth -= 1
            if depth == 0:
                rest = s[i + 1:].strip()
                if not rest:
                    inner = s[open_pos + 1:i].strip()
                    return inner if inner else None
                return None  # closing paren is not at end of string

    return None  # unmatched paren


# ── Stages 5-9: Full candidate cleaning ──────────────────────────────────────

def clean_to_expr(candidate: str):
    """
    Run stages 5-9 (plus 6.5) on a single RHS string.

    Returns:
        ok          : True if expression was parsed successfully
        cleaned     : the normalised LaTeX string that was attempted
        expr        : live SymPy expression on success, None on failure
        reason      : '' on success, rejection reason on failure

    Prefer this over clean_candidate() when building an in-memory pipeline:
    the caller receives a SymPy object directly and avoids a string round-trip.
    """
    # Stage 5: pre-filter
    banned, token = has_banned_construct(candidate)
    if banned:
        inner = _strip_single_banned(candidate)
        if inner is None:
            return False, candidate, None, f'banned:{token}'
        return clean_to_expr(inner)

    # Stage 6: normalise
    cleaned = substitute_latex(candidate)
    if not cleaned:
        return False, candidate, None, 'empty_after_substitution'

    # Stage 6.5: strip ambiguous Greek whole-wrappers (loop until stable)
    while True:
        inner = strip_greek_wrapper(cleaned)
        if inner is None:
            break
        cleaned = inner
    if not cleaned:
        return False, candidate, None, 'empty_after_greek_strip'

    # Stage 7: unknown commands
    unknown = find_unknown_commands(cleaned)
    if unknown:
        return False, cleaned, None, f'unknown_commands:{",".join(unknown[:5])}'

    # Stage 8: parse
    try:
        expr = parse_latex(cleaned)
    except Exception as e:
        return False, cleaned, None, f'parse_error:{str(e)[:120]}'

    # Stage 9: post-filter
    # Derivatives → placeholder symbols so the surrounding tree is preserved.
    derivs = expr.atoms(Derivative)
    if derivs:
        subs = {d: Symbol(f'_dv{i}') for i, d in enumerate(derivs)}
        expr = expr.subs(subs)

    # Reject trivially degenerate results (bare symbol or number)
    if isinstance(expr, (Number, Symbol)):
        return False, cleaned, expr, 'trivial_expr'

    return True, cleaned, expr, ''


def clean_candidate(candidate: str) -> tuple[bool, str, str, str]:
    """
    Backward-compatible wrapper around clean_to_expr().

    Returns:
        ok          : True if expression was parsed successfully
        cleaned     : the normalised LaTeX string that was attempted
        sympy_str   : str(sympy_expr) on success (or on trivial_expr), '' on other failure
        reason      : '' on success, rejection reason on failure
    """
    ok, cleaned, expr, reason = clean_to_expr(candidate)
    sympy_str = str(expr) if expr is not None else ''
    return ok, cleaned, sympy_str, reason


# ── Process one raw JSONL record ──────────────────────────────────────────────

def process_record(record: dict) -> tuple[list[dict], list[dict]]:
    """
    Process one raw TeXExpressions record through the full pipeline.

    Returns:
        parsed   : list of successfully parsed expression dicts
        rejected : list of rejected candidate dicts with reason
    """
    raw_latex = record.get('latex', '')
    meta = {
        'arxiv_id':               record.get('arxiv_id'),
        'cited_by_count':         record.get('cited_by_count'),
        'arxiv_primary_category': record.get('arxiv_primary_category'),
        'raw_latex':              raw_latex,
    }

    parsed:   list[dict] = []
    rejected: list[dict] = []

    # Stage 1
    body = strip_environment(raw_latex)
    # Stage 2
    body = remove_metadata(body)
    # Stage 3
    lines = split_lines(body) or [body]

    for line in lines:
        # Stage 4
        rhs = extract_rhs(line)
        if rhs is None:
            rejected.append({
                **meta,
                'candidate': line.strip(),
                'rejection_stage': 'extract_rhs',
                'reason': 'no_equals',
            })
            continue

        # Stages 5-9
        ok, cleaned, sympy_str, reason = clean_candidate(rhs)

        if ok:
            parsed.append({**meta, 'cleaned_latex': cleaned, 'sympy_str': sympy_str})
        else:
            rejected.append({
                **meta,
                'candidate': rhs,
                'rejection_stage': 'clean',
                'reason': reason,
            })

    return parsed, rejected


# ── Process all JSONL files under a directory ─────────────────────────────────

def process_all(
    input_dir: Path,
    out_parsed:   Path,
    out_rejected: Path,
) -> dict:
    """
    Walk input_dir recursively for *.jsonl files, run the full pipeline,
    write results to out_parsed and out_rejected JSONL files.

    Deduplicates by (arxiv_id, file, index) to avoid double-counting
    records that appear in both per-category and aggregate files.

    Returns a summary stats dict.
    """
    out_parsed.parent.mkdir(parents=True, exist_ok=True)
    out_rejected.parent.mkdir(parents=True, exist_ok=True)

    stats = {
        'total_records':    0,
        'total_candidates': 0,
        'parsed':           0,
        'rejected':         0,
    }
    seen: set[tuple] = set()

    with out_parsed.open('w', encoding='utf-8') as fp, \
         out_rejected.open('w', encoding='utf-8') as fr:

        for jsonl_path in sorted(input_dir.rglob('*.jsonl')):
            for raw_line in jsonl_path.open(encoding='utf-8'):
                raw_line = raw_line.strip()
                if not raw_line:
                    continue
                try:
                    record = json.loads(raw_line)
                except json.JSONDecodeError:
                    continue

                # Deduplicate
                key = (
                    record.get('arxiv_id'),
                    record.get('file'),
                    record.get('index'),
                )
                if key in seen:
                    continue
                seen.add(key)

                stats['total_records'] += 1
                p_list, r_list = process_record(record)

                stats['total_candidates'] += len(p_list) + len(r_list)
                stats['parsed']           += len(p_list)
                stats['rejected']         += len(r_list)

                for rec in p_list:
                    fp.write(json.dumps(rec, ensure_ascii=False) + '\n')
                for rec in r_list:
                    fr.write(json.dumps(rec, ensure_ascii=False) + '\n')

    return stats


# ── Sort parsed output into category folders (mirrors TeXExpressions layout) ──

# Maps arxiv primary category prefix → subfolder name.
# Categories not matching any prefix are placed in 'other/'.
_CATEGORY_FOLDERS: list[tuple[tuple[str, ...], str]] = [
    (('cs.',),                                                        'cs'),
    (('astro-ph', 'cond-mat', 'hep-', 'nucl-', 'physics', 'quant-ph', 'gr-qc',
      'math-ph', 'nlin'),                                             'physics'),
    (('q-bio.',),                                                     'q-bio'),
    (('stat.',),                                                      'stat'),
    (('math.',),                                                      'math'),
    (('econ.',),                                                      'econ'),
    (('eess.',),                                                      'eess'),
]


def _category_folder(cat: str) -> str:
    """Return the subfolder name for an arxiv primary category string."""
    if not cat:
        return 'other'
    for prefixes, folder in _CATEGORY_FOLDERS:
        if any(cat.startswith(p) for p in prefixes):
            return folder
    return 'other'


def sort_parsed(
    parsed_jsonl: Path,
    out_dir: Path,
) -> dict:
    """
    Read parsed_jsonl and write records into a directory tree that mirrors
    the TeXExpressions layout:

        out_dir/
          {folder}/
            {arxiv_primary_category}.jsonl   ← per-category file
            {folder}.jsonl                   ← aggregate for the folder

    Returns a dict mapping category → record count.
    """
    import collections

    # Collect all records grouped by category
    by_category: dict[str, list[dict]] = collections.defaultdict(list)
    with parsed_jsonl.open(encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            cat = rec.get('arxiv_primary_category') or 'unknown'
            by_category[cat].append(rec)

    # Write per-category files and accumulate folder aggregates
    by_folder: dict[str, list[dict]] = collections.defaultdict(list)
    counts: dict[str, int] = {}

    for cat, records in sorted(by_category.items()):
        folder = _category_folder(cat)
        folder_dir = out_dir / folder
        folder_dir.mkdir(parents=True, exist_ok=True)

        cat_path = folder_dir / f'{cat}.jsonl'
        with cat_path.open('w', encoding='utf-8') as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + '\n')

        by_folder[folder].extend(records)
        counts[cat] = len(records)

    # Write per-folder aggregate files
    for folder, records in by_folder.items():
        agg_path = out_dir / folder / f'{folder}.jsonl'
        with agg_path.open('w', encoding='utf-8') as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + '\n')

    return counts


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Clean TeXExpressions JSONL files for SymPy parsing'
    )
    parser.add_argument(
        '--input_dir', default='TeXExpressions',
        help='Root directory of JSONL files (default: TeXExpressions)'
    )
    parser.add_argument(
        '--out_dir', default='out',
        help='Output directory (default: out)'
    )
    args = parser.parse_args()

    summary = process_all(
        input_dir=Path(args.input_dir),
        out_parsed=Path(args.out_dir) / 'parsed_expressions.jsonl',
        out_rejected=Path(args.out_dir) / 'rejected_expressions.jsonl',
    )

    total_c = summary['total_candidates']
    yield_pct = 100 * summary['parsed'] / max(1, total_c)

    print(f"Records processed : {summary['total_records']}")
    print(f"Candidates total  : {total_c}")
    print(f"Parsed            : {summary['parsed']}")
    print(f"Rejected          : {summary['rejected']}")
    print(f"Yield             : {yield_pct:.1f}%")
