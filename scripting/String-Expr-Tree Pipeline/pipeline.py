"""
pipeline.py

End-to-end pipeline: raw LaTeX → ExpressionTree.

Pipeline stages
---------------
  Stage 1   Strip outer environment wrapper        (latex_cleaner)
  Stage 2   Remove metadata markers                (latex_cleaner)
  Stage 3   Split multiline equations on \\\\      (latex_cleaner)
  Stage 4   Extract RHS (split on bare =)          (latex_cleaner)
  Stage 5   Pre-filter banned constructs           (latex_cleaner)
  Stage 6   Normalise LaTeX                        (latex_cleaner)
  Stage 6.5 Strip ambiguous Greek whole-wrappers   (latex_cleaner)
  Stage 7   Reject unknown \\commands              (latex_cleaner)
  Stage 8   Parse with SymPy parse_latex           (latex_cleaner)
  Stage 9   Replace Derivative atoms → placeholders(latex_cleaner)
  Stage 10  Filter scalar/constant terms           (scalar_filter)
  Stage 11  Convert SymPy expr → ExpressionTree    (sympy_to_tree_2)

Semantic note
-------------
Stage 10 (scalar filtering) is intentionally lossy.  Additive constants and
multiplicative scalar coefficients are discarded because the project assumes
that a linear transformation (a·f + b) is applied externally.  Only the
functional skeleton of the expression is preserved.

If scalar filtering would remove the entire expression, it falls back to the
unfiltered SymPy expression (see scalar_filter.filter_scalars).

Public API
----------
raw_latex_to_tree(raw_latex: str) -> ExpressionTree
    Full pipeline from a raw LaTeX string to an ExpressionTree.
    Returns the tree for the FIRST successfully converted candidate when the
    input contains multiple equations (multiline).
    Raises PipelineError if no candidate yields a valid tree.

raw_latex_to_trees(raw_latex: str) -> list[ExpressionTree]
    Like raw_latex_to_tree but returns ALL valid trees (one per candidate).
    Returns an empty list if none succeed.

process_record_to_trees(record: dict) -> tuple[list[dict], list[dict]]
    Batch-mode entry point.  Processes one raw TeXExpressions JSONL record
    and returns (parsed_list, rejected_list).  Each parsed dict contains
    keys: arxiv_id, cited_by_count, arxiv_primary_category, raw_latex,
          cleaned_latex, sympy_str, tree.
"""

from __future__ import annotations

import sys
from pathlib import Path

# ── sys.path setup ────────────────────────────────────────────────────────────
# This file lives at <root>/scripting/String-Expr-Tree Pipeline/pipeline.py.
# Add this folder (for latex_cleaner, scalar_filter, sympy_to_tree_2)
# and the project root (for expr.tree, expr.ops).

_HERE = Path(__file__).resolve().parent          # .../String-Expr-Tree Pipeline/
_ROOT = _HERE.parent.parent                      # .../Symbolic-RNN/

for _p in (_HERE, _ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# ── Imports ───────────────────────────────────────────────────────────────────

from latex_cleaner import (
    strip_environment,
    remove_metadata,
    split_lines,
    extract_rhs,
    clean_to_expr,
)
from scalar_filter import filter_scalars
from sympy_to_tree_2 import sympy_to_tree, UnsupportedExprError


# ── Exception ─────────────────────────────────────────────────────────────────

class PipelineError(ValueError):
    """Raised when the pipeline cannot produce a tree for the given input."""


# ── Internal: single-candidate conversion (Stages 5-11) ──────────────────────

def _candidate_to_tree(candidate: str) -> tuple:
    """
    Run stages 5-11 on one RHS candidate string.

    Returns:
        ok                  : bool
        cleaned             : normalised LaTeX string
        filtered_expr       : SymPy expression after scalar filtering, or None
        tree                : ExpressionTree on success, or None
        reason              : '' on success, rejection reason on failure
    """
    # Stages 5-9: clean and parse to a live SymPy expr
    ok, cleaned, expr, reason = clean_to_expr(candidate)
    if not ok:
        return False, cleaned, None, None, reason

    # Stage 10: scalar filtering
    filtered_expr = filter_scalars(expr)

    # Stage 11: SymPy expr → ExpressionTree
    try:
        tree = sympy_to_tree(filtered_expr)
    except UnsupportedExprError as e:
        return False, cleaned, filtered_expr, None, f'unsupported_tree:{e}'

    return True, cleaned, filtered_expr, tree, ''


# ── Public API ────────────────────────────────────────────────────────────────

def raw_latex_to_trees(raw_latex: str) -> list:
    """
    Convert raw LaTeX to a list of ExpressionTrees (one per valid candidate).

    Applies stages 1-4 to extract candidates, then stages 5-11 to each.
    Returns all trees that succeed; silently skips failures.
    """
    body = strip_environment(raw_latex)          # Stage 1
    body = remove_metadata(body)                 # Stage 2
    lines = split_lines(body) or [body]          # Stage 3

    trees = []
    for line in lines:
        rhs = extract_rhs(line)                  # Stage 4
        if rhs is None:
            # No '=' found: treat the whole line as the expression (e.g. a
            # standalone formula passed directly rather than an equation).
            rhs = line.strip()
        ok, _cleaned, _expr, tree, _reason = _candidate_to_tree(rhs)
        if ok:
            trees.append(tree)

    return trees


def raw_latex_to_tree(raw_latex: str):
    """
    Convert raw LaTeX to an ExpressionTree.

    Applies the full pipeline (stages 1-11).  If the input contains multiple
    equations (multiline), returns the tree for the FIRST successfully
    converted candidate.

    Parameters
    ----------
    raw_latex : str
        A raw LaTeX string as found in a paper (may include \\begin{equation},
        alignment characters, multiline bodies, etc.).

    Returns
    -------
    ExpressionTree

    Raises
    ------
    PipelineError
        If no candidate from this input yields a valid tree.
    """
    trees = raw_latex_to_trees(raw_latex)
    if not trees:
        raise PipelineError(f'No valid tree produced for: {raw_latex!r}')
    return trees[0]


def process_record_to_trees(record: dict) -> tuple[list[dict], list[dict]]:
    """
    Process one raw TeXExpressions JSONL record through the full pipeline.

    Returns
    -------
    parsed : list[dict]
        Each dict has keys: arxiv_id, cited_by_count, arxiv_primary_category,
        raw_latex, cleaned_latex, sympy_str, tree.
    rejected : list[dict]
        Each dict has keys: arxiv_id, cited_by_count, arxiv_primary_category,
        raw_latex, candidate, rejection_stage, reason.
    """
    raw_latex = record.get('latex', '')
    meta = {
        'arxiv_id':               record.get('arxiv_id'),
        'cited_by_count':         record.get('cited_by_count'),
        'arxiv_primary_category': record.get('arxiv_primary_category'),
        'raw_latex':              raw_latex,
    }

    body = strip_environment(raw_latex)
    body = remove_metadata(body)
    lines = split_lines(body) or [body]

    parsed:   list[dict] = []
    rejected: list[dict] = []

    for line in lines:
        rhs = extract_rhs(line)
        if rhs is None:
            # No '=' found: treat the whole line as the expression, same as
            # raw_latex_to_trees.  If it still fails, the actual rejection
            # reason (banned, unknown command, etc.) is recorded below.
            rhs = line.strip()

        ok, cleaned, filtered_expr, tree, reason = _candidate_to_tree(rhs)
        if ok:
            parsed.append({
                **meta,
                'cleaned_latex': cleaned,
                'sympy_str':     str(filtered_expr),
                'tree':          tree,
            })
        else:
            rejected.append({
                **meta,
                'candidate':       rhs,
                'rejection_stage': 'clean_or_tree',
                'reason':          reason,
            })

    return parsed, rejected
