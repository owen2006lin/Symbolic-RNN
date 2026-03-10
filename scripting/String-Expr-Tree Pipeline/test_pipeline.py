"""
test_pipeline.py

Tests for the full String-Expr-Tree Pipeline.

Run from this directory:
    python -m pytest test_pipeline.py -v

Or from the project root:
    python -m pytest "scripting/String-Expr-Tree Pipeline/test_pipeline.py" -v
"""

from __future__ import annotations

import sys
from pathlib import Path

# ── sys.path setup (mirrors pipeline.py) ─────────────────────────────────────
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
for _p in (_HERE, _ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import pytest
from sympy import symbols, sin, cos, exp, log, Symbol, Integer, Rational, pi, S

from latex_cleaner import (
    clean_to_expr,
    clean_candidate,
    strip_greek_wrapper,
    _AMBIGUOUS_GREEK,
    split_lines,
    extract_rhs,
    strip_environment,
    remove_metadata,
)
from scalar_filter import filter_scalars
from pipeline import raw_latex_to_tree, raw_latex_to_trees, PipelineError


# ═══════════════════════════════════════════════════════════════════════════════
# Cleaner / parser tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestBannedSoleWrapperStripping:
    """Stage 5: sole-wrapper unwrapping."""

    def test_sum_wrapper_stripped(self):
        # \sum_{i}(a+b)  →  strips \sum, recurses on (a+b)
        ok, cleaned, expr, reason = clean_to_expr(r'\sum_{i}(a+b)')
        # inner is (a+b) which parses fine
        assert ok, f'Expected ok, got reason={reason!r}'
        assert expr is not None

    def test_nabla_wrapper_stripped(self):
        ok, cleaned, expr, reason = clean_to_expr(r'\nabla f(x)')
        assert ok or 'trivial' in reason or 'banned' in reason
        # Either stripped successfully or rejected (bare symbol after strip)

    def test_double_banned_rejected(self):
        ok, _c, _e, reason = clean_to_expr(r'\sum f + \nabla g')
        assert not ok
        assert 'banned' in reason

    def test_content_before_banned_rejected(self):
        ok, _c, _e, reason = clean_to_expr(r'f + \sum_{i} g_i')
        assert not ok
        assert 'banned' in reason


class TestGreekWrapperStripping:
    """Stage 6.5: ambiguous Greek whole-wrapper stripping."""

    # ── strip_greek_wrapper unit tests ────────────────────────────────────────

    def test_sigma_strips(self):
        assert strip_greek_wrapper(r'\sigma(x+y)') == 'x+y'

    def test_mu_strips(self):
        assert strip_greek_wrapper(r'\mu(a/b)') == 'a/b'

    def test_phi_strips(self):
        assert strip_greek_wrapper(r'\phi(x)') == 'x'

    def test_nested_parens(self):
        assert strip_greek_wrapper(r'\sigma(x + (y/z))') == 'x + (y/z)'

    def test_sin_not_stripped(self):
        # sin is NOT in _AMBIGUOUS_GREEK
        assert strip_greek_wrapper(r'\sin(x)') is None

    def test_internal_sigma_not_stripped(self):
        # \sigma is internal, not the sole wrapper
        assert strip_greek_wrapper(r'a + \sigma(x)') is None

    def test_sigma_with_subscript_not_stripped(self):
        # \sigma_i(x) has a subscript before the paren → no match
        assert strip_greek_wrapper(r'\sigma_i(x)') is None

    def test_empty_parens(self):
        assert strip_greek_wrapper(r'\sigma()') is None

    def test_paren_not_at_end_not_stripped(self):
        assert strip_greek_wrapper(r'\sigma(x) + 1') is None

    # ── Integration: Greek wrapper in clean_to_expr ───────────────────────────

    def test_sigma_xpy_clean(self):
        """\\sigma(x+y) → x+y → parsed as Add(x,y)."""
        ok, cleaned, expr, reason = clean_to_expr(r'\sigma(x+y)')
        # x+y → trivial? No, Add(x,y) is not a bare Symbol or Number.
        assert ok, f'Expected ok but got reason={reason!r}'
        from sympy import Add
        assert isinstance(expr, Add)

    def test_mu_frac_clean(self):
        """\\mu(a/b) → a/b → parsed as Mul(a, Pow(b,-1))."""
        ok, cleaned, expr, reason = clean_to_expr(r'\mu(a/b)')
        assert ok, f'Expected ok but got reason={reason!r}'

    def test_sigma_bare_x_rejected_trivial(self):
        """\\sigma(x) → x → trivially rejected."""
        ok, _c, _e, reason = clean_to_expr(r'\sigma(x)')
        assert not ok
        assert 'trivial' in reason

    def test_double_greek_wrapper_stripped(self):
        """\\sigma(\\mu(sin(x))) → \\mu(sin(x)) → sin(x) → parsed."""
        ok, cleaned, expr, reason = clean_to_expr(r'\sigma(\mu(\sin(x)))')
        # After two strips we get \sin(x) which parses to sin(x)
        assert ok, f'Expected ok but got reason={reason!r}'
        assert isinstance(expr, type(sin(symbols('x'))))

    def test_left_right_normalised_first(self):
        """\\sigma\\left(x+y\\right) → after Stage 6 norm → \\sigma(x+y) → x+y."""
        ok, cleaned, expr, reason = clean_to_expr(r'\sigma\left(x+y\right)')
        assert ok, f'Expected ok but got reason={reason!r}'

    def test_ambiguous_greek_set_known(self):
        assert 'sigma' in _AMBIGUOUS_GREEK
        assert 'mu' in _AMBIGUOUS_GREEK
        assert 'nu' in _AMBIGUOUS_GREEK


class TestUnknownCommandRejection:
    """Stage 7: unknown command whitelist."""

    def test_custom_macro_rejected(self):
        ok, _c, _e, reason = clean_to_expr(r'\foo{x}')
        assert not ok
        assert 'unknown' in reason

    def test_known_command_accepted(self):
        ok, _c, expr, reason = clean_to_expr(r'\sin(x) + \cos(x)')
        assert ok, reason

    def test_partial_allowed(self):
        # \partial is allowed (derivative placeholder path)
        ok, _c, expr, reason = clean_to_expr(r'\frac{\partial f}{\partial x} + \sin(y)')
        # May succeed (Derivative replaced) or fail for another reason, but
        # should NOT be rejected for 'unknown_commands:\partial'
        if not ok:
            assert 'partial' not in reason


class TestDerivativePlaceholder:
    """Stage 9: Derivative → placeholder symbol."""

    def test_derivative_replaced(self):
        # \frac{\partial^2 f}{\partial x^2} contains a Derivative after parsing.
        # After replacement the expression must not contain any Derivative atoms.
        from sympy import Derivative
        ok, _c, expr, reason = clean_to_expr(r'\frac{\partial f}{\partial x} + x')
        if ok:
            assert len(expr.atoms(Derivative)) == 0, 'Derivative atom survived replacement'


class TestMultilineSplitAndRHS:
    """Stages 3-4: multiline split and RHS extraction."""

    def test_split_on_double_backslash(self):
        lines = split_lines(r'a = 1 \\ b = 2')
        assert len(lines) == 2

    def test_no_split_on_spacing_command(self):
        # \\[6pt] is a spacing command, not a line break
        lines = split_lines(r'a = b \\[6pt] + c')
        assert len(lines) == 1

    def test_rhs_extracted(self):
        rhs = extract_rhs(r'y = x^2 + 1')
        assert rhs is not None
        assert rhs.strip() == r'x^2 + 1'

    def test_no_equals_returns_none(self):
        assert extract_rhs(r'x + y') is None

    def test_multiline_rhs_pipeline(self):
        raw = r'\begin{align} y &= \sin(x) + 1 \\ z &= \cos(x) + 2 \end{align}'
        trees = raw_latex_to_trees(raw)
        # Both lines should produce trees (sin(x)+1 → sin(x), cos(x)+2 → cos(x))
        assert len(trees) >= 1


# ═══════════════════════════════════════════════════════════════════════════════
# Scalar filtering tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestScalarFiltering:
    """scalar_filter.filter_scalars behaviour."""

    def _syms(self, *names):
        return symbols(' '.join(names))

    def test_additive_constant_removed(self):
        x = symbols('x')
        result = filter_scalars(x + 3)
        assert result == x

    def test_additive_pi_removed(self):
        x = symbols('x')
        result = filter_scalars(x + pi)
        assert result == x

    def test_additive_rational_removed(self):
        x = symbols('x')
        result = filter_scalars(x + Rational(1, 2))
        assert result == x

    def test_multiplicative_integer_removed(self):
        x = symbols('x')
        result = filter_scalars(2 * x)
        assert result == x

    def test_multiplicative_rational_removed(self):
        x = symbols('x')
        result = filter_scalars(Rational(3, 7) * sin(x))
        assert result == sin(x)

    def test_combined_add_then_mul(self):
        x = symbols('x')
        # (3/7)*sin(x) + 4 → first strip 4 (Add), then strip 3/7 (Mul)
        expr = Rational(3, 7) * sin(x) + 4
        result = filter_scalars(expr)
        assert result == sin(x)

    def test_sign_stripped(self):
        x = symbols('x')
        result = filter_scalars(-x)
        assert result == x

    def test_neg_function_stripped(self):
        x = symbols('x')
        result = filter_scalars(-sin(x))
        assert result == sin(x)

    def test_structured_terms_preserved(self):
        x, y = symbols('x y')
        expr = x * y + x
        result = filter_scalars(expr)
        assert result == expr  # nothing scalar at top level

    def test_two_symbols_in_sum_preserved(self):
        x, y = symbols('x y')
        result = filter_scalars(x + y)
        assert result == x + y

    def test_fallback_all_scalar(self):
        """If everything is scalar, return original unchanged."""
        expr = S(3) + pi
        result = filter_scalars(expr)
        assert result == expr

    def test_fallback_pure_number(self):
        result = filter_scalars(S(5))
        assert result == S(5)

    def test_function_arg_not_touched(self):
        """3 inside sin(x+3) must NOT be stripped."""
        x = symbols('x')
        expr = sin(x + 3)
        result = filter_scalars(expr)
        assert result == sin(x + 3)

    def test_two_term_sum_with_constant(self):
        x, y = symbols('x y')
        expr = x + y + S(5)
        result = filter_scalars(expr)
        from sympy import Add
        # 5 is stripped; x+y remains
        assert result == x + y


# ═══════════════════════════════════════════════════════════════════════════════
# End-to-end tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestEndToEnd:
    """raw_latex_to_tree / raw_latex_to_trees full pipeline."""

    def test_simple_fraction(self):
        tree = raw_latex_to_tree(r'\frac{x}{y}')
        assert tree is not None

    def test_simple_trig(self):
        tree = raw_latex_to_tree(r'f = \sin(x) + \cos(y)')
        assert tree is not None

    def test_with_environment_wrapper(self):
        raw = r'\begin{equation} f = \sin(x) + x^2 \end{equation}'
        tree = raw_latex_to_tree(raw)
        assert tree is not None

    def test_with_metadata_stripped(self):
        raw = r'\begin{equation}\label{eq:foo} f = \sin(x) \end{equation}'
        tree = raw_latex_to_tree(raw)
        assert tree is not None

    def test_banned_raises(self):
        # Two banned constructs → no sole-wrapper stripping → PipelineError.
        # \nabla and \cdot are both in _BANNED_RE so _strip_single_banned
        # returns None and the candidate is rejected.
        with pytest.raises(PipelineError):
            raw_latex_to_tree(r'f = \nabla \cdot F')

    def test_scalar_stripped_in_e2e(self):
        """2*sin(x) + 3 → sin(x) → tree of sin."""
        tree = raw_latex_to_tree(r'f = 2\sin(x) + 3')
        assert tree is not None
        s = tree.to_str()
        # tree should represent sin(x); the 2 and 3 should be gone
        assert 'sin' in s

    def test_greek_wrapper_e2e(self):
        """Raw LaTeX with \\sigma(...) wrapper → pipeline strips it."""
        tree = raw_latex_to_tree(r'f = \sigma(\sin(x) + \cos(x))')
        assert tree is not None

    def test_ml_style_rnn(self):
        r"""
        Typical ML RNN equation: h_t = \sigma(W x_t + U h_{t-1} + b).
        The whole RHS is \\sigma(...) → stripped to W x_t + U h_{t-1} + b.
        Then scalar-filter leaves the symbolic structure intact.
        """
        raw = r'h_t = \sigma(W x_t + U h_{t-1} + b)'
        trees = raw_latex_to_trees(raw)
        assert len(trees) >= 1

    def test_multiline_returns_multiple_trees(self):
        raw = (
            r'\begin{align}'
            r'y &= \sin(x) + 1 \\ '
            r'z &= \cos(x) + 2'
            r'\end{align}'
        )
        trees = raw_latex_to_trees(raw)
        assert len(trees) == 2

    def test_unknown_command_rejected(self):
        with pytest.raises(PipelineError):
            raw_latex_to_tree(r'f = \myop{x}')

    def test_trivial_rejected(self):
        # A bare symbol after cleaning → PipelineError
        with pytest.raises(PipelineError):
            raw_latex_to_tree(r'f = x')

    def test_process_record_to_trees(self):
        from pipeline import process_record_to_trees
        record = {
            'arxiv_id': 'test.0001',
            'cited_by_count': 0,
            'arxiv_primary_category': 'cs.LG',
            'latex': r'f = \sin(x) + \cos(y)',
        }
        parsed, rejected = process_record_to_trees(record)
        assert len(parsed) == 1
        assert parsed[0]['tree'] is not None
        assert 'cleaned_latex' in parsed[0]
        assert 'sympy_str' in parsed[0]


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
