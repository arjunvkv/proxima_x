"""
Marginal Portfolio Selector for CEV v2.

Greedy selection using Effective Information Gain (EIG).
Selects top-k symbols by marginal utility — asking whether adding
each candidate increases total portfolio utility.
"""

from __future__ import annotations

from typing import Any

from research.cev_v2.core.siv_engine import SIVEngine


class MarginalPortfolioSelector:
    """Select top-k symbols using greedy marginal-utility optimisation.

    This replaces flat-penalty approaches with a true portfolio-aware
    selection that evaluates: "Does adding this symbol increase total
    portfolio utility?"

    Key principles:
    - Never removes symbols from consideration.
    - Never forces diversification.
    - Never enforces currency caps.
    - Strong correlated signals are preserved IF their marginal utility
      is genuinely higher.
    - First pick is always the highest-conviction symbol.
    """

    def __init__(self, siv_engine: SIVEngine | None = None) -> None:
        """Initialise the selector with an optional SIVEngine instance.

        Parameters
        ----------
        siv_engine : SIVEngine or None
            Pre-configured SIV engine.  If ``None``, a fresh one is
            created internally when needed.
        """
        self._siv_engine = siv_engine

    def select_top_k(
        self,
        ranked_symbols: list[tuple[str, float]],
        k: int = 3,
        siv_engine: SIVEngine | None = None,
    ) -> list[tuple[str, float, float]]:
        """Select top-k symbols by greedy marginal utility.

        Parameters
        ----------
        ranked_symbols : list[tuple[str, float]]
            Pre-sorted list of (symbol, conviction_score) tuples,
            ranked by conviction (highest first).
        k : int, optional
            Number of symbols to select (default 3).
        siv_engine : SIVEngine, optional
            Optional pre-configured SIVEngine instance. If not provided,
            a new one is created internally.

        Returns
        -------
        list[tuple[str, float, float]]
            Selected symbols in selection order, each as
            (symbol, eig_score, avg_independence).
        """
        # ── handle edge cases ──────────────────────────────────────
        if not ranked_symbols:
            return []

        if k <= 0:
            return []

        if k >= len(ranked_symbols):
            # No selection needed; return all with their EIG scores
            return self._return_all(ranked_symbols, siv_engine)

        # ── create engine if needed ────────────────────────────────
        engine = siv_engine or self._siv_engine or SIVEngine()

        selected: list[tuple[str, float]] = []
        selected_full: list[tuple[str, float, float]] = []

        for _ in range(k):
            best_candidate: tuple[str, float] | None = None
            best_eig = -1.0
            best_independence = 0.0

            for symbol, conviction in ranked_symbols:
                # Skip already-selected candidates
                if any(sym == symbol for sym, _ in selected):
                    continue

                candidate = (symbol, conviction)
                eig, avg_indep = self._compute_marginal_utility(
                    candidate, selected, engine,
                )

                if eig > best_eig:
                    best_eig = eig
                    best_independence = avg_indep
                    best_candidate = candidate

            # Safety guard — should never happen if k <= len(ranked)
            if best_candidate is None:
                break

            sym, conv = best_candidate
            selected.append(best_candidate)
            selected_full.append((sym, best_eig, best_independence))

        return selected_full

    # ── helpers ────────────────────────────────────────────────────

    def _compute_marginal_utility(
        self,
        candidate: tuple[str, float],
        selected: list[tuple[str, float]],
        siv_engine: SIVEngine,
    ) -> tuple[float, float]:
        """Compute EIG and average independence for a candidate.

        Parameters
        ----------
        candidate : tuple[str, float]
            (symbol, conviction_score) of the candidate symbol.
        selected : list[tuple[str, float]]
            Already-selected symbols.
        siv_engine : SIVEngine
            Engine used to compute pairwise independence scores.

        Returns
        -------
        tuple[float, float]
            (eig, avg_independence)
        """
        symbol, conviction = candidate

        # First pick — no penalty, independence is 1.0
        if not selected:
            return (conviction, 1.0)

        # Compute average independence against every already-selected symbol
        total_independence = 0.0
        for sel_sym, _ in selected:
            # Symmetric independence score
            # (0 = perfectly correlated, 1 = fully independent)
            indep_score = siv_engine.compute_independence(symbol, sel_sym)
            total_independence += indep_score

        avg_independence = total_independence / len(selected)

        # Effective Information Gain = conviction * average independence
        eig = conviction * avg_independence

        return (eig, avg_independence)

    # ── validation ─────────────────────────────────────────────────

    def validate_selection(
        self,
        selected: list[tuple[str, float, float]],
        original: list[tuple[str, float]],
    ) -> dict[str, Any]:
        """Compare the marginal-utility selection against naive top-k.

        Parameters
        ----------
        selected : list[tuple[str, float, float]]
            Output from ``select_top_k``.
        original : list[tuple[str, float]]
            Original pre-sorted (symbol, conviction) list.

        Returns
        -------
        dict
            Keys::
                selected_symbols    — list of symbol strings in selection order
                dropped_symbols     — symbols from original top-k that were excluded
                eig_scores          — dict {symbol: eig_score, ...}
                conviction_comparison — dict with:
                    'naive_top_k'        — first N symbols from original
                    'marginal_top_k'     — selected symbols
                    'dropped'            — symbols dropped relative to naive
                    'added'              — symbols added relative to naive
        """
        selected_symbols = [s for s, _, _ in selected]
        k = len(selected_symbols)

        # Naive top-k from original
        naive_top_k = [sym for sym, _ in original[:k]]

        # Build score maps
        eig_scores = {s: eig for s, eig, _ in selected}

        # Dropped: in naive top-k but not in marginal selection
        dropped = [s for s in naive_top_k if s not in selected_symbols]

        # Added: in marginal selection but not in naive top-k
        added = [s for s in selected_symbols if s not in naive_top_k]

        return {
            "selected_symbols": selected_symbols,
            "dropped_symbols": dropped,
            "eig_scores": eig_scores,
            "conviction_comparison": {
                "naive_top_k": naive_top_k,
                "marginal_top_k": selected_symbols,
                "dropped": dropped,
                "added": added,
            },
        }

    # ── internal utilities ─────────────────────────────────────────

    def _return_all(
        self,
        ranked_symbols: list[tuple[str, float]],
        siv_engine: SIVEngine | None,
    ) -> list[tuple[str, float, float]]:
        """Return all symbols with computed EIG (no selection needed).

        When *k >= len(ranked_symbols)* every symbol is included.
        EIG is still computed so the caller has consistent data.
        """
        engine = siv_engine or self._siv_engine or SIVEngine()
        result: list[tuple[str, float, float]] = []

        for i, (symbol, conviction) in enumerate(ranked_symbols):
            # Build already-selected set from previously processed symbols
            selected_sofar = [
                (s, c) for s, c in ranked_symbols[:i]
            ]
            eig, avg_indep = self._compute_marginal_utility(
                (symbol, conviction), selected_sofar, engine,
            )
            result.append((symbol, eig, avg_indep))

        return result
