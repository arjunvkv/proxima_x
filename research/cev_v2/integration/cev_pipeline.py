"""
CEV v2 Pipeline Integration.

Bridges the CEV v2 SIV / Marginal-Portfolio-Selector engine with the
existing Proxima execution pipeline.  Provides a clean entry point that
can be called from ``_select_balanced_top3()``.
"""

from research.cev_v2.core.siv_engine import SIVEngine
from research.cev_v2.core.marginal_selector import MarginalPortfolioSelector


class CEVPipelineAdapter:
    """Adapter that exposes a simple ``process_top3()`` interface.

    State
    -----
    siv_engine : SIVEngine
        Engine for computing pairwise symbol independence.
    selector : MarginalPortfolioSelector
        Greedy marginal-utility-based portfolio selector.
    """

    def __init__(self) -> None:
        self.siv_engine = SIVEngine()
        self.selector = MarginalPortfolioSelector(siv_engine=self.siv_engine)

    # ------------------------------------------------------------------
    # Primary entry point
    # ------------------------------------------------------------------

    def process_top3(
        self,
        technical_rankings: list[tuple[str, float]],
        k: int = 3,
    ) -> list[tuple[str, float, float]]:
        """Run the full CEV v2 pipeline over raw technical rankings.

        Pipeline flow::

            technical_rankings
                    ↓
            SIVEngine (independence mapping via pairwise matrix)
                    ↓
            MarginalPortfolioSelector (EIG optimisation)
                    ↓
            final ranked top-k with metadata

        Parameters
        ----------
        technical_rankings : list[tuple[str, float]]
            Symbols pre-ranked by raw conviction, sorted descending.
            Each element is ``(symbol, conviction)``.
        k : int
            Number of symbols to select (default 3).

        Returns
        -------
        list[tuple[str, float, float]]
            Selected symbols with their conviction and EIG:
            ``[(symbol, conviction, eig), ...]``.
        """
        return self.selector.select_top_k(technical_rankings, k=k)

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    def get_siv_matrix(self, symbols: list[str]) -> dict[tuple[str, str], float]:
        """Return the full pairwise SIV matrix for a list of symbols.

        Parameters
        ----------
        symbols : list[str]
            Symbols to analyse.

        Returns
        -------
        dict[(str, str), float]
            Pairwise SIV values.
        """
        return self.siv_engine.compute_pairwise_siv_matrix(symbols)

    def get_analysis(
        self,
        symbol: str,
        portfolio: list[str],
        conviction: float | None = None,
    ) -> dict:
        """Return a full independence analysis for *symbol* vs *portfolio*.

        Parameters
        ----------
        symbol : str
            Symbol to analyse.
        portfolio : list[str]
            Existing portfolio symbols.
        conviction : float or None
            Raw conviction score (optional).

        Returns
        -------
        dict
            Full analysis dictionary from :meth:`SIVEngine.get_full_analysis`.
        """
        return self.siv_engine.get_full_analysis(symbol, portfolio, conviction)
