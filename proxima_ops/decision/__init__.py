"""Proxima Ops — Decision Module.

Pure data aggregation layer that builds the canonical pre-trade
snapshot per cycle. This is the explainability layer of the system.
"""

from .pre_trade_snapshot import PreTradeSnapshot
from .shadow_mirror import ShadowDecisionMirror
from .decision_gate import DecisionGate
from .contracts import Decision, GateResult, PortfolioState, SignalOutput

__all__ = ["PreTradeSnapshot", "ShadowDecisionMirror", "DecisionGate",
           "Decision", "GateResult", "PortfolioState", "SignalOutput"]
