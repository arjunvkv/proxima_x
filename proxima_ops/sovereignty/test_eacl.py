"""Quick unit tests for ExecutionArbitrationCollapse."""

import sys
sys.path.insert(0, ".")

from proxima_x.proxima_ops.sovereignty.eacl import ExecutionArbitrationCollapse


eacl = ExecutionArbitrationCollapse()


def test_all_hold():
    """All modules say HOLD / no signal, resolved=False."""
    r = eacl.resolve(
        dce_decision={"action": "HOLD", "confidence": 0.0},
        tamk_result={"authorized": False},
        era_result={"valid": False},
        loef_result={"opportunity_density": 0.0},
        ecp_result={"execution_readiness": 0.0},
    )
    assert r["resolved"] is False
    assert r["final_action"] == "HOLD"
    assert r["arbitration_loops"] == 1
    assert len(r["conflicts_resolved"]) == 0
    print("  PASS test_all_hold")


def test_buy_with_conflicts():
    """DCE says BUY, TAMK blocks, ERA invalid, LOEF low => 3 conflicts."""
    r = eacl.resolve(
        dce_decision={"action": "BUY", "confidence": 0.8},
        tamk_result={"authorized": False},
        era_result={"valid": False},
        loef_result={"opportunity_density": 0.2},
        ecp_result={"execution_readiness": 0.9},
    )
    assert r["resolved"] is True
    assert r["final_action"] == "BUY"
    assert len(r["conflicts_resolved"]) == 3
    assert "DCE vs TAMK" in r["conflicts_resolved"]
    assert "DCE vs ERA" in r["conflicts_resolved"]
    assert "LOEF vs DCE" in r["conflicts_resolved"]
    print("  PASS test_buy_with_conflicts")


def test_ses_override():
    """When ses_result has an action it takes precedence."""
    r = eacl.resolve(
        dce_decision={"action": "BUY", "confidence": 0.8},
        tamk_result={"authorized": True},
        era_result={"valid": True},
        loef_result={"opportunity_density": 0.9},
        ecp_result={"execution_readiness": 0.9},
        ses_result={"action": "HOLD"},
    )
    assert r["final_action"] == "HOLD"
    assert r["resolved"] is True
    print("  PASS test_ses_override")


def test_decision_oscillation():
    """Flip from BUY->SELL triggers oscillation flag."""
    eacl._previous_action = None  # reset

    r1 = eacl.resolve(
        dce_decision={"action": "BUY", "confidence": 0.6},
        tamk_result={"authorized": True},
        era_result={"valid": True},
        loef_result={"opportunity_density": 0.7},
        ecp_result={"execution_readiness": 0.5},
    )
    assert r1["decision_oscillation"] is False

    r2 = eacl.resolve(
        dce_decision={"action": "SELL", "confidence": 0.6},
        tamk_result={"authorized": True},
        era_result={"valid": True},
        loef_result={"opportunity_density": 0.7},
        ecp_result={"execution_readiness": 0.5},
    )
    assert r2["decision_oscillation"] is True
    print("  PASS test_decision_oscillation")


def test_normalisation():
    """Sum of weights clamped to <= 1.0."""
    # All weights at max => raw sum = 0.3 + 0.25 + 0.2 + 0.15 + 0.1 = 1.0
    # With loef_density=1.0 and ecp_readiness=1.0, it hits exactly 1.0
    r = eacl.resolve(
        dce_decision={"action": "BUY", "confidence": 1.0},
        tamk_result={"authorized": True},
        era_result={"valid": True},
        loef_result={"opportunity_density": 1.0},
        ecp_result={"execution_readiness": 1.0},
    )
    total = sum(r["resolution_weights"].values())
    assert total <= 1.0 + 1e-9, f"Sum {total} > 1.0"
    print("  PASS test_normalisation")


def test_holds_no_oscillation():
    """HOLD->HOLD does NOT trigger oscillation flag."""
    eacl._previous_action = None  # reset

    r1 = eacl.resolve(
        dce_decision={"action": "HOLD", "confidence": 0.0},
        tamk_result={"authorized": False},
        era_result={"valid": False},
        loef_result={"opportunity_density": 0.0},
        ecp_result={"execution_readiness": 0.0},
    )
    assert r1["decision_oscillation"] is False

    r2 = eacl.resolve(
        dce_decision={"action": "HOLD", "confidence": 0.0},
        tamk_result={"authorized": False},
        era_result={"valid": False},
        loef_result={"opportunity_density": 0.0},
        ecp_result={"execution_readiness": 0.0},
    )
    assert r2["decision_oscillation"] is False
    print("  PASS test_holds_no_oscillation")


def test_exception_safety():
    """Bad/non-dict inputs produce safe fallback."""
    r = eacl.resolve(
        dce_decision=None,
        tamk_result="invalid",
        era_result=123,
        loef_result=None,
        ecp_result=None,
    )
    assert r["resolved"] is False
    assert r["final_action"] == "HOLD"
    assert r["arbitration_loops"] == 1
    print("  PASS test_exception_safety")


def test_weight_values():
    """Spot-check individual weight contributions."""
    r = eacl.resolve(
        dce_decision={"action": "BUY", "confidence": 0.5},
        tamk_result={"authorized": True},
        era_result={"valid": True},
        loef_result={"opportunity_density": 0.4},
        ecp_result={"execution_readiness": 0.6},
    )
    w = r["resolution_weights"]
    # dce: 0.3 (action != HOLD)
    assert abs(w["dce"] - 0.3) < 1e-9, f"dce weight {w['dce']} != 0.3"
    # tamk: 0.25 (authorized)
    assert abs(w["tamk"] - 0.25) < 1e-9, f"tamk weight {w['tamk']} != 0.25"
    # era: 0.2 (valid)
    assert abs(w["era"] - 0.2) < 1e-9, f"era weight {w['era']} != 0.2"
    # loef: 0.15 * 0.4 = 0.06
    assert abs(w["loef"] - 0.06) < 1e-9, f"loef weight {w['loef']} != 0.06"
    # ecp: 0.1 * 0.6 = 0.06
    assert abs(w["ecp"] - 0.06) < 1e-9, f"ecp weight {w['ecp']} != 0.06"
    # Sum = 0.3 + 0.25 + 0.2 + 0.06 + 0.06 = 0.87 <= 1.0, no normalisation
    total = sum(w.values())
    assert abs(total - 0.87) < 1e-9, f"Sum {total} != 0.87"
    print("  PASS test_weight_values")


if __name__ == "__main__":
    test_all_hold()
    test_buy_with_conflicts()
    test_ses_override()
    test_decision_oscillation()
    test_normalisation()
    test_holds_no_oscillation()
    test_exception_safety()
    test_weight_values()
    print("\nAll tests passed!")
