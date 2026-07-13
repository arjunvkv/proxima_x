import sys
sys.path.insert(0, "currency_decomposition")

from narrative.state import NarrativeState, NarrativeIdentity, NarrativePhase, NarrativeMetrics
from narrative.overlay import narrative_alignment, maturity_penalty, narrative_quality


def _make_narrative(leader="CAD", phase=NarrativePhase.GROWING, nmi=0.4, rank_churn=0.1):
    return NarrativeState(
        identity=NarrativeIdentity(leader=leader, opponents=("USD", "EUR"), direction=1),
        phase=phase,
        nmi=nmi,
        age=20,
        metrics=NarrativeMetrics(rank_churn=rank_churn),
        active=True,
    )


def test_alignment_no_narrative():
    assert narrative_alignment("CADJPY", None) == 1.0


def test_alignment_inactive():
    n = _make_narrative()
    n.active = False
    assert narrative_alignment("CADJPY", n) == 1.0


def test_alignment_neutral_pair():
    n = _make_narrative(leader="CAD")
    assert narrative_alignment("EURUSD", n) == 1.0


def test_alignment_growing():
    n = _make_narrative(phase=NarrativePhase.GROWING)
    assert narrative_alignment("CADJPY", n) == 1.15


def test_alignment_emerging():
    n = _make_narrative(phase=NarrativePhase.EMERGING)
    assert narrative_alignment("CADJPY", n) == 1.05


def test_alignment_mature():
    n = _make_narrative(phase=NarrativePhase.MATURE)
    assert narrative_alignment("CADJPY", n) == 1.05


def test_alignment_decaying():
    n = _make_narrative(phase=NarrativePhase.DECAYING)
    assert narrative_alignment("CADJPY", n) == 0.90


def test_alignment_exhausted():
    n = _make_narrative(phase=NarrativePhase.EXHAUSTED)
    assert narrative_alignment("CADJPY", n) == 0.80


def test_maturity_penalty_no_narrative():
    assert maturity_penalty("CADJPY", None) == 1.0


def test_maturity_penalty_neutral():
    n = _make_narrative(leader="CAD")
    assert maturity_penalty("EURUSD", n) == 1.0


def test_maturity_penalty_growing():
    n = _make_narrative(phase=NarrativePhase.GROWING, nmi=0.4)
    assert maturity_penalty("CADJPY", n) == 1.0


def test_maturity_penalty_exhausted():
    n = _make_narrative(phase=NarrativePhase.EXHAUSTED, nmi=0.95)
    assert maturity_penalty("CADJPY", n) == 0.75


def test_maturity_penalty_decaying():
    n = _make_narrative(phase=NarrativePhase.DECAYING, nmi=0.75)
    assert maturity_penalty("CADJPY", n) == 0.85


def test_narrative_quality_no_narrative():
    assert narrative_quality(None) == 0.5


def test_narrative_quality_inactive():
    n = _make_narrative()
    n.active = False
    assert narrative_quality(n) == 0.5


def test_narrative_quality_growing():
    n = _make_narrative(phase=NarrativePhase.GROWING, nmi=0.4, rank_churn=0.1)
    q = narrative_quality(n)
    stability = 1.0 - 0.1
    expected = 1.0 * 0.4 * stability
    assert abs(q - expected) < 1e-9


def test_narrative_quality_exhausted():
    n = _make_narrative(phase=NarrativePhase.EXHAUSTED, nmi=0.95, rank_churn=0.0)
    q = narrative_quality(n)
    expected = 0.2 * 0.95 * 1.0
    assert abs(q - expected) < 1e-9


def test_combined_multiplier_range():
    n = _make_narrative(phase=NarrativePhase.GROWING)
    align = narrative_alignment("CADJPY", n)
    mat = maturity_penalty("CADJPY", n)
    combined = align * mat
    assert 0.85 <= combined <= 1.15


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"FAIL: {t.__name__}: {e}")
    print(f"\n{passed} passed, {failed} failed")
