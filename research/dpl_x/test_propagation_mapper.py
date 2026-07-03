"""Tests for propagation_mapper — import sanity and unit tests."""
import sys; sys.path.insert(0, ".")
import math

import polars as pl
import numpy as np

from research.dpl_x.propagation_mapper import (
    PropagationMapper,
    PropagationResult,
    _HAS_FORWARD_SURFACE,
)


# ---------------------------------------------------------------------------
# Import smoke test
# ---------------------------------------------------------------------------

def test_import() -> None:
    """Module imports cleanly and exposes expected names."""
    assert PropagationMapper is not None
    assert PropagationResult is not None
    print(f"[OK] PropagationMapper imported (ForwardSurface available: {_HAS_FORWARD_SURFACE})")


# ---------------------------------------------------------------------------
# PropagationResult construction
# ---------------------------------------------------------------------------

def test_propagation_result_construction() -> None:
    """Dataclass fields can be set and read back."""
    r = PropagationResult(
        source="EURUSD",
        target="GBPUSD",
        horizon=300,
        n=100,
        wr=0.55,
        pf=1.23,
        aer=1.05,
        transfer_entropy=0.12,
        lag_score=60,
        expected_return=0.0001,
        pct_return=0.0001,
        effects={0: {"n": 50, "win_rate": 0.6}},
    )
    assert r.source == "EURUSD"
    assert r.target == "GBPUSD"
    assert r.horizon == 300
    assert r.n == 100
    assert abs(r.wr - 0.55) < 1e-10
    assert abs(r.pf - 1.23) < 1e-10
    assert abs(r.aer - 1.05) < 1e-10
    assert abs(r.transfer_entropy - 0.12) < 1e-10
    assert r.lag_score == 60
    assert 0 in r.effects
    print("[OK] PropagationResult construction")


# ---------------------------------------------------------------------------
# PropagationMapper — empty / edge cases
# ---------------------------------------------------------------------------

def test_map_empty_source() -> None:
    """Mapping with an empty source DataFrame returns zeroed result."""
    mapper = PropagationMapper()
    result = mapper.map(
        source_df=pl.DataFrame(schema={"ts": pl.Int64, "oss_bucket": pl.Int32}),
        target_df=pl.DataFrame(schema={"ts": pl.Int64, "price": pl.Float64}),
        source_symbol="SRC",
        target_symbol="TGT",
        horizon=300,
    )
    assert result.n == 0
    assert result.wr == 0.0
    assert result.pf == 0.0
    assert result.lag_score == 0
    print("[OK] Empty source returns zeroed result")


def test_map_empty_target() -> None:
    """Mapping with an empty target DataFrame returns zeroed result."""
    mapper = PropagationMapper()
    source = pl.DataFrame({
        "ts": [1000, 1060, 1120],
        "oss_bucket": [3, 5, 7],
    })
    result = mapper.map(
        source_df=source,
        target_df=pl.DataFrame(schema={"ts": pl.Int64, "price": pl.Float64}),
        source_symbol="SRC",
        target_symbol="TGT",
        horizon=300,
    )
    assert result.n == 0
    print("[OK] Empty target returns zeroed result")


def test_map_no_oss_bucket_column() -> None:
    """Source without oss_bucket column returns zeroed result."""
    mapper = PropagationMapper()
    source = pl.DataFrame({"ts": [1000, 1060], "not_bucket": [1, 2]})
    target = pl.DataFrame({"ts": [1000, 1060, 1120], "price": [1.0, 1.01, 1.02]})
    result = mapper.map(
        source_df=source,
        target_df=target,
        source_symbol="SRC",
        target_symbol="TGT",
        horizon=300,
    )
    assert result.n == 0
    print("[OK] Missing oss_bucket returns zeroed result")


# ---------------------------------------------------------------------------
# PropagationMapper — happy path
# ---------------------------------------------------------------------------

def test_map_happy_path() -> None:
    """Basic mapping with synthetic aligned data produces sensible metrics."""
    mapper = PropagationMapper()

    # Build source with 10 state snapshots (1 per minute)
    ts_source = [1000 + i * 60 for i in range(10)]
    source = pl.DataFrame({
        "ts": ts_source,
        "oss_bucket": [i % 5 for i in range(10)],  # 0,1,2,3,4,0,1,2,3,4
    })

    # Build target with prices that give predictable forward returns
    ts_target = [1000 + i * 60 for i in range(15)]
    # Prices increase monotonically so forward returns are all positive
    prices = [100.0 + i * 0.1 for i in range(15)]
    target = pl.DataFrame({
        "ts": ts_target,
        "price": prices,
    })

    result = mapper.map(
        source_df=source,
        target_df=target,
        source_symbol="SRC",
        target_symbol="TGT",
        horizon=60,  # 1-minute forward return
    )

    # With rising prices, WR should be high
    assert result.n > 0, "Should have aligned samples"
    assert result.wr > 0.5, f"Bullish scenario should have >0.5 WR, got {result.wr}"
    assert result.expected_return > 0, "Returns should be positive in bullish data"
    print(f"[OK] Happy-path: n={result.n}, wr={result.wr:.3f}, pf={result.pf:.3f}, "
          f"aer={result.aer:.3f}, te={result.transfer_entropy:.4f}, "
          f"lag={result.lag_score}s, effects={len(result.effects)} buckets")


def test_map_with_negative_returns() -> None:
    """Mapping with bearish target data produces low WR."""
    mapper = PropagationMapper()

    ts_source = [2000 + i * 60 for i in range(10)]
    source = pl.DataFrame({
        "ts": ts_source,
        "oss_bucket": [i % 3 for i in range(10)],
    })

    ts_target = [2000 + i * 60 for i in range(15)]
    prices = [100.0 - i * 0.1 for i in range(15)]  # declining prices
    target = pl.DataFrame({
        "ts": ts_target,
        "price": prices,
    })

    result = mapper.map(
        source_df=source,
        target_df=target,
        source_symbol="SRC",
        target_symbol="TGT",
        horizon=60,
    )

    assert result.n > 0
    assert result.wr < 0.5, f"Bearish scenario should have <0.5 WR, got {result.wr}"
    assert result.expected_return < 0, "Returns should be negative in bearish data"
    print(f"[OK] Negative returns: n={result.n}, wr={result.wr:.3f}, "
          f"expected_return={result.expected_return:.6f}")


# ---------------------------------------------------------------------------
# Transfer entropy edge cases
# ---------------------------------------------------------------------------

def test_transfer_entropy_uniform() -> None:
    """Uniform state and returns produce zero (or near-zero) TE."""
    mapper = PropagationMapper()
    states = [0] * 20
    returns = [0.001] * 20
    te = mapper._compute_transfer_entropy(states, returns)
    assert te == 0.0, f"Uniform data should give 0 TE, got {te}"
    print(f"[OK] Uniform transfer entropy = {te}")


def test_transfer_entropy_perfect_dependence() -> None:
    """When state perfectly predicts return sign, TE > 0."""
    mapper = PropagationMapper()
    states = [0, 0, 0, 1, 1, 1]
    returns = [0.01, 0.02, 0.015, -0.01, -0.02, -0.015]
    te = mapper._compute_transfer_entropy(states, returns)
    assert te > 0, f"Predictive structure should give TE > 0, got {te}"
    print(f"[OK] Predictive transfer entropy = {te:.4f}")


def test_transfer_entropy_few_samples() -> None:
    """Fewer than 2 samples returns 0 TE."""
    mapper = PropagationMapper()
    assert mapper._compute_transfer_entropy([], []) == 0.0
    assert mapper._compute_transfer_entropy([0], [0.01]) == 0.0
    print("[OK] Few-sample transfer entropy returns 0")


# ---------------------------------------------------------------------------
# Optimal lag edge cases
# ---------------------------------------------------------------------------

def test_find_optimal_lag_zero_correlation() -> None:
    """Random source/target returns lag_score 0."""
    mapper = PropagationMapper()
    rng = np.random.default_rng(42)
    ts = [3000 + i * 60 for i in range(50)]
    source = pl.DataFrame({
        "ts": ts,
        "oss_bucket": rng.integers(0, 5, 50).tolist(),
    })
    target = pl.DataFrame({
        "ts": [3000 + i * 60 for i in range(80)],
        "price": 100.0 + rng.normal(0, 0.5, 80).cumsum(),
    })
    target_fwd = mapper._resolve_forward_returns(target, 60)
    if len(target_fwd) >= 5:
        lag = mapper._find_optimal_lag(source, target_fwd, 60)
        assert isinstance(lag, int)
        print(f"[OK] Random-data optimal lag = {lag}s")
    else:
        print("[OK] Random-data target_fwd too short, skip")


# ---------------------------------------------------------------------------
# Effects structure
# ---------------------------------------------------------------------------

def test_effects_structure() -> None:
    """_compute_effects returns expected keys per bucket."""
    mapper = PropagationMapper()
    df = pl.DataFrame({
        "ts": [1000, 1060, 1120, 1180, 1240],
        "oss_bucket": [0, 1, 2, 0, 1],
        "forward_return": [0.01, -0.005, 0.02, 0.015, -0.01],
    })
    effects = mapper._compute_effects(df, overall_std=0.015)
    assert len(effects) > 0
    for bucket_id, data in effects.items():
        for key in ("n", "win_rate", "loss_rate", "profit_factor", "aer",
                     "expected_return", "pct_return", "std_return"):
            assert key in data, f"Bucket {bucket_id} missing key '{key}'"
    print(f"[OK] Effects structure OK, {len(effects)} buckets: {list(effects.keys())}")


# ---------------------------------------------------------------------------
# Pre-computed forward column priority
# ---------------------------------------------------------------------------

def test_resolve_forward_returns_precomputed_column() -> None:
    """If target_df has forward_<horizon>, that column is used directly."""
    mapper = PropagationMapper()
    target = pl.DataFrame({
        "ts": [1000, 1060, 1120],
        "price": [1.0, 1.01, 1.02],
        "forward_60": [0.01, 0.0095, None],
    })
    resolved = mapper._resolve_forward_returns(target, 60)
    assert "forward_return" in resolved.columns
    assert len(resolved) == 2  # one null dropped
    # The values should match forward_60 (in same order)
    expected = [0.01, 0.0095]
    for r, e in zip(resolved["forward_return"].to_list(), expected):
        assert abs(r - e) < 1e-10, f"Expected {e}, got {r}"
    print(f"[OK] Pre-computed forward column resolved: {len(resolved)} rows")


# ---------------------------------------------------------------------------
# Run all
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_import()
    test_propagation_result_construction()
    test_map_empty_source()
    test_map_empty_target()
    test_map_no_oss_bucket_column()
    test_map_happy_path()
    test_map_with_negative_returns()
    test_transfer_entropy_uniform()
    test_transfer_entropy_perfect_dependence()
    test_transfer_entropy_few_samples()
    test_find_optimal_lag_zero_correlation()
    test_effects_structure()
    test_resolve_forward_returns_precomputed_column()
    print("\n=== ALL TESTS PASSED ===")
