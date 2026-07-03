"""Run RQ5 + RQ6 Analysis for EURJPY

Tests if adaptive_time_coordinate survives regime segmentation and controls state evolution.

RQ5: Regime Segmentation
- Classify each timestep into 5 regimes using EnergyDynamics components
- For each regime, analyze adaptive_time_coordinate properties

RQ6: State Mutation Analysis  
- Compute state mutation rates conditioned on adaptive_time_coordinate levels
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, Any

import numpy as np
import polars as pl

# Add parent directory to path to allow imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from research.mechanism_discovery.energy_dynamics import EnergyDynamics
from research.mechanism_discovery.temporal_topology import TemporalTopology
from research.state_discovery import StateDiscoveryEngine


def load_eurjpy_data() -> Dict[str, np.ndarray]:
    """Load EURJPY data and prepare arrays for analysis."""
    df = pl.read_parquet("proxima_x/data/market/EURJPY.parquet")
    
    # Ensure we have all required columns
    required_cols = ["open", "high", "low", "close", "volume"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    # Sort by timestamp
    df = df.sort("timestamp")
    
    # Prepare data arrays
    N = len(df)
    
    price = df["close"].to_numpy()
    returns = np.diff(price, prepend=price[0])
    volume = df["volume"].to_numpy()
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    
    return {
        "price": price,
        "returns": returns,
        "volume": volume,
        "high": high,
        "low": low,
        "N": N,
        "df": df,
    }


def classify_regime_by_energy(
    energy: np.ndarray,
    storage: np.ndarray,
    creation: np.ndarray,
    dissipation: np.ndarray,
    returns: np.ndarray,
) -> np.ndarray:
    """Classify each timestep into 5 regimes based on EnergyDynamics components.
    
    Regimes:
    - 0: Trending - High energy_storage + directional returns
    - 1: Ranging - Low energy_storage + low energy_dissipation  
    - 2: Volatile - High energy_dissipation + high energy_creation
    - 3: Quiet - Low all energy components
    - 4: Shock - Extreme energy_creation (top 1%)
    """
    N = len(energy)
    regime = np.zeros(N, dtype=np.int64)
    
    # Normalize energy components for comparison
    storage_norm = storage / (np.max(storage) + 1e-12)
    creation_norm = creation / (np.max(creation) + 1e-12)
    dissipation_norm = dissipation / (np.max(dissipation) + 1e-12)
    
    # Returns direction (1 for positive, -1 for negative, 0 for small)
    returns_dir = np.sign(returns)
    returns_dir[np.abs(returns) < 1e-12] = 0
    
    # Thresholds (1st percentile, median, 99th percentile)
    creation_threshold_99 = np.percentile(creation_norm, 99)
    
    for i in range(N):
        # Check for shock (extreme energy creation - top 1%)
        if creation_norm[i] > creation_threshold_99:
            regime[i] = 4
            continue
            
        # Trending: High storage + directional returns
        if storage_norm[i] > 0.6 and abs(returns_dir[i]) > 0:
            regime[i] = 0
            continue
            
        # Volatile: High dissipation + high creation  
        if dissipation_norm[i] > 0.6 and creation_norm[i] > 0.6:
            regime[i] = 2
            continue
            
        # Ranging: Low storage + low dissipation
        if storage_norm[i] < 0.3 and dissipation_norm[i] < 0.3:
            regime[i] = 1
            continue
            
        # Quiet: Low all energy components
        if storage_norm[i] < 0.2 and dissipation_norm[i] < 0.2 and creation_norm[i] < 0.2:
            regime[i] = 3
            continue
            
    return regime


def compute_information_gain(pred: np.ndarray, actual: np.ndarray, window: int = 50) -> np.ndarray:
    """Compute Information Gain (Mutual Information) between predictions and actual."""
    N = len(pred)
    ig_values = np.zeros(N)
    
    for i in range(window, N):
        pred_window = pred[i - window : i]
        actual_window = actual[i - window : i]
        
        # Create joint histogram
        pred_bins = np.digitize(pred_window, np.linspace(pred_window.min(), pred_window.max(), 10))
        actual_bins = np.digitize(actual_window, np.linspace(actual_window.min(), actual_window.max(), 10))
        
        # Joint probability
        joint_probs = np.zeros((10, 10))
        for p, a in zip(pred_bins, actual_bins):
            joint_probs[p - 1, a - 1] += 1
        joint_probs /= np.sum(joint_probs)
        
        # Marginals
        pred_probs = np.sum(joint_probs, axis=1)
        actual_probs = np.sum(joint_probs, axis=0)
        
        # Mutual Information
        mi = 0.0
        for p in range(10):
            for a in range(10):
                if joint_probs[p, a] > 0 and pred_probs[p] > 0 and actual_probs[a] > 0:
                    mi += joint_probs[p, a] * np.log(joint_probs[p, a] / (pred_probs[p] * actual_probs[a]))
        
        ig_values[i] = mi
    
    return ig_values


def compute_transferability(adaptive_time: np.ndarray, window: int = 50) -> np.ndarray:
    """Compute cross-correlation of adaptive_time_coordinate across overlapping windows."""
    N = len(adaptive_time)
    transfer = np.zeros(N)
    
    for i in range(window, N):
        past_window = adaptive_time[i - window : i]
        # Use future window with same length
        future_start = min(i + window, N)
        future_end = min(i + window, N)
        future_window = adaptive_time[i:i + window]
        
        if len(future_window) > 1 and len(future_window) == len(past_window):
            corr = np.corrcoef(past_window, future_window)[0, 1]
            transfer[i] = abs(corr) if not np.isnan(corr) else 0.0
    
    return transfer


def compute_state_mutation_rates(
    state_sequence: np.ndarray, window: int = 50
) -> Dict[str, np.ndarray]:
    """Compute per-window state mutation rates.
    
    Returns:
    - birth_rate: new states appearing / total states
    - death_rate: states disappearing / total states  
    - mutation_rate: states changing assignment / total states
    """
    N = len(state_sequence)
    
    birth_rate = np.zeros(N)
    death_rate = np.zeros(N)
    mutation_rate = np.zeros(N)
    
    for i in range(window, N):
        current_window = state_sequence[i - window : i]
        next_window = state_sequence[i : min(i + window, N)]
        
        if len(next_window) == 0:
            continue
            
        # Get unique states
        current_states = np.unique(current_window)
        next_states = np.unique(next_window)
        
        total_states = len(current_states) + len(next_states)
        if total_states == 0:
            continue
            
        # Birth rate: new states appearing
        new_states = np.setdiff1d(next_states, current_states)
        birth_rate[i] = len(new_states) / total_states
        
        # Death rate: states disappearing  
        disappeared_states = np.setdiff1d(current_states, next_states)
        death_rate[i] = len(disappeared_states) / total_states
        
        # Mutation rate: states changing assignment
        overlap = np.intersect1d(current_states, next_states)
        mutated = 0
        for state in overlap:
            # Count occurrences in both windows
            current_count = np.sum(current_window == state)
            next_count = np.sum(next_window == state)
            if current_count > 0 and next_count > 0:
                # Check if position has changed significantly
                current_pos = np.where(current_window == state)[0]
                next_pos = np.where(next_window == state)[0]
                if len(current_pos) > 0 and len(next_pos) > 0:
                    pos_change = abs(current_pos[0] - next_pos[0])
                    if pos_change > window * 0.2:  # More than 20% of window
                        mutated += 1
        
        mutation_rate[i] = mutated / total_states
    
    return {
        "birth_rate": birth_rate,
        "death_rate": death_rate,
        "mutation_rate": mutation_rate,
    }


def main():
    print("Loading EURJPY data...")
    data = load_eurjpy_data()
    N = data["N"]
    price = data["price"]
    returns = data["returns"]
    volume = data["volume"]
    high = data["high"]
    low = data["low"]
    df = data["df"]
    
    print(f"Data loaded: {N} rows")
    print(f"Price range: {price.min():.2f} - {price.max():.2f}")
    
    # Initialize mechanisms
    temporal_topology = TemporalTopology()
    energy_dynamics = EnergyDynamics()
    state_discovery = StateDiscoveryEngine()
    
    # Create mechanism data dict
    mechanism_data = {
        "price": price,
        "returns": returns,
        "volume": volume,
        "high": high,
        "low": low,
    }
    
    print("\nComputing TemporalTopology (adaptive_time_coordinate)...")
    temporal_result = temporal_topology.compute(mechanism_data)
    adaptive_time = temporal_result["adaptive_time_coordinate"]
    time_regime_tt = temporal_result["time_regime"]
    
    print("\nComputing EnergyDynamics...")
    energy_result = energy_dynamics.compute(mechanism_data)
    
    # Extract energy components
    energy = energy_result["energy_creation"]
    storage = energy_result["energy_storage"]
    creation = energy_result["energy_creation"]
    dissipation = energy_result["energy_dissipation"]
    
    print("\nComputing StateDiscovery...")
    # Build simple state vector from energy dynamics features
    energy_features = np.column_stack([
        energy_result["energy_creation"],
        energy_result["energy_storage"],
        energy_result["energy_dissipation"],
        energy_result["energy_balance"],
    ])
    
    # Simple state assignment using clustering
    from sklearn.cluster import KMeans
    n_states = min(5, N // 100)
    kmeans = KMeans(n_clusters=n_states, random_state=42, n_init=5)
    state_assignments = kmeans.fit_predict(energy_features)
    
    print(f"\nState mutation analysis: {n_states} states identified")
    
    # RQ5: Regime Segmentation
    print("\n" + "=" * 72)
    print("RQ5: Regime Segmentation")
    print("=" * 72)
    
    # Classify regimes using energy dynamics
    energy_regimes = classify_regime_by_energy(
        energy, storage, creation, dissipation, returns
    )
    
    # Define regime names
    regime_names = ["trending", "ranging", "volatile", "quiet", "shock"]
    
    print("\nRegime distribution:")
    for regime_id, name in enumerate(regime_names):
        count = np.sum(energy_regimes == regime_id)
        percentage = 100 * count / N
        print(f"  {name:10s}: {count:6d} ({percentage:5.2f}%)")
    
    # Compute adaptive_time statistics for each regime
    rq5_regimes = {}
    
    for regime_id, regime_name in enumerate(regime_names):
        mask = energy_regimes == regime_id
        if not np.any(mask):
            rq5_regimes[regime_name] = {"adaptive_time_dist": {}, "ig": 0.0, "transfer": 0.0}
            continue
        
        adaptive_time_regime = adaptive_time[mask]
        
        # Distribution statistics
        rq5_regimes[regime_name] = {
            "adaptive_time_dist": {
                "mean": float(np.mean(adaptive_time_regime)),
                "std": float(np.std(adaptive_time_regime)),
                "min": float(np.min(adaptive_time_regime)),
                "max": float(np.max(adaptive_time_regime)),
                "median": float(np.median(adaptive_time_regime)),
                "p25": float(np.percentile(adaptive_time_regime, 25)),
                "p75": float(np.percentile(adaptive_time_regime, 75)),
                "p99": float(np.percentile(adaptive_time_regime, 99)),
            }
        }
        
        # Information Gain with future returns
        future_returns = np.roll(returns, -50)
        future_returns[-50:] = 0  # Pad end with zeros
        
        # Compute IG only for this regime using the mask indices
        mask_indices = np.where(mask)[0]
        if len(mask_indices) > 0:
            # Get adaptive_time and future_returns for this regime
            adaptive_time_subset = adaptive_time[mask_indices]
            future_returns_subset = future_returns[mask_indices]
            
            # Compute IG for the subset
            ig_values_subset = compute_information_gain(
                adaptive_time_subset,
                future_returns_subset,
                window=50
            )
            ig_mean = float(np.mean(ig_values_subset)) if len(ig_values_subset) > 0 else 0.0
            
            # Transferability (cross-correlation)
            transfer_values_subset = compute_transferability(
                adaptive_time_subset,
                window=50
            )
            transfer_mean = float(np.mean(transfer_values_subset)) if len(transfer_values_subset) > 0 else 0.0
        else:
            ig_mean = 0.0
            transfer_mean = 0.0
        
        rq5_regimes[regime_name]["ig"] = float(ig_mean)
        rq5_regimes[regime_name]["transfer"] = float(transfer_mean)
        
        print(f"\n{regime_name:10s}:")
        print(f"  Adaptive time: mean={rq5_regimes[regime_name]['adaptive_time_dist']['mean']:.6f}")
        print(f"  IG with future returns: {ig_mean:.6f}")
        print(f"  Transferability: {transfer_mean:.6f}")
    
    # RQ6: State Mutation Analysis
    print("\n" + "=" * 72)
    print("RQ6: State Mutation Analysis")
    print("=" * 72)
    
    # Compute state mutation rates
    mutation_rates = compute_state_mutation_rates(state_assignments, window=50)
    
    # Condition on adaptive_time_coordinate levels
    adaptive_time_q33 = np.percentile(adaptive_time, 33)
    adaptive_time_q66 = np.percentile(adaptive_time, 66)
    
    print(f"\nAdaptive time percentiles: Q33={adaptive_time_q33:.6f}, Q66={adaptive_time_q66:.6f}")
    
    rq6_state_mutation = {}
    
    # Low adaptive time (0-33 percentile)
    low_mask = adaptive_time <= adaptive_time_q33
    if np.any(low_mask):
        rq6_state_mutation["low_adaptive_time"] = {
            "birth_rate": float(np.mean(mutation_rates["birth_rate"][low_mask])),
            "death_rate": float(np.mean(mutation_rates["death_rate"][low_mask])),
            "mutation_rate": float(np.mean(mutation_rates["mutation_rate"][low_mask])),
        }
        print(f"\nLow adaptive time:")
        print(f"  Birth rate: {rq6_state_mutation['low_adaptive_time']['birth_rate']:.6f}")
        print(f"  Death rate: {rq6_state_mutation['low_adaptive_time']['death_rate']:.6f}")
        print(f"  Mutation rate: {rq6_state_mutation['low_adaptive_time']['mutation_rate']:.6f}")
    
    # Medium adaptive time (33-66 percentile)
    med_mask = (adaptive_time > adaptive_time_q33) & (adaptive_time <= adaptive_time_q66)
    if np.any(med_mask):
        rq6_state_mutation["medium_adaptive_time"] = {
            "birth_rate": float(np.mean(mutation_rates["birth_rate"][med_mask])),
            "death_rate": float(np.mean(mutation_rates["death_rate"][med_mask])),
            "mutation_rate": float(np.mean(mutation_rates["mutation_rate"][med_mask])),
        }
        print(f"\nMedium adaptive time:")
        print(f"  Birth rate: {rq6_state_mutation['medium_adaptive_time']['birth_rate']:.6f}")
        print(f"  Death rate: {rq6_state_mutation['medium_adaptive_time']['death_rate']:.6f}")
        print(f"  Mutation rate: {rq6_state_mutation['medium_adaptive_time']['mutation_rate']:.6f}")
    
    # High adaptive time (66-100 percentile)
    high_mask = adaptive_time > adaptive_time_q66
    if np.any(high_mask):
        rq6_state_mutation["high_adaptive_time"] = {
            "birth_rate": float(np.mean(mutation_rates["birth_rate"][high_mask])),
            "death_rate": float(np.mean(mutation_rates["death_rate"][high_mask])),
            "mutation_rate": float(np.mean(mutation_rates["mutation_rate"][high_mask])),
        }
        print(f"\nHigh adaptive time:")
        print(f"  Birth rate: {rq6_state_mutation['high_adaptive_time']['birth_rate']:.6f}")
        print(f"  Death rate: {rq6_state_mutation['high_adaptive_time']['death_rate']:.6f}")
        print(f"  Mutation rate: {rq6_state_mutation['high_adaptive_time']['mutation_rate']:.6f}")
    
    # Prepare final output
    output = {
        "asset": "EURJPY",
        "rq5_regimes": rq5_regimes,
        "rq6_state_mutation": rq6_state_mutation,
        "verdict": "Does adaptive time survive regime segmentation? Does it control state evolution?",
    }
    
    # Save to JSON
    output_path = Path("proxima_x/reality/tri_rq5_rq6_regime_state.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert numpy values to Python types for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(v) for v in obj]
        else:
            return obj
    
    output_serializable = convert_to_serializable(output)
    
    with open(output_path, "w") as f:
        json.dump(output_serializable, f, indent=2, default=str)
    
    print("\n" + "=" * 72)
    print(f"Results saved to: {output_path}")
    print("=" * 72)
    
    return output


if __name__ == "__main__":
    main()
