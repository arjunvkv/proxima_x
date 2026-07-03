import json
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, 'C:/Trading/Agentic_Trading/proxima_x')
from research.mechanism_discovery.temporal_topology import TemporalTopology

def load_data():
    df = pd.read_parquet('proxima_x/data/market/EURJPY.parquet')
    price = df['close'].values
    returns = np.diff(price, prepend=price[0])
    volume = df['volume'].values.astype(float)
    high = df['high'].values
    low = df['low'].values
    return price, returns, volume, high, low

def compute_temporal_topology(price, returns, volume, high, low):
    data = {'price': price, 'returns': returns, 'volume': volume, 'high': high, 'low': low}
    tt = TemporalTopology()
    return tt.compute(data)

def rolling_entropy(arr, window):
    """Compute rolling entropy of digitized array."""
    n = len(arr)
    result = np.zeros(n)
    for i in range(window, n):
        window_data = arr[i-window:i]
        _, counts = np.unique(window_data, return_counts=True)
        probs = counts / counts.sum()
        result[i] = -np.sum(probs * np.log(probs + 1e-12))
    return result

def rolling_std(arr, window):
    """Compute rolling standard deviation."""
    n = len(arr)
    result = np.zeros(n)
    for i in range(window, n):
        result[i] = np.std(arr[i-window:i])
    return result

def count_state_changes(regime_arr, window):
    """Count regime/state transitions in rolling window."""
    n = len(regime_arr)
    result = np.zeros(n)
    for i in range(window, n):
        window_data = regime_arr[i-window:i]
        changes = np.sum(np.diff(window_data) != 0)
        result[i] = changes
    return result

def compute_future_outcomes(returns, entropy, state_changes, regime_changes, horizons, start_idx):
    """Compute future outcomes for each horizon starting from start_idx."""
    outcomes = {}
    n = len(returns)
    for h in horizons:
        end_idx = min(start_idx + h, n - 1)
        if end_idx <= start_idx:
            outcomes[f'horizon_{h}'] = {
                'future_returns_mean': 0.0,
                'future_returns_std': 0.0,
                'future_returns_skew': 0.0,
                'future_returns_kurtosis': 0.0,
                'future_volatility': 0.0,
                'future_entropy': 0.0,
                'future_state_changes': 0.0,
                'future_regime_changes': 0.0,
            }
            continue
        
        future_ret = returns[start_idx:end_idx]
        future_ent = entropy[start_idx:end_idx]
        future_state = state_changes[start_idx:end_idx]
        future_regime = regime_changes[start_idx:end_idx]
        
        outcomes[f'horizon_{h}'] = {
            'future_returns_mean': float(np.mean(future_ret)) if len(future_ret) > 0 else 0.0,
            'future_returns_std': float(np.std(future_ret)) if len(future_ret) > 0 else 0.0,
            'future_returns_skew': float(pd.Series(future_ret).skew()) if len(future_ret) > 2 else 0.0,
            'future_returns_kurtosis': float(pd.Series(future_ret).kurtosis()) if len(future_ret) > 3 else 0.0,
            'future_volatility': float(np.mean(future_ent)) if len(future_ent) > 0 else 0.0,
            'future_entropy': float(np.mean(future_ent)) if len(future_ent) > 0 else 0.0,
            'future_state_changes': float(np.sum(future_state)) if len(future_state) > 0 else 0.0,
            'future_regime_changes': float(np.sum(future_regime)) if len(future_regime) > 0 else 0.0,
        }
    return outcomes

def rq2_analysis(adaptive_time, returns, entropy, state_changes, regime_changes, horizons):
    """RQ2: Extreme value analysis using percentile buckets of adaptive_time_coordinate VALUES."""
    percentiles = [0, 5, 25, 50, 75, 95, 100]
    pct_values = np.percentile(adaptive_time, percentiles)
    
    bucket_names = ['0-5%', '5-25%', '25-50%', '50-75%', '75-95%', '95-100%']
    results = {}
    
    for i, name in enumerate(bucket_names):
        mask = (adaptive_time >= pct_values[i]) & (adaptive_time <= pct_values[i+1])
        indices = np.where(mask)[0]
        
        if len(indices) == 0:
            results[name] = {f'horizon_{h}': {k: 0.0 for k in [
                'future_returns_mean', 'future_returns_std', 'future_returns_skew', 'future_returns_kurtosis',
                'future_volatility', 'future_entropy', 'future_state_changes', 'future_regime_changes'
            ]} for h in horizons}
            continue
        
        bucket_outcomes = {f'horizon_{h}': {k: [] for k in [
            'future_returns_mean', 'future_returns_std', 'future_returns_skew', 'future_returns_kurtosis',
            'future_volatility', 'future_entropy', 'future_state_changes', 'future_regime_changes'
        ]} for h in horizons}
        
        for idx in indices:
            outcomes = compute_future_outcomes(returns, entropy, state_changes, regime_changes, horizons, idx)
            for h in horizons:
                for k in outcomes[f'horizon_{h}']:
                    bucket_outcomes[f'horizon_{h}'][k].append(outcomes[f'horizon_{h}'][k])
        
        avg_outcomes = {}
        for h in horizons:
            avg_outcomes[f'horizon_{h}'] = {
                k: float(np.mean(v)) if len(v) > 0 else 0.0 
                for k, v in bucket_outcomes[f'horizon_{h}'].items()
            }
        results[name] = avg_outcomes
    
    return results

def rq3_analysis(adaptive_time, returns, entropy, state_changes, regime_changes, horizons):
    """RQ3: Expansion vs Contraction analysis using adaptive_time_rate."""
    adaptive_time_rate = np.diff(adaptive_time, prepend=adaptive_time[0])
    
    p95 = np.percentile(adaptive_time_rate, 95)
    p05 = np.percentile(adaptive_time_rate, 5)
    
    expansion_mask = adaptive_time_rate > p95
    contraction_mask = adaptive_time_rate < p05
    neutral_mask = (adaptive_time_rate >= p05) & (adaptive_time_rate <= p95)
    
    regimes = {
        'rapid_expansion': expansion_mask,
        'rapid_contraction': contraction_mask,
        'neutral': neutral_mask
    }
    
    results = {}
    for name, mask in regimes.items():
        indices = np.where(mask)[0]
        
        if len(indices) == 0:
            results[name] = {f'horizon_{h}': {k: 0.0 for k in [
                'future_volatility', 'future_entropy', 'future_state_mutation_rate', 'future_information_gain'
            ]} for h in horizons}
            continue
        
        bucket_outcomes = {f'horizon_{h}': {k: [] for k in [
            'future_volatility', 'future_entropy', 'future_state_mutation_rate', 'future_information_gain'
        ]} for h in horizons}
        
        for idx in indices:
            outcomes = compute_future_outcomes(returns, entropy, state_changes, regime_changes, horizons, idx)
            for h in horizons:
                bucket_outcomes[f'horizon_{h}']['future_volatility'].append(outcomes[f'horizon_{h}']['future_volatility'])
                bucket_outcomes[f'horizon_{h}']['future_entropy'].append(outcomes[f'horizon_{h}']['future_entropy'])
                bucket_outcomes[f'horizon_{h}']['future_state_mutation_rate'].append(outcomes[f'horizon_{h}']['future_state_changes'])
                bucket_outcomes[f'horizon_{h}']['future_information_gain'].append(outcomes[f'horizon_{h}']['future_regime_changes'])
        
        avg_outcomes = {}
        for h in horizons:
            avg_outcomes[f'horizon_{h}'] = {
                k: float(np.mean(v)) if len(v) > 0 else 0.0 
                for k, v in bucket_outcomes[f'horizon_{h}'].items()
            }
        results[name] = avg_outcomes
    
    return results

def main():
    horizons = [1, 5, 20, 50, 100, 500]
    
    price, returns, volume, high, low = load_data()
    result = compute_temporal_topology(price, returns, volume, high, low)
    
    adaptive_time = result['adaptive_time_coordinate']
    time_density = result['time_density']
    time_regime = result['time_regime']
    
    entropy = rolling_entropy(time_regime, 20)
    vol = rolling_std(returns, 20)
    state_changes = count_state_changes(time_regime, 20)
    regime_changes = count_state_changes(time_regime, 20)
    
    rq2_results = rq2_analysis(adaptive_time, returns, entropy, state_changes, regime_changes, horizons)
    rq3_results = rq3_analysis(adaptive_time, returns, entropy, state_changes, regime_changes, horizons)
    
    key_finding = ""
    exp_1 = rq3_results['rapid_expansion']['horizon_1']['future_volatility']
    con_1 = rq3_results['rapid_contraction']['horizon_1']['future_volatility']
    neu_1 = rq3_results['neutral']['horizon_1']['future_volatility']
    
    if exp_1 > neu_1 and con_1 > neu_1:
        key_finding = "Both rapid expansion and contraction precede higher volatility vs neutral"
    elif exp_1 > neu_1:
        key_finding = "Rapid expansion precedes higher volatility"
    elif con_1 > neu_1:
        key_finding = "Rapid contraction precedes higher volatility"
    else:
        key_finding = "No clear leading indicator signal at horizon 1"
    
    output = {
        "asset": "EURJPY",
        "rq2_percentile_buckets": rq2_results,
        "rq3_expansion_contraction": rq3_results,
        "key_finding": key_finding
    }
    
    with open('proxima_x/reality/tri_rq2_rq3_consequences.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print("Analysis complete. Results saved to proxima_x/reality/tri_rq2_rq3_consequences.json")
    print(f"Key finding: {key_finding}")

if __name__ == '__main__':
    main()