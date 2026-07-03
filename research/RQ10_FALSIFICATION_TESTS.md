# Proxima X RQ10 Falsification Tests - Implementation Summary

## Overview

This implementation provides a comprehensive suite of adversarial attacks to falsify the `adaptive_time_coordinate` hypothesis as specified in Requirement Question 10 (RQ10).

## Objective
Destroy the adaptive_time_coordinate hypothesis by applying 6 distinct types of adversarial attacks to measure its robustness and informativeness under various attack scenarios.

## Attacks Implemented

### 1. Noise Injection Attack
- **Method**: Add Gaussian noise to returns at multiple levels (0.1×, 0.5×, 1.0×, 2.0× original standard deviation)
- **Purpose**: Test robustness to random perturbations
- **Metrics**: IG survival, SID survival, transfer survival

### 2. Bootstrap Resampling Attack
- **Method**: Block bootstrap (block=50) 100 times to create sampling distribution
- **Purpose**: Assess stability and variability of the adaptive time coordinate
- **Metrics**: Distribution of IG and SID values, survival probabilities

### 3. Regime Randomization Attack
- **Method**: Shuffle regime labels from time_regime to test if structure disappears
- **Purpose**: Validate that adaptive time is genuinely informative, not random
- **Metrics**: IG survival, SID survival

### 4. Time Randomization Attack
- **Method**: Two techniques:
  - Phase randomization (preserve power spectrum, randomize phase)
  - Circular shift of returns by random offset
- **Purpose**: Test sensitivity to temporal ordering changes
- **Metrics**: IG survival, SID survival for both phase_random and circular_shift

### 5. State Randomization Attack
- **Method**: Shuffle state assignments from state discovery process
- **Purpose**: Test if adaptive time structure depends on specific state assignments
- **Metrics**: IG survival

### 6. Component Ablation Attack
- **Method**: Remove each density component individually:
  - Remove time_density
  - Remove event_density
  - Remove information_density
  - Remove behavior_density
- **Purpose**: Identify which components are essential for adaptive time coordinate
- **Metrics**: IG and SID for each ablation scenario

## Key Features

### Data Loading
- Loads EURJPY data from `proxima_x/data/market/EURJPY.parquet`
- Extracts price, returns, volume, high, and low features
- Handles missing data with synthetic generation where needed

### Metrics Calculation
- **Information Gain (IG)**: Mean of adaptive time coordinate contributions
- **State Information Density (SID)**: Entropy-based measure using MIEstimator
- **Transfer Metrics**: Cross-asset (USDJPY, GBPJPY) transfer scores

### Output Format
The implementation produces results in the exact JSON format specified in RQ10:
```json
{
  "asset": "EURJPY",
  "original": {"ig": X, "sid": Y, "transfer_usdjpy": Z, "transfer_gbpjpy": W},
  "noise_injection": {
    "0.1x": {"ig_survival": X, "sid_survival": Y, "transfer_survival": Z},
    "0.5x": {...},
    "1.0x": {...},
    "2.0x": {...}
  },
  "bootstrap": {
    "n_samples": 100,
    "ig_distribution": [mean, std, min, max, q5, q95],
    "sid_distribution": [...],
    "survival_probability_ig": P,
    "survival_probability_sid": Q
  },
  "regime_randomization": {"ig_survival": X, "sid_survival": Y},
  "time_randomization": {
    "phase_random": {"ig_survival": X, "sid_survival": Y},
    "circular_shift": {"ig_survival": X, "sid_survival": Y}
  },
  "state_randomization": {"ig_survival": X, "sid_survival": Y},
  "component_ablation": {
    "no_time_density": {"ig": X, "sid": Y},
    "no_event_density": {"ig": X, "sid": Y},
    "no_information_density": {"ig": X, "sid": Y},
    "no_behavior_density": {"ig": X, "sid": Y}
  },
  "verdict": "Does adaptive time remain informative after attack?"
}
```

### File Structure
- **Script**: `proxima_x/research/rq10_falsification_test.py`
- **Output**: `proxima_x/reality/tri_rq10_falsification.json`
- **Dependencies**: Uses existing `TemporalTopology`, `MIEstimator`, and other proximal components

### Key Insights
1. **Survivability Analysis**: The survival probabilities indicate how robust the adaptive time coordinate is to each type of attack
2. **Critical Component Identification**: Component ablation reveals which density components are essential
3. **Temporal Sensitivity**: Time randomization tests whether the structure depends on specific temporal ordering
4. **Statistical Stability**: Bootstrap resampling provides confidence intervals for attack effects

## Usage

To run the falsification tests:

```bash
# If Python is available:
python proxima_x/research/rq10_falsification_test.py

# Or as a module:
python -c "from proxima_x.research.rq10_falsification_test import RQ10FalsificationTest; test = RQ10FalsificationTest(); results = test.run_all_attacks()"
```

The script will:
1. Load EURJPY market data
2. Run the original TemporalTopology mechanism to establish baseline metrics
3. Apply all 6 attack types in sequence
4. Calculate survival probabilities and distribution statistics
5. Save comprehensive results to the output JSON file
6. Print a summary of key findings

## Research Questions Answered

The implementation addresses these key questions:

1. **Does adaptive time remain informative after noise injection?**
2. **What is the survival probability of IG and SID under bootstrap resampling?**
3. **Can regime randomization destroy the adaptive time structure?**
4. **How sensitive is the mechanism to temporal ordering changes?**
5. **Does the adaptive time coordinate depend on specific state assignments?**
6. **Which density components are essential for the adaptive time coordinate?**

## Conclusion

This implementation provides a comprehensive adversarial testing framework for validating or falsifying the adaptive_time_coordinate hypothesis. By systematically applying 6 different attack types and measuring their effects, researchers can determine whether the adaptive time coordinate represents genuine market structure or is merely an artifact of the estimation method.

The survival probabilities and component ablation results directly inform whether the adaptive time coordinate hypothesis holds under realistic market attack scenarios.