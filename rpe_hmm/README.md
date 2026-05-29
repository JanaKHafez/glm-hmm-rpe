# RPE Analysis Integration Guide

This folder contains code to test whether **Reward Prediction Errors (RPEs)** drive motivational state transitions better than raw rewards, using the robust `ssm.HMM_TO` implementation from Mohammadi et al. (2025).

## Core Hypothesis

> RPE signals (not raw rewards) govern transitions between engaged and disengaged behavioral states in mice.

## The Models

All models share the same **observation model** (choice probability depends on stimulus contrast weighted by state-specific sensitivity), but differ in their **transition model** (what predicts state switches):

| Model | Transition Regressors | Description |
|-------|----------------------|-------------|
| **M0** | Constant (bias only) | Null model — state transitions happen at fixed background rate |
| **M1** | Filtered reward, choice, stim_side (τ=4) | Raw reward history baseline (Mohammadi et al.) |
| **M2-Q** | cumpos RPE, cumneg RPE, abs RPE, current RPE | Q-learning RPE-based transitions |
| **M2-Belief** | Same as M2-Q but using belief-state RPE | Bayesian observer RPE-based transitions |
| **M3-Q** | M1 + M2-Q combined | Combined raw reward + Q-learning RPE |
| **M3-Belief** | M1 + M2-Belief combined | Combined raw reward + belief-state RPE |

### Model Nesting

```
M0  ⊂  M1  ⊂  M3
               ↑
M0  ⊂  M2  ⊂  M3
```

## The Analyses

### Analysis 1 — Model Comparison: Does RPE beat raw reward?
- **Test**: Wilcoxon signed-rank on ΔLL(M2-Q − M1) across subjects
- **Effect size**: Rank-biserial correlation
- **CI**: Bootstrap 95% confidence interval on mean ΔLL
- **Null check**: Permutation test with circular-shifted RPE sequences

### Analysis 1b — Belief-state vs Q-learning RPE
- **Test**: Wilcoxon signed-rank on ΔLL(M2-Belief − M2-Q)
- Tests whether brain computes RPE more like delta-rule or Bayesian inference

### Analysis 2 — Transition Weight Signatures
- Extracts learned weights on each RPE regressor for:
  - Engaged → Disengaged transitions
  - Disengaged → Engaged transitions
- **Statistic**: Bootstrap 95% CIs on mean weights (excludes zero?)

### Analysis 3 — Individual Differences
- **Test**: Spearman correlation between α⁺ (positive learning rate) and mean engaged-state dwell time
- Tests whether faster learners have less stable engagement

### Analysis 4 — Block Boundary Analysis
- Peri-event analysis around contingency shifts (±20 trials)
- **Test**: One-sample t-test on RPE at boundary vs 0
- Descriptive: P(engaged) pre vs post boundary

## Files

- `rpe_glmhmm_full_analysis.py` — Complete analysis script with all models and analyses
- `rpe_glmhmm_analysis.py` — Simplified version (M1, M2-Q, M3-Q only)
- `outputs/` — Results (CSVs, figures, JSON)

## Prerequisites

1. **Install the SSM fork** (required for `ssm.HMM_TO`):
   ```bash
   cd /home/khafez/janhaf2n
   git clone https://github.com/Zeinab-Mohammadi/ssm.git
   cd ssm
   pip install numpy cython
   pip install -e .
   ```

2. **Run data preprocessing** (if not done):
   ```bash
   cd /home/khafez/janhaf2n/glm-hmm-rpe/data/ibl
   python make_animal_list_tables.py
   python processed_data_input_matrices.py
   ```

## Running the Full Analysis

```bash
cd /home/khafez/janhaf2n/glm-hmm-rpe/rpe_analysis
python rpe_glmhmm_full_analysis.py
```

### Quick Test (5 mice)
Edit `rpe_glmhmm_full_analysis.py` line ~67:
```python
N_MICE = 5  # Set to None for all mice
```

## Configuration Parameters

Edit at the top of `rpe_glmhmm_full_analysis.py`:

```python
# GLM-HMM settings
N_STATES = 4                # Number of HMM states (K=4 per your spec)
N_FOLDS = 5                 # Cross-validation folds
N_EM_ITERS_CV = 50          # EM iterations for CV
N_EM_ITERS_FULL = 100       # EM iterations for full fits
PRIOR_SIGMA = 4.0           # Observation GLM prior
TRANSITION_ALPHA = 2.0      # Transition prior (sticky)

# Q-learning settings
RL_N_RESTARTS = 8           # Multi-start optimization
RL_BOUNDS = {
    "alpha_pos": (1e-3, 0.999),
    "alpha_neg": (1e-3, 0.999),
    "beta": (0.1, 20.0),
    "v0": (0.0, 1.0),
}

# Belief-state RL settings
BELIEF_REWARD_PROBS = [0.2, 0.5, 0.8]  # Hidden reward states
BELIEF_HAZARD = 0.02                    # Transition hazard

# RPE regressors
RPE_WINDOW = 10             # Sliding window for cumulative RPE
TAU_FILTER = 4.0            # Exponential filter (matches Mohammadi et al.)

# Statistics
BOOTSTRAP_N = 200           # Bootstrap resamples
N_PERMUTATIONS = 50         # Permutation test iterations
NULL_MODE = "circular_shift"  # or "shuffle"

# Block boundary analysis
BOUNDARY_WINDOW = 20        # Half-width of peri-boundary window
```

## Outputs

After running, you'll find in `outputs/`:

| File | Description |
|------|-------------|
| `summary.csv` | Per-subject mean LLs for all 6 models |
| `results.json` | All statistical results (can recreate figures) |
| `fig1_model_comparison.png` | Boxplot + ΔLL histogram |
| `fig2_transition_weights.png` | RPE weight bar plots by direction |
| `fig3_individual_differences.png` | α⁺ vs dwell time scatter |
| `fig4_block_boundaries.png` | Peri-event RPE and engagement |

## Understanding the Results

### Analysis 1 Output
```
Mean ΔLL (M2-Q − M1): +0.0123 bits/trial
95% Bootstrap CI: [+0.0045, +0.0201]
Wilcoxon p=0.0032, rank-biserial r=+0.64
Permutation p=0.0180
Direction: RPE > Raw Reward
```
- **Positive ΔLL**: M2 (RPE) fits better than M1 (raw reward)
- **p < 0.05**: Effect is statistically significant
- **r > 0**: Consistent direction across subjects
- **Perm p < 0.05**: Effect is due to temporal alignment, not spurious

### Analysis 2 Output
```
Engaged → Disengaged:
  cumneg: mean=-0.15, 95% CI [-0.22, -0.08], excludes 0: YES
  cumpos: mean=+0.02, 95% CI [-0.05, +0.09], excludes 0: no
```
- **Excludes 0**: This regressor reliably predicts the transition
- **Sign**: Negative cumneg → negative RPEs cause disengagement

## Troubleshooting

### "SSM package not found"
```bash
git clone https://github.com/Zeinab-Mohammadi/ssm.git
pip install -e ./ssm
```

### "mice_names.npz not found"
Run preprocessing first (see Prerequisites above).

### Fits are slow
- Reduce `N_EM_ITERS_CV` to 25
- Reduce `N_STATES` to 3
- Reduce `N_PERMUTATIONS` to 20
- Set `N_MICE = 5` for testing

### Memory errors
The script processes mice one at a time, so memory should not be an issue.

### "No block boundaries found"
This happens if `left_probs` (probabilityLeft) isn't in the processed data. The block boundary analysis will be skipped but other analyses will run.

## Citation

If you use this analysis, cite:
- Mohammadi et al. (2025) for the GLM-HMM framework and SSM package
- Your work for the RPE hypothesis and Q-learning/belief-state integration
