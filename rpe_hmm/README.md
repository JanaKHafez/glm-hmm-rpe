# RPE GLM-HMM Analysis

This folder contains the current end-to-end script for testing whether reward prediction errors (RPEs) explain motivational state transitions better than raw reward history in the IBL GLM-HMM framework.

The current entry point is `run_rpe_hmm.py`.

## What the script does

`run_rpe_hmm.py` runs a full pipeline that:

1. loads preprocessed IBL mouse data,
2. fits a Q-learning model per mouse,
3. derives Q-based and belief-based RPE regressors,
4. fits multiple GLM-HMM transition models using `ssm.HMM_TO`,
5. compares held-out log-likelihood across models,
6. prints and saves log-likelihood differences between each model and the `L0` baseline,
7. fits full-session `M2-Q` models for state-sequence summaries,
8. saves summary tables, figures, and JSON outputs.

## Current model set

All models share the same observation model and differ only in the transition regressors.

| Model | Transition regressors | Notes |
|---|---|---|
| `L0` | Training-set choice frequencies only | Current baseline used in code |
| `M1` | Original preprocessed Mohammadi-style transition regressors | Raw reward / choice / stimulus history baseline |
| `M2-Q` | Filtered positive RPE, filtered negative RPE, running absolute RPE, current RPE | Q-learning RPE model |
| `M2-Belief` | Same structure as `M2-Q`, using belief-state RPEs | Bayesian belief learner variant |
| `M3-Q` | `M1` + `M2-Q` | Combined raw history + Q-RPE |
| `M3-Belief` | `M1` + `M2-Belief` | Combined raw history + belief-RPE |

## Analyses currently produced

### 1. Model comparison: `M2-Q` vs `M1`

- per-subject held-out $\Delta LL = LL(M2\text{-}Q) - LL(M1)$,
- Wilcoxon signed-rank test,
- rank-biserial effect size,
- bootstrap confidence interval for the mean difference.

### 1b. Model comparison: `M2-Belief` vs `M2-Q`

- per-subject held-out $\Delta LL = LL(M2\text{-}Belief) - LL(M2\text{-}Q)$,
- Wilcoxon signed-rank test,
- rank-biserial effect size,
- bootstrap confidence interval.

### Baseline comparison summary

Before the main inferential analyses, the script prints and saves per-model:

- $\Delta LL = LL(\text{model}) - LL(L0)$,
- bootstrap confidence interval for the mean difference,
- bootstrap SE.

Saved outputs include `baseline_model_deltas.csv` and `baseline_model_deltas.json`.

### 2. Transition-event summary

Using full-session `M2-Q` fits, the script summarizes RPE-linked regressors at realized Viterbi state switches for:

- engaged $\rightarrow$ disengaged,
- disengaged $\rightarrow$ engaged.

The reported quantities are the mean regressor values at those transition events, with bootstrap confidence intervals.

### 3. Individual-differences summary

- Spearman correlation between `alpha_pos` and mean engaged-state dwell time.

## Requirements

### 1. SSM package

The script imports `ssm.HMM_TO`. This repository already has an `ssm/` folder at the repo root, but it still needs to be installed into the active Python environment.

Example:

```bash
cd ssm
pip install numpy cython
pip install -e .
```

### 2. Preprocessed IBL data

The script expects data under:

- `data/ibl/Della_cluster_data/separate_mouse_data/`

Required files include:

- `mice_names.npz`
- `*_processed.npz`
- `*_rewarded.npz`
- optionally `*_fold_session_map.npz`

If these files are missing, run the IBL preprocessing pipeline first.

## Running the analysis

```bash
python rpe_hmm/run_rpe_hmm.py
```

## Quick test mode

For a short test run, edit `run_rpe_hmm.py` and set:

```python
N_MICE = 5
```

Set it back to:

```python
N_MICE = None
```

to analyze all mice.

## Default configuration

```python
# GLM-HMM settings
N_STATES = 4
N_FOLDS = 5
N_EM_ITERS_CV = 50
N_EM_ITERS_FULL = 100
PRIOR_SIGMA = 4.0
TRANSITION_ALPHA = 2.0

# RL settings
RL_N_RESTARTS = 8
RL_MAX_ITER = 500

# Belief model
BELIEF_REWARD_PROBS = np.array([0.2, 0.5, 0.8], dtype=float)
BELIEF_HAZARD = 0.02

# RPE regressors
RPE_WINDOW = 10
TAU_FILTER = 4.0

# Statistics
BOOTSTRAP_N = 200

# Scope / compute
N_MICE = None
N_WORKERS = mp.cpu_count() - 1
```

## Main outputs

After a run, the script typically writes:

| File | Description |
|---|---|
| `summary.csv` | Per-mouse summary of held-out log-likelihood and RL parameters |
| `baseline_model_deltas.csv` | Mean $\Delta LL$ of each model versus `L0`, with bootstrap CI/SE |
| `baseline_model_deltas.json` | JSON version of the baseline-delta summary |
| `results.json` | Main analysis results bundle |
| `fig1_model_comparison.png` | Model comparison boxplot and $\Delta LL$ histogram |
| `fig2_transition_weights.png` | Event-based RPE transition summary figure |
| `fig3_individual_differences.png` | `alpha_pos` vs dwell-time scatter |
| `live/cv_mouse_*.json` | Per-mouse incremental CV outputs |
| `live/full_mouse_*.json` | Per-mouse incremental full-session outputs |
| `live/analysis_*.json` | Incremental analysis-stage outputs |