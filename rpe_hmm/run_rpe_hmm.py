# -*- coding: utf-8 -*-
"""
Full RPE-based GLM-HMM Analysis using Mohammadi et al. (2025) SSM Framework

This script implements the complete analysis pipeline:
- L0/M0: Baseline model from Mohammadi et al. (training-set choice frequencies)
- M1: Raw reward history (Mohammadi et al. baseline)
- M2-Q: Q-learning RPE transitions
- M2-Belief: Belief-state RPE transitions  
- M3-Q: Combined (M1 + M2-Q)
- M3-Belief: Combined (M1 + M2-Belief)

Analyses:
1. Model comparison with Wilcoxon, bootstrap CI, (permutation test)
1b. Belief vs Q-learning RPE comparison
2. Transition weight signatures (engaged↔disengaged)... UNDER DEVELOPMENT
3. Individual differences (α⁺ vs dwell time)... UNDER DEVELOPMENT
4. Block boundary peri-event analysis... UNDER DEVELOPMENT
"""

import os
import sys

# Force UTF-8 stdio to avoid UnicodeEncodeError on some terminals/locales.
# (Keeps existing Unicode print strings like em-dash/arrows/Greek letters.)
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import json
import pickle
import warnings
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict

import numpy as np
import numpy.random as npr
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
from scipy.special import expit, softmax
from scipy.stats import wilcoxon, spearmanr, ttest_1samp
from sklearn.model_selection import KFold
from tqdm.auto import tqdm

# Add repo paths for imports
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "GLMHMM_and_GLM_frameworks"))
sys.path.insert(0, str(REPO_ROOT / "data" / "ibl"))

# Import the robust SSM package
try:
    import ssm
    print(f"Using ssm package: {ssm.__file__}")
except ImportError:
    raise ImportError(
        "SSM package not found. Install the Mohammadi fork:\n"
        "  git clone https://github.com/Zeinab-Mohammadi/ssm.git\n"
        "  pip install -e ./ssm"
    )

from data_utils import (
    mice_names_info, mask_for_violations,
    data_segmentation_session, divide_sessions_for_test_train
)

warnings.filterwarnings("ignore")

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.linewidth": 1.2,
    "xtick.major.width": 1.2,
    "ytick.major.width": 1.2,
    "figure.dpi": 150,
})

# ==============================================================================
# CONFIG
# ==============================================================================

# Paths
DATA_DIR = REPO_ROOT / "data" / "ibl" / "Della_cluster_data"
OUT_DIR = REPO_ROOT / "rpe_analysis" / "outputs"
LIVE_DIR = OUT_DIR / "live"
OUT_DIR.mkdir(parents=True, exist_ok=True)
LIVE_DIR.mkdir(parents=True, exist_ok=True)

# GLM-HMM settings
N_STATES = 4                # Number of HMM states (K=4 per your spec)
N_FOLDS = 2                 # Cross-validation folds
N_EM_ITERS_CV = 50          # EM iterations for CV (faster)
N_EM_ITERS_FULL = 100       # EM iterations for full fits
PRIOR_SIGMA = 4.0           # Observation GLM prior
TRANSITION_ALPHA = 2.0      # Transition prior (sticky)

# Q-learning settings
RL_N_RESTARTS = 8
RL_MAX_ITER = 500
RL_BOUNDS = {
    "alpha_pos": (1e-3, 1 - 1e-3),
    "alpha_neg": (1e-3, 1 - 1e-3),
    "beta": (0.1, 20.0),
    "v0": (0.0, 1.0),
}

# Belief-state RL settings
BELIEF_REWARD_PROBS = np.array([0.2, 0.5, 0.8], dtype=float)
BELIEF_HAZARD = 0.02

# RPE regressor settings
RPE_WINDOW = 10             # Sliding window for cumulative RPE
TAU_FILTER = 4.0            # Exponential filter tau (matching Mohammadi et al.)

# Statistical settings
BOOTSTRAP_N = 200           # Bootstrap resamples for CIs
N_PERMUTATIONS = 00         # Permutation test iterations
NULL_MODE = "circular_shift"  # or "shuffle"

# Block boundary analysis
BOUNDARY_WINDOW = 20        # Half-width of peri-boundary window

# How many mice to analyze (None = all)
N_MICE = None  # Set to 5 for quick test

# Parallelization settings
N_WORKERS = mp.cpu_count() - 1  # Number of parallel workers

# Figure settings
FIGURE_DPI = 150

# ==============================================================================
# Q-LEARNING MODEL
# ==============================================================================

def convert_for_json(obj):
    """Convert numpy/pandas objects into JSON-serializable objects."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    if isinstance(obj, (np.bool_, )):
        return bool(obj)
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    if isinstance(obj, pd.Series):
        return obj.to_dict()
    if isinstance(obj, dict):
        return {k: convert_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_for_json(x) for x in obj]
    if isinstance(obj, tuple):
        return [convert_for_json(x) for x in obj]
    return obj


def save_json(path, payload):
    """Atomically save JSON payload to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(convert_for_json(payload), f, indent=2)
    os.replace(tmp, path)


def build_summary_df(all_cv_results):
    """Build per-mouse summary DataFrame from CV results."""
    rows = []
    for r in all_cv_results:
        if r is None or len(r.get("ll_m1", [])) == 0:
            continue
        rows.append({
            "mouse": r["mouse"],
            "ll_m0": np.nanmean(r["ll_m0"]),
            "ll_m1": np.nanmean(r["ll_m1"]),
            "ll_m2_q": np.nanmean(r["ll_m2_q"]),
            "ll_m2_b": np.nanmean(r["ll_m2_b"]),
            "ll_m3_q": np.nanmean(r["ll_m3_q"]),
            "ll_m3_b": np.nanmean(r["ll_m3_b"]),
            "alpha_pos": r["rl_params"][0],
            "alpha_neg": r["rl_params"][1],
            "beta": r["rl_params"][2],
            "n_folds": len(r["ll_m1"]),
        })
    summary_df = pd.DataFrame(rows)
    if len(summary_df) > 0:
        summary_df["delta_m2q_m1"] = summary_df["ll_m2_q"] - summary_df["ll_m1"]
        summary_df["delta_m2b_m2q"] = summary_df["ll_m2_b"] - summary_df["ll_m2_q"]
    return summary_df

def q_learning_nll(params, choices, rewards):
    """Compute negative log-likelihood and RPEs for Q-learning model."""
    alpha_pos, alpha_neg, beta, v0 = params
    V = np.array([v0, v0], dtype=float)
    nll = 0.0
    rpes, values = [], []
    
    for t in range(len(choices)):
        c = int(choices[t])
        p_right = expit(beta * (V[1] - V[0]))
        p_chose = p_right if c == 1 else (1.0 - p_right)
        nll -= np.log(p_chose + 1e-12)
        
        r = float(rewards[t])
        rpe = r - V[c]
        rpes.append(rpe)
        values.append(V[c])
        V[c] += (alpha_pos if rpe > 0 else alpha_neg) * rpe
    
    return nll, rpes, values


def fit_rl_model(choices, rewards, rng=None):
    """Fit Q-learning model with multiple restarts."""
    if rng is None:
        rng = np.random.default_rng(0)
    
    best_nll = np.inf
    best_params = None
    bounds = [
        RL_BOUNDS["alpha_pos"],
        RL_BOUNDS["alpha_neg"],
        RL_BOUNDS["beta"],
        RL_BOUNDS["v0"],
    ]
    
    for _ in range(RL_N_RESTARTS):
        x0 = [
            rng.uniform(*RL_BOUNDS["alpha_pos"]),
            rng.uniform(*RL_BOUNDS["alpha_neg"]),
            rng.uniform(*RL_BOUNDS["beta"]),
            rng.uniform(*RL_BOUNDS["v0"]),
        ]
        res = minimize(
            lambda p: q_learning_nll(p, choices, rewards)[0],
            x0=x0, bounds=bounds, method="L-BFGS-B",
            options={"maxiter": RL_MAX_ITER, "ftol": 1e-10},
        )
        if res.fun < best_nll:
            best_nll, best_params = res.fun, res.x
    
    _, rpes, values = q_learning_nll(best_params, choices, rewards)
    return best_params, np.array(rpes), np.array(values)


def check_boundary_hit(params):
    """
    Check if critical parameters hit their bounds.
    
    Only checks alpha_pos, alpha_neg, and beta (not v0).
    v0 hitting 0 or 1 is fine—it's just the initial value estimate.
    Learning rates hitting bounds suggests the Q-learning model is a poor fit.
    """
    alpha_pos, alpha_neg, beta, v0 = params
    
    # Check learning rates (hitting bounds = poor RL fit)
    if np.isclose(alpha_pos, RL_BOUNDS["alpha_pos"][0], atol=1e-4) or \
       np.isclose(alpha_pos, RL_BOUNDS["alpha_pos"][1], atol=1e-4):
        return True
    if np.isclose(alpha_neg, RL_BOUNDS["alpha_neg"][0], atol=1e-4) or \
       np.isclose(alpha_neg, RL_BOUNDS["alpha_neg"][1], atol=1e-4):
        return True
    
    # Check beta (hitting lower bound = no choice sensitivity)
    if np.isclose(beta, RL_BOUNDS["beta"][0], atol=1e-4) or \
       np.isclose(beta, RL_BOUNDS["beta"][1], atol=1e-4):
        return True
    
    # v0 is NOT checked - hitting 0 or 1 is fine (just initial value)
    return False


# ==============================================================================
# BELIEF-STATE RL MODEL
# ==============================================================================

def fit_belief_model(rewards, reward_probs=BELIEF_REWARD_PROBS, hazard=BELIEF_HAZARD):
    """
    Bayesian belief-state learner over a hidden reward state.
    
    Returns:
        belief_rpes: reward - expected reward under belief
        belief_df: dataframe of belief-derived regressors
        posteriors: posterior over hidden states for each trial
    """
    rewards = np.asarray(rewards, dtype=float)
    p_state = np.asarray(reward_probs, dtype=float)
    K = len(p_state)
    
    # Transition matrix with hazard rate
    T = np.full((K, K), hazard / (K - 1), dtype=float)
    np.fill_diagonal(T, 1.0 - hazard)
    
    post = np.ones(K, dtype=float) / K
    
    belief_rpes = []
    belief_mean = []
    belief_entropy = []
    belief_surprise = []
    belief_kl = []
    posteriors = []
    
    eps = 1e-12
    
    for r in rewards:
        # Prior: propagate through transition
        prior = post @ T
        p_pred = float(np.dot(prior, p_state))
        rpe = r - p_pred
        belief_rpes.append(rpe)
        belief_mean.append(p_pred)
        
        # Surprise (negative log-likelihood)
        belief_surprise.append(
            -(r * np.log(p_pred + eps) + (1.0 - r) * np.log(1.0 - p_pred + eps))
        )
        
        # Posterior update (Bayes rule)
        like = p_state if r > 0.5 else (1.0 - p_state)
        post_new = prior * like
        post_new /= post_new.sum() + eps
        
        # Information-theoretic quantities
        belief_entropy.append(-(post_new * np.log(post_new + eps)).sum())
        belief_kl.append(np.sum(post_new * np.log((post_new + eps) / (prior + eps))))
        
        post = post_new
        posteriors.append(post.copy())
    
    belief_df = pd.DataFrame({
        "belief_rpe": belief_rpes,
        "belief_mean": belief_mean,
        "belief_surprise": belief_surprise,
        "belief_entropy": belief_entropy,
        "belief_kl": belief_kl,
    })
    
    return np.array(belief_rpes), belief_df, np.array(posteriors)


# ==============================================================================
# TRANSITION REGRESSORS
# ==============================================================================

def exp_filter(x, tau=TAU_FILTER):
    """
    Exponential filter matching Mohammadi et al. (2025) transition regressors.
    Each output value is a normalized weighted sum of current and past inputs,
    with weights decaying as exp(-k/tau) for lag k.
    """
    x = np.asarray(x, dtype=float)
    T = len(x)
    if T == 0:
        return x
    out = np.zeros(T)
    decay = np.exp(-1.0 / tau)
    running = 0.0
    for t in range(T):
        running = x[t] + decay * running
        out[t] = running
    norm = (1.0 - decay ** T) / (1.0 - decay + 1e-12)
    return out / (norm + 1e-12)


def compute_rpe_regressors(rpes):
    """
    Compute RPE-based transition regressors.
    Returns DataFrame with columns for each regressor type.
    """
    rpes = np.array(rpes)
    k = np.ones(RPE_WINDOW) / RPE_WINDOW
    
    return pd.DataFrame({
        "rpe_cumpos": np.convolve(np.maximum(rpes, 0), k, mode="same"),
        "rpe_cumneg": np.convolve(np.minimum(rpes, 0), k, mode="same"),
        "rpe_abs_running": np.convolve(np.abs(rpes), k, mode="same"),
        "rpe_current": rpes,
        # Exponentially filtered versions
        "rpe_cumpos_filt": exp_filter(np.maximum(rpes, 0)),
        "rpe_cumneg_filt": exp_filter(np.minimum(rpes, 0)),
    })


def build_trans_m0(n):
    """LEGACY M0: constant-transition null model kept for reference only."""
    return np.ones((n, 1))


def compute_l0_baseline_ll(train_choices, test_choices, n_categories=2):
    """
    L0 baseline from the repository/paper.

    Estimate class probabilities from training non-violation choices and score the
    held-out non-violation test choices under that fixed categorical model.
    Returns normalized log-likelihood per valid test trial.
    """
    train_choices = np.asarray(train_choices).reshape(-1).astype(int)
    test_choices = np.asarray(test_choices).reshape(-1).astype(int)

    if len(train_choices) == 0 or len(test_choices) == 0:
        return np.nan

    train_counts = np.bincount(train_choices, minlength=n_categories).astype(float)
    test_counts = np.bincount(test_choices, minlength=n_categories).astype(float)
    train_probs = train_counts / max(train_counts.sum(), 1.0)
    train_probs = np.clip(train_probs, 1e-12, 1.0)

    ll_total = float(np.sum(test_counts * np.log(train_probs)))
    return ll_total / len(test_choices)


def build_trans_m2(rpe_df):
    """
    M2: RPE-based transitions.
    Regressors: cumpos, cumneg, abs_running, current (all RPE-based).
    """
    cols = ["rpe_cumpos_filt", "rpe_cumneg_filt", "rpe_abs_running", "rpe_current"]
    X = rpe_df[cols].fillna(0).values
    # Standardize
    X = (X - X.mean(0)) / (X.std(0) + 1e-8)
    # Add bias
    return np.column_stack([X, np.ones(len(rpe_df))])


def build_trans_m3(trans_mat_orig, rpe_df):
    """
    M3: Combined (M1 original + M2 RPE).
    Uses the original preprocessed transition regressors from Mohammadi et al.
    """
    # Use original transition regressors (already includes proper violation handling,
    # session boundaries, filtered past choice/stim_side/reward, and cosine basis)
    m1 = trans_mat_orig  # Keep all original regressors (no bias to exclude - original has none)
    m2 = build_trans_m2(rpe_df)  # includes bias
    return np.column_stack([m1, m2])


def permute_rpes(rpes, mode="circular_shift", rng=None):
    """
    Null generator for permutation test.
    Breaks temporal alignment while preserving structure.
    """
    rpes = np.asarray(rpes).copy()
    rng = rng or np.random.default_rng(0)
    
    if len(rpes) < 3:
        return rpes
    
    if mode == "circular_shift":
        shift = int(rng.integers(1, len(rpes)))
        return np.roll(rpes, shift)
    
    if mode == "shuffle":
        return rng.permutation(rpes)
    
    raise ValueError(f"Unknown mode: {mode}")


# ==============================================================================
# GLM-HMM FITTING (using ssm.HMM_TO)
# ==============================================================================

def fit_glmhmm(
    obs_inputs,      # (T, M_obs) observation regressors
    trans_inputs,    # (T, M_trans) transition regressors  
    choices,         # (T, 1) binary choices
    mask,            # (T,) violation mask
    K=N_STATES,
    n_iters=N_EM_ITERS_CV,
    prior_sigma=PRIOR_SIGMA,
    transition_alpha=TRANSITION_ALPHA,
):
    """
    Fit GLM-HMM using ssm.HMM_TO.
    Returns fitted model and final log-likelihood.
    """
    D = 1  # observation dimension
    C = 2  # number of choice categories
    M_obs = obs_inputs.shape[1]
    M_trans = trans_inputs.shape[1]
    
    # Wrap in lists (ssm expects list of sessions)
    obs_list = [obs_inputs]
    trans_list = [trans_inputs]
    choice_list = [choices.reshape(-1, 1) if choices.ndim == 1 else choices]
    mask_list = [mask.reshape(-1) if mask.ndim > 1 else mask]
    
    # Create GLM-HMM
    hmm = ssm.HMM_TO(
        K, D,
        M_trans=M_trans,
        M_obs=M_obs,
        observations="input_driven_obs_diff_inputs",
        observation_kwargs=dict(C=C, prior_sigma=prior_sigma),
        transitions="inputdrivenalt",
        transition_kwargs=dict(alpha=transition_alpha, kappa=0),
    )
    
    # Fit with EM
    lls = hmm.fit(
        choice_list,
        observation_input=obs_list,
        transition_input=trans_list,
        train_masks=mask_list,
        method="em",
        num_iters=n_iters,
        verbose=0,
        tolerance=1e-4,
    )
    
    final_ll = lls[-1] if len(lls) > 0 else np.nan
    return hmm, final_ll


def compute_test_ll(hmm, obs_inputs, trans_inputs, choices, mask):
    """Compute normalized log-likelihood on test data."""
    choices = choices.reshape(-1, 1) if choices.ndim == 1 else choices
    ll = hmm.log_likelihood(
        [choices],
        observation_input=[obs_inputs],
        transition_input=[trans_inputs],
    )
    n_valid = mask.sum()
    return ll / n_valid if n_valid > 0 else np.nan


def get_viterbi_states(hmm, obs_inputs, trans_inputs, choices, mask):
    """Get most likely state sequence via Viterbi."""
    choices = choices.reshape(-1, 1) if choices.ndim == 1 else choices
    states = hmm.most_likely_states(
        data=choices,
        transition_input=trans_inputs,
        observation_input=obs_inputs,
    )
    return states


def get_posteriors(hmm, obs_inputs, trans_inputs, choices, mask):
    """Get posterior state probabilities."""
    choices = choices.reshape(-1, 1) if choices.ndim == 1 else choices
    posteriors = hmm.expected_states(
        data=choices,
        transition_input=trans_inputs,
        observation_input=obs_inputs,
    )[0]  # (T, K)
    return posteriors


def identify_engaged_state(hmm):
    """
    Identify the engaged state as the one with highest contrast sensitivity.
    The observation weights are W_obs of shape (K, M_obs).
    Engaged state has highest absolute weight on contrast (first regressor).
    """
    W_obs = hmm.observations.params  # varies by ssm version
    if hasattr(hmm.observations, 'Wk'):
        W_obs = hmm.observations.Wk
    elif hasattr(hmm.observations, 'W'):
        W_obs = hmm.observations.W
    else:
        # Try to get from params tuple
        try:
            W_obs = hmm.observations.params[0]
        except:
            return 0  # fallback
    
    # Engaged state = highest |weight| on contrast (column 0)
    return int(np.argmax(np.abs(W_obs[:, 0])))


def get_transition_weights(hmm):
    """
    Extract transition weight matrix.
    Returns W_trans of shape (K, K, M_trans).
    """
    if hasattr(hmm.transitions, 'Ws'):
        return hmm.transitions.Ws
    elif hasattr(hmm.transitions, 'W'):
        return hmm.transitions.W
    else:
        try:
            return hmm.transitions.params[0]
        except:
            return None


# ==============================================================================
# PER-MOUSE ANALYSIS
# ==============================================================================

def _analyze_mouse_cv_worker(mouse_name, data_dir):
    """
    Worker function for parallel CV analysis.
    Creates its own RNG from mouse name hash.
    """
    rng = np.random.default_rng(hash(mouse_name) % 2**32)
    return analyze_mouse_cv(mouse_name, data_dir, rng)


def analyze_mouse_cv(mouse_name, data_dir, rng):
    """
    Cross-validated model comparison for one mouse.
    Returns per-session results for all models.
    """
    # Load data
    processed_file = data_dir / "separate_mouse_data" / f"{mouse_name}_processed.npz"
    rewarded_file = data_dir / "separate_mouse_data" / f"{mouse_name}_rewarded.npz"
    fold_map_file = data_dir / "separate_mouse_data" / f"{mouse_name}_fold_session_map.npz"
    
    if not processed_file.exists() or not rewarded_file.exists():
        print(f"  [{mouse_name}] SKIP: missing data files")
        return None
    
    # Load preprocessed data
    container = np.load(processed_file, allow_pickle=True)
    obs_mat = container["arr_0"]      # observation regressors
    trans_mat_orig = container["arr_1"]  # original transition regressors
    output = container["arr_2"]       # choices (T, 1)
    session_ids = container["arr_3"]  # session identifiers
    
    # Load rewards
    rew_container = np.load(rewarded_file, allow_pickle=True)
    rewards_raw = rew_container["arr_0"].flatten()
    
    # Convert rewards: feedbackType is {-1, 1} → we want {0, 1}
    rewards = ((rewards_raw + 1) / 2).astype(float)
    
    # Load or create fold mapping
    if fold_map_file.exists():
        fold_container = np.load(fold_map_file, allow_pickle=True)
        fold_map = fold_container["arr_0"]
    else:
        fold_map = divide_sessions_for_test_train(session_ids, N_FOLDS)
    
    # Prepare data
    choices = output.flatten().astype(int)
    valid_mask = (choices != -1).astype(int)
    choices_clean = np.where(choices == -1, 0, choices)
    
    # Add bias to obs_mat if needed
    if obs_mat.shape[1] < 5:
        obs_mat = np.hstack([obs_mat, np.ones((obs_mat.shape[0], 1))])
    
    valid_idx = np.where(valid_mask == 1)[0]
    if len(valid_idx) < 100:
        print(f"  [{mouse_name}] SKIP: only {len(valid_idx)} valid trials")
        return None
    
    # Fit Q-learning
    try:
        rl_params, rpes_q, values_q = fit_rl_model(
            choices_clean[valid_idx], rewards[valid_idx], rng=rng
        )
    except Exception as e:
        print(f"  [{mouse_name}] SKIP: RL fit failed: {e}")
        return None
    
    if check_boundary_hit(rl_params):
        print(f"  [{mouse_name}] SKIP: RL params hit boundary: {rl_params}")
        return None
    
    print(f"  [{mouse_name}] {len(valid_idx)} valid trials, RL params: "
          f"a+={rl_params[0]:.3f} a-={rl_params[1]:.3f} b={rl_params[2]:.2f}")
    
    # Expand RPEs to full length
    rpes_q_full = np.zeros(len(choices))
    rpes_q_full[valid_idx] = rpes_q
    
    # Fit belief-state model
    rpes_b_full = np.zeros(len(choices))
    belief_rpes, _, _ = fit_belief_model(rewards[valid_idx])
    rpes_b_full[valid_idx] = belief_rpes
    
    # Build all regressor sets
    rpe_df_q = compute_rpe_regressors(rpes_q_full)
    rpe_df_b = compute_rpe_regressors(rpes_b_full)
    
    # Legacy constant-transition M0 is intentionally not used.
    # L0 baseline is computed directly from training/test choice frequencies.
    # M1: Use original preprocessed transition regressors from Mohammadi et al.
    # This includes: filtered past choice, stim_side, past reward (all with proper
    # violation handling and session boundaries), plus raised cosine basis functions.
    trans_m1 = trans_mat_orig
    trans_m2_q = build_trans_m2(rpe_df_q)
    trans_m2_b = build_trans_m2(rpe_df_b)
    trans_m3_q = build_trans_m3(trans_mat_orig, rpe_df_q)
    trans_m3_b = build_trans_m3(trans_mat_orig, rpe_df_b)
    
    results = {
        "mouse": mouse_name,
        "rl_params": rl_params,
        "rpes_q": rpes_q,
        "belief_rpes": belief_rpes,
        "ll_m0": [], "ll_m1": [],
        "ll_m2_q": [], "ll_m2_b": [],
        "ll_m3_q": [], "ll_m3_b": [],
        "n_trials": [],
        "delta_ll_real": [],      # M2_q - M1
        "delta_ll_null": [],      # permutation nulls
        "sessions": [],
    }
    
    # Cross-validation
    unique_sessions = np.unique(session_ids)
    
    for fold in range(N_FOLDS):
        # Get train/test sessions
        test_sess = set(fold_map[fold_map[:, 1] == fold, 0])
        train_sess = set(fold_map[fold_map[:, 1] != fold, 0])
        
        train_idx = np.array([s in train_sess for s in session_ids]) & (valid_mask == 1)
        test_idx = np.array([s in test_sess for s in session_ids]) & (valid_mask == 1)
        
        if train_idx.sum() < 50 or test_idx.sum() < 20:
            continue
        
        # L0 baseline from the paper/repository (not an HMM fit)
        ll_m0 = compute_l0_baseline_ll(
            choices_clean[train_idx],
            choices_clean[test_idx],
            n_categories=2,
        )
        fold_lls = {"m0": ll_m0}
        results["ll_m0"].append(ll_m0)

        model_configs = [
            ("m1", trans_m1),
            ("m2_q", trans_m2_q),
            ("m2_b", trans_m2_b),
            ("m3_q", trans_m3_q),
            ("m3_b", trans_m3_b),
        ]

        fold_ok = int(np.isfinite(ll_m0))
        for name, trans_regs in model_configs:
            try:
                hmm, _ = fit_glmhmm(
                    obs_mat[train_idx], trans_regs[train_idx],
                    choices_clean[train_idx], valid_mask[train_idx],
                    n_iters=N_EM_ITERS_CV
                )
                test_ll = compute_test_ll(
                    hmm, obs_mat[test_idx], trans_regs[test_idx],
                    choices_clean[test_idx], valid_mask[test_idx]
                )
                fold_lls[name] = test_ll
                results[f"ll_{name}"].append(test_ll)
                fold_ok += 1
            except Exception as e:
                fold_lls[name] = np.nan
                results[f"ll_{name}"].append(np.nan)
                if name == "m1":  # only log first HMM model error per fold to avoid spam
                    print(f"  [{mouse_name}] fold {fold} model {name} FAILED: {e}")
        if fold_ok > 0:
            ll_str = ', '.join(f"{k}={v:.4f}" for k, v in fold_lls.items() if np.isfinite(v))
            print(f"  [{mouse_name}] fold {fold}: {fold_ok}/6 models OK  {ll_str}")
        
        results["n_trials"].append(test_idx.sum())
        
        # Real delta (M2_q - M1)
        delta_real = fold_lls.get("m2_q", np.nan) - fold_lls.get("m1", np.nan)
        results["delta_ll_real"].append(delta_real)
        
        # Permutation null for this fold
        null_deltas = []
        for _ in range(N_PERMUTATIONS):
            rpes_perm = permute_rpes(rpes_q_full, mode=NULL_MODE, rng=rng)
            rpe_df_perm = compute_rpe_regressors(rpes_perm)
            trans_m2_perm = build_trans_m2(rpe_df_perm)
            
            try:
                hmm_m1, _ = fit_glmhmm(
                    obs_mat[train_idx], trans_m1[train_idx],
                    choices_clean[train_idx], valid_mask[train_idx],
                    n_iters=N_EM_ITERS_CV
                )
                ll_m1 = compute_test_ll(
                    hmm_m1, obs_mat[test_idx], trans_m1[test_idx],
                    choices_clean[test_idx], valid_mask[test_idx]
                )
                
                hmm_m2, _ = fit_glmhmm(
                    obs_mat[train_idx], trans_m2_perm[train_idx],
                    choices_clean[train_idx], valid_mask[train_idx],
                    n_iters=N_EM_ITERS_CV
                )
                ll_m2 = compute_test_ll(
                    hmm_m2, obs_mat[test_idx], trans_m2_perm[test_idx],
                    choices_clean[test_idx], valid_mask[test_idx]
                )
                
                null_deltas.append(ll_m2 - ll_m1)
            except:
                null_deltas.append(np.nan)
        
        results["delta_ll_null"].append(null_deltas)
    
    return results


def _analyze_mouse_full_worker(mouse_name, data_dir):
    """
    Worker function for parallel full-session analysis.
    Creates its own RNG from mouse name hash.
    """
    rng = np.random.default_rng(hash(mouse_name) % 2**32)
    return analyze_mouse_full(mouse_name, data_dir, rng)


def analyze_mouse_full(mouse_name, data_dir, rng):
    """
    Full-session M2-Q fit for transition weight analysis (Analysis 2)
    and state sequence extraction (Analysis 3, 4).
    """
    # Load data (same as CV)
    processed_file = data_dir / "separate_mouse_data" / f"{mouse_name}_processed.npz"
    rewarded_file = data_dir / "separate_mouse_data" / f"{mouse_name}_rewarded.npz"
    
    if not processed_file.exists() or not rewarded_file.exists():
        print(f"  [{mouse_name}] SKIP full: missing data files")
        return None
    
    container = np.load(processed_file, allow_pickle=True)
    obs_mat = container["arr_0"]
    output = container["arr_2"]
    session_ids = container["arr_3"]
    left_probs = container["arr_4"] if len(container.files) > 4 else None
    
    rew_container = np.load(rewarded_file, allow_pickle=True)
    rewards_raw = rew_container["arr_0"].flatten()
    rewards = ((rewards_raw + 1) / 2).astype(float)
    
    choices = output.flatten().astype(int)
    valid_mask = (choices != -1).astype(int)
    choices_clean = np.where(choices == -1, 0, choices)
    
    if obs_mat.shape[1] < 5:
        obs_mat = np.hstack([obs_mat, np.ones((obs_mat.shape[0], 1))])
    
    valid_idx = np.where(valid_mask == 1)[0]
    if len(valid_idx) < 100:
        print(f"  [{mouse_name}] SKIP full: only {len(valid_idx)} valid trials")
        return None
    
    # Fit Q-learning
    try:
        rl_params, rpes_q, _ = fit_rl_model(
            choices_clean[valid_idx], rewards[valid_idx], rng=rng
        )
    except Exception as e:
        print(f"  [{mouse_name}] SKIP full: RL fit failed: {e}")
        return None
    
    if check_boundary_hit(rl_params):
        print(f"  [{mouse_name}] SKIP full: RL params hit boundary")
        return None
    
    rpes_q_full = np.zeros(len(choices))
    rpes_q_full[valid_idx] = rpes_q
    rpe_df_q = compute_rpe_regressors(rpes_q_full)
    trans_m2_q = build_trans_m2(rpe_df_q)
    
    results = {
        "mouse": mouse_name,
        "rl_params": rl_params,
        "W_obs_list": [],
        "W_trans_list": [],
        "states_list": [],
        "rpes_list": [],
        "block_list": [],
        "engaged_idx": [],
        "session_ids": [],
    }
    
    # Fit each session separately
    unique_sessions = np.unique(session_ids)
    for sess_id in unique_sessions:
        sess_idx = (session_ids == sess_id) & (valid_mask == 1)
        if sess_idx.sum() < 50:
            continue
        
        try:
            hmm, _ = fit_glmhmm(
                obs_mat[sess_idx], trans_m2_q[sess_idx],
                choices_clean[sess_idx], valid_mask[sess_idx],
                n_iters=N_EM_ITERS_FULL
            )
            
            states = get_viterbi_states(
                hmm, obs_mat[sess_idx], trans_m2_q[sess_idx],
                choices_clean[sess_idx], valid_mask[sess_idx]
            )
            
            eng_idx = identify_engaged_state(hmm)
            W_trans = get_transition_weights(hmm)
            
            results["W_trans_list"].append(W_trans)
            results["states_list"].append(states)
            results["rpes_list"].append(rpes_q_full[sess_idx])
            results["engaged_idx"].append(eng_idx)
            results["session_ids"].append(sess_id)
            
            # Block info if available
            if left_probs is not None:
                results["block_list"].append(left_probs[sess_idx])
            else:
                results["block_list"].append(None)
                
        except Exception as e:
            print(f"  [{mouse_name}] session {sess_id}: fit failed: {e}")
            continue
    
    print(f"  [{mouse_name}] full: {len(results['states_list'])}/{len(unique_sessions)} sessions fitted")
    return results


# ==============================================================================
# STATISTICS
# ==============================================================================

def bootstrap_mean_ci(arr, n_boot=BOOTSTRAP_N, ci=95):
    """Bootstrap mean with percentile CI and SE."""
    arr = np.asarray(arr, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return np.nan, np.nan, np.nan, np.nan
    
    rng = np.random.default_rng(42)
    boots = np.array([
        rng.choice(arr, size=len(arr), replace=True).mean()
        for _ in range(n_boot)
    ])
    
    lo = np.percentile(boots, (100 - ci) / 2)
    hi = np.percentile(boots, 100 - (100 - ci) / 2)
    se = float(np.std(boots, ddof=1))
    return float(np.mean(arr)), float(lo), float(hi), se


def wilcoxon_test(x):
    """Wilcoxon signed-rank test with rank-biserial effect size."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n_nonzero = int(np.sum(x != 0))
    if n_nonzero < 3:
        return np.nan, np.nan, np.nan, 0
    
    stat, p = wilcoxon(x, alternative="two-sided")
    # Rank-biserial correlation
    r_rb = 1.0 - (2.0 * stat) / (n_nonzero * (n_nonzero + 1))
    return stat, p, r_rb, n_nonzero


def compute_dwell_times(states, engaged_idx):
    """Compute dwell times in engaged state."""
    dwells = []
    count = 0
    in_eng = False
    for s in states:
        if s == engaged_idx:
            in_eng = True
            count += 1
        else:
            if in_eng and count > 0:
                dwells.append(count)
            in_eng = False
            count = 0
    if in_eng and count > 0:
        dwells.append(count)
    return dwells


# ==============================================================================
# ANALYSIS FUNCTIONS
# ==============================================================================

def analysis_1_model_comparison(all_cv_results):
    """
    Analysis 1: Model comparison (M2-Q vs M1).
    Tests whether RPE predicts state transitions better than raw reward.
    """
    print("\n" + "=" * 70)
    print("ANALYSIS 1 — Model Comparison: RPE vs Raw Reward Transitions")
    print("=" * 70)
    
    # Compute per-subject mean ΔLL
    subject_deltas = []
    subject_deltas_null = []
    
    for r in all_cv_results:
        if len(r["ll_m1"]) == 0:
            continue
        
        mean_m1 = np.nanmean(r["ll_m1"])
        mean_m2_q = np.nanmean(r["ll_m2_q"])
        delta = mean_m2_q - mean_m1
        subject_deltas.append(delta)
        
        # Collect null deltas
        for null_arr in r["delta_ll_null"]:
            subject_deltas_null.extend(null_arr)
    
    subject_deltas = np.array(subject_deltas)
    subject_deltas_null = np.array([x for x in subject_deltas_null if np.isfinite(x)])
    
    # Wilcoxon test
    stat, p, r_rb, n = wilcoxon_test(subject_deltas)
    
    # Bootstrap CI
    mean_delta, ci_lo, ci_hi, se = bootstrap_mean_ci(subject_deltas)
    
    # Permutation p-value
    if len(subject_deltas_null) > 0:
        perm_p = (1 + np.sum(subject_deltas_null >= mean_delta)) / (len(subject_deltas_null) + 1)
    else:
        perm_p = np.nan
    
    print(f"  N subjects: {n}")
    print(f"  Mean ΔLL (M2-Q − M1): {mean_delta:+.4f} bits/trial")
    print(f"  95% Bootstrap CI: [{ci_lo:+.4f}, {ci_hi:+.4f}], SE={se:.4f}")
    print(f"  Wilcoxon stat={stat:.1f}, p={p:.4e}, rank-biserial r={r_rb:+.3f}")
    print(f"  Permutation p={perm_p:.4e} (mode={NULL_MODE}, n_perm={N_PERMUTATIONS})")
    direction = "RPE > Raw Reward" if mean_delta > 0 else "Raw Reward ≥ RPE"
    print(f"  Direction: {direction}")
    
    return {
        "n_subjects": n,
        "mean_delta": mean_delta,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "se": se,
        "wilcoxon_stat": stat,
        "wilcoxon_p": p,
        "rank_biserial_r": r_rb,
        "perm_p": perm_p,
        "subject_deltas": subject_deltas,
        "null_deltas": subject_deltas_null,
    }


def analysis_1b_belief_vs_q(all_cv_results):
    """
    Analysis 1b: Belief-state vs Q-learning RPE comparison.
    """
    print("\n" + "=" * 70)
    print("ANALYSIS 1b — Belief-state vs Q-learning RPE (M2 variants)")
    print("=" * 70)
    
    subject_deltas = []
    for r in all_cv_results:
        if len(r["ll_m2_q"]) == 0 or len(r["ll_m2_b"]) == 0:
            continue
        delta = np.nanmean(r["ll_m2_b"]) - np.nanmean(r["ll_m2_q"])
        subject_deltas.append(delta)
    
    subject_deltas = np.array(subject_deltas)
    stat, p, r_rb, n = wilcoxon_test(subject_deltas)
    mean_delta, ci_lo, ci_hi, se = bootstrap_mean_ci(subject_deltas)
    
    print(f"  N subjects: {n}")
    print(f"  Mean ΔLL (Belief − Q): {mean_delta:+.4f} bits/trial")
    print(f"  95% Bootstrap CI: [{ci_lo:+.4f}, {ci_hi:+.4f}], SE={se:.4f}")
    print(f"  Wilcoxon stat={stat:.1f}, p={p:.4e}, rank-biserial r={r_rb:+.3f}")
    direction = "Belief > Q" if mean_delta > 0 else "Q ≥ Belief"
    print(f"  Direction: {direction}")
    
    return {
        "n_subjects": n,
        "mean_delta": mean_delta,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "se": se,
        "wilcoxon_stat": stat,
        "wilcoxon_p": p,
        "rank_biserial_r": r_rb,
    }


def analysis_2_transition_weights(all_full_results):
    """
    Analysis 2: RPE signatures at state transitions.
    Extracts transition weights for engaged↔disengaged directions.
    """
    print("\n" + "=" * 70)
    print("ANALYSIS 2 — RPE Signatures at Motivational State Transitions")
    print("=" * 70)
    
    # Collect weights: cumpos, cumneg, abs_running, current
    # For both directions: engaged→disengaged, disengaged→engaged
    e2d = {i: [] for i in range(4)}  # engaged to disengaged
    d2e = {i: [] for i in range(4)}  # disengaged to engaged
    
    n_3d, n_2d, n_skipped = 0, 0, 0

    for r in all_full_results:
        for W_trans, eng in zip(r["W_trans_list"], r["engaged_idx"]):
            if W_trans is None:
                n_skipped += 1
                continue

            W = np.asarray(W_trans)
            if W.ndim == 3:
                # Pair-specific transition weights: (K, K, M)
                n_3d += 1
                K = W.shape[0]
                dis_states = [s for s in range(K) if s != eng]

                for j in dis_states:
                    for reg_i in range(min(4, W.shape[2])):
                        e2d[reg_i].append(W[eng, j, reg_i])

                for i in dis_states:
                    for reg_i in range(min(4, W.shape[2])):
                        d2e[reg_i].append(W[i, eng, reg_i])

            elif W.ndim == 2:
                # inputdrivenalt stores destination-only weights:
                # usually (K-1, M), with implicit zero row for one reference state.
                # Build a K x M destination matrix so each state's incoming
                # sensitivity to regressors can be compared.
                n_2d += 1
                M = W.shape[1]

                if W.shape[0] == N_STATES - 1:
                    W_dest = np.vstack([W, np.zeros((1, M))])
                elif W.shape[0] == N_STATES:
                    W_dest = W
                else:
                    n_skipped += 1
                    continue

                K = W_dest.shape[0]
                if eng < 0 or eng >= K:
                    n_skipped += 1
                    continue

                dis_states = [s for s in range(K) if s != eng]

                # Engaged -> Disengaged: incoming weights to disengaged destinations.
                for j in dis_states:
                    for reg_i in range(min(4, M)):
                        e2d[reg_i].append(W_dest[j, reg_i])

                # Disengaged -> Engaged: source-invariant in this formulation,
                # so append the engaged destination weight once per fit.
                for reg_i in range(min(4, M)):
                    d2e[reg_i].append(W_dest[eng, reg_i])

            else:
                n_skipped += 1
                continue

    print(f"  Transition weight formats: 3D={n_3d}, 2D={n_2d}, skipped={n_skipped}")
    
    reg_names = ["cumpos", "cumneg", "abs_running", "current"]
    results = {"engaged_to_disengaged": {}, "disengaged_to_engaged": {}}
    
    for direction, data, label in [
        ("engaged_to_disengaged", e2d, "Engaged → Disengaged"),
        ("disengaged_to_engaged", d2e, "Disengaged → Engaged"),
    ]:
        print(f"\n  {label}:")
        for reg_i, reg_name in enumerate(reg_names):
            arr = np.array(data.get(reg_i, []))
            if len(arr) == 0:
                print(f"    {reg_name}: no data")
                continue
            mean, lo, hi, se = bootstrap_mean_ci(arr)
            sign = "positive" if mean > 0 else "negative"
            excludes_zero = "YES" if (lo > 0 or hi < 0) else "no"
            print(f"    {reg_name}: mean={mean:+.4f}, 95% CI [{lo:+.4f}, {hi:+.4f}], "
                  f"SE={se:.4f}, excludes 0: {excludes_zero}")
            results[direction][reg_name] = {
                "mean": mean, "ci_lo": lo, "ci_hi": hi, "se": se, "n": len(arr)
            }
    
    return results


def analysis_3_individual_differences(all_full_results, all_cv_results):
    """
    Analysis 3: Learning rate vs state stability correlation.
    Tests whether α⁺ correlates with engaged-state dwell time.
    """
    print("\n" + "=" * 70)
    print("ANALYSIS 3 — Individual Differences: Learning Rate vs State Stability")
    print("=" * 70)
    
    rows = []
    for r_full, r_cv in zip(all_full_results, all_cv_results):
        # Compute mean dwell time
        all_dwells = []
        for states, eng in zip(r_full["states_list"], r_full["engaged_idx"]):
            all_dwells.extend(compute_dwell_times(states, eng))
        
        if not all_dwells or r_cv["rl_params"] is None:
            continue
        
        rows.append({
            "mouse": r_full["mouse"],
            "alpha_pos": r_cv["rl_params"][0],
            "alpha_neg": r_cv["rl_params"][1],
            "beta": r_cv["rl_params"][2],
            "mean_dwell": np.mean(all_dwells),
            "median_dwell": np.median(all_dwells),
            "n_dwells": len(all_dwells),
        })
    
    df = pd.DataFrame(rows)
    
    if len(df) < 3:
        print("  Not enough subjects for correlation analysis")
        return df, np.nan, np.nan
    
    r, p = spearmanr(df["alpha_pos"], df["mean_dwell"])
    
    print(f"  N subjects: {len(df)}")
    print(f"  Spearman ρ (α⁺ vs mean dwell): {r:+.3f}")
    print(f"  p-value: {p:.4e}")
    print(f"  ρ² (variance explained): {r**2:.3f}")
    print(f"  Mean engaged dwell: {df['mean_dwell'].mean():.1f} trials")
    
    assoc = "positive" if r > 0 else "negative"
    print(f"  Association: {assoc}")
    
    return df, r, p


def analysis_4_block_boundaries(all_full_results):
    """
    Analysis 4: Peri-event RPE and engagement around block switches.
    """
    print("\n" + "=" * 70)
    print("ANALYSIS 4 — Block Boundary Analysis")
    print("=" * 70)
    
    rpe_events = []
    eng_events = []
    
    for r in all_full_results:
        for rpes, states, block, eng in zip(
            r["rpes_list"], r["states_list"], r["block_list"], r["engaged_idx"]
        ):
            if block is None:
                continue
            
            T = len(rpes)
            
            # Find block boundaries
            for t in range(1, len(block)):
                if np.isnan(block[t-1]) or np.isnan(block[t]):
                    continue
                if block[t-1] == block[t]:
                    continue
                
                b = t  # boundary trial
                if b - BOUNDARY_WINDOW < 0 or b + BOUNDARY_WINDOW >= T:
                    continue
                
                # Extract window
                rpe_window = rpes[b - BOUNDARY_WINDOW: b + BOUNDARY_WINDOW + 1]
                state_window = states[b - BOUNDARY_WINDOW: b + BOUNDARY_WINDOW + 1]
                eng_window = (state_window == eng).astype(float)
                
                rpe_events.append(rpe_window)
                eng_events.append(eng_window)
    
    if not rpe_events:
        print("  No block boundaries found")
        return None, None, None
    
    rpe_mat = np.array(rpe_events)
    eng_mat = np.array(eng_events)
    
    # RPE at boundary
    mid = BOUNDARY_WINDOW
    rpe_at_boundary = rpe_mat[:, mid]
    t_stat, t_p = ttest_1samp(rpe_at_boundary, popmean=0)
    
    # Engagement pre vs post
    pre_eng = eng_mat[:, :mid].mean()
    post_eng = eng_mat[:, mid+1:].mean()
    delta_eng = post_eng - pre_eng
    
    print(f"  Block boundaries found: {len(rpe_mat)}")
    print(f"  Mean RPE at boundary: {rpe_at_boundary.mean():+.4f}")
    print(f"  One-sample t-test vs 0: t={t_stat:.3f}, p={t_p:.4e}")
    print(f"  P(engaged) pre: {pre_eng:.3f}")
    print(f"  P(engaged) post: {post_eng:.3f}")
    print(f"  ΔP(engaged): {delta_eng:+.3f}")
    
    return rpe_mat, eng_mat, {
        "n_boundaries": len(rpe_mat),
        "mean_rpe_at_boundary": float(rpe_at_boundary.mean()),
        "t_stat": float(t_stat),
        "t_p": float(t_p),
        "p_engaged_pre": float(pre_eng),
        "p_engaged_post": float(post_eng),
        "delta_p_engaged": float(delta_eng),
    }


# ==============================================================================
# FIGURES
# ==============================================================================

def fig_model_comparison(summary_df, h1_results):
    """Figure: Model comparison boxplot + histogram."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Boxplot of all models
    ax = axes[0]
    models = ["ll_m0", "ll_m1", "ll_m2_q", "ll_m2_b", "ll_m3_q", "ll_m3_b"]
    labels = ["L0\nBase", "M1\nRaw", "M2-Q\nRPE", "M2-B\nBelief", "M3-Q\nComb", "M3-B\nComb"]
    colors = ["#bdbdbd", "#6baed6", "#fc8d59", "#756bb1", "#9ecae1", "#9e9ac8"]
    
    data = [summary_df[m].dropna().values for m in models]
    bp = ax.boxplot(data, patch_artist=True, labels=labels)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_ylabel("CV Log-Likelihood (bits/trial)")
    ax.set_title("Model Comparison Across Subjects", fontweight="bold")
    ax.axhline(summary_df["ll_m1"].mean(), color="gray", linestyle="--", alpha=0.5)
    sns.despine(ax=ax)
    
    # Histogram of ΔLL
    ax = axes[1]
    deltas = np.asarray(h1_results["subject_deltas"], dtype=float)
    deltas = deltas[np.isfinite(deltas)] if len(deltas) > 0 else deltas
    if len(deltas) == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
    else:
        ax.hist(deltas, bins=15, color="#fc8d59", alpha=0.7, edgecolor="white")
    if len(deltas) > 0:
        ax.axvline(0, color="black", linestyle="--", lw=2)
        ax.axvline(np.mean(deltas), color="red", lw=2, label=f"Mean={np.mean(deltas):+.4f}")
        ax.set_xlabel("ΔLL (M2-Q − M1)")
        ax.set_ylabel("Count (subjects)")
        p_val = h1_results['wilcoxon_p']
        p_str = f"{p_val:.3e}" if np.isfinite(p_val) else "N/A"
        ax.set_title(f"RPE vs Raw Reward\nWilcoxon p={p_str}", fontweight="bold")
        ax.legend()
    sns.despine(ax=ax)
    
    plt.tight_layout()
    fig.savefig(OUT_DIR / "fig1_model_comparison.png", dpi=FIGURE_DPI)
    plt.close()


def fig_transition_weights(h2_results):
    """Figure: Transition weight bar plots."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    reg_names = ["cumpos", "cumneg", "abs_running", "current"]
    labels = ["Cum Pos\nRPE", "Cum Neg\nRPE", "Abs Running\nRPE", "Current\nRPE"]
    
    for ax, (direction, title) in zip(axes, [
        ("engaged_to_disengaged", "Engaged → Disengaged"),
        ("disengaged_to_engaged", "Disengaged → Engaged"),
    ]):
        data = h2_results.get(direction, {})
        means = [data.get(r, {}).get("mean", 0) for r in reg_names]
        cis = [(data.get(r, {}).get("ci_lo", 0), data.get(r, {}).get("ci_hi", 0)) 
               for r in reg_names]
        errs = [[m - lo for m, (lo, hi) in zip(means, cis)],
                [hi - m for m, (lo, hi) in zip(means, cis)]]
        
        colors = ["#d73027" if m > 0 else "#4575b4" for m in means]
        ax.bar(labels, means, color=colors, alpha=0.7, edgecolor="white")
        ax.errorbar(range(4), means, yerr=errs, fmt="none", color="black", capsize=4)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_title(title, fontweight="bold")
        ax.set_ylabel("Transition Weight")
        sns.despine(ax=ax)
    
    plt.tight_layout()
    fig.savefig(OUT_DIR / "fig2_transition_weights.png", dpi=FIGURE_DPI)
    plt.close()


def fig_individual_differences(h3_df, r, p):
    """Figure: Learning rate vs dwell time scatter."""
    if len(h3_df) < 3:
        return
    
    fig, ax = plt.subplots(figsize=(6, 5))
    
    ax.scatter(h3_df["alpha_pos"], h3_df["mean_dwell"], 
               s=60, alpha=0.7, color="#fc8d59", edgecolor="white")
    
    # Regression line
    m, b = np.polyfit(h3_df["alpha_pos"], h3_df["mean_dwell"], 1)
    xs = np.linspace(h3_df["alpha_pos"].min(), h3_df["alpha_pos"].max(), 100)
    ax.plot(xs, m * xs + b, color="#d95f02", lw=2)
    
    ax.set_xlabel("α⁺ (positive learning rate)")
    ax.set_ylabel("Mean Engaged Dwell Time (trials)")
    ax.set_title(f"Learning Rate vs State Stability\nSpearman ρ={r:+.3f}, p={p:.3e}", 
                 fontweight="bold")
    sns.despine()
    
    plt.tight_layout()
    fig.savefig(OUT_DIR / "fig3_individual_differences.png", dpi=FIGURE_DPI)
    plt.close()


def fig_block_boundaries(rpe_mat, eng_mat, h4_stats):
    """Figure: Peri-boundary RPE and engagement."""
    if rpe_mat is None:
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    offsets = np.arange(-BOUNDARY_WINDOW, BOUNDARY_WINDOW + 1)
    
    # RPE
    ax = axes[0]
    mean_rpe = rpe_mat.mean(0)
    sem_rpe = rpe_mat.std(0) / np.sqrt(len(rpe_mat))
    ax.fill_between(offsets, mean_rpe - sem_rpe, mean_rpe + sem_rpe, alpha=0.3, color="#2171b5")
    ax.plot(offsets, mean_rpe, color="#2171b5", lw=2)
    ax.axvline(0, color="black", linestyle="--", lw=1, label="Block boundary")
    ax.axhline(0, color="gray", linestyle="--", lw=0.5)
    ax.set_xlabel("Trials from block boundary")
    ax.set_ylabel("Mean RPE")
    ax.set_title("RPE Around Block Switches", fontweight="bold")
    ax.legend()
    sns.despine(ax=ax)
    
    # Engagement
    ax = axes[1]
    mean_eng = eng_mat.mean(0)
    sem_eng = eng_mat.std(0) / np.sqrt(len(eng_mat))
    ax.fill_between(offsets, mean_eng - sem_eng, mean_eng + sem_eng, alpha=0.3, color="#238b45")
    ax.plot(offsets, mean_eng, color="#238b45", lw=2)
    ax.axvline(0, color="black", linestyle="--", lw=1)
    ax.axhline(h4_stats["p_engaged_pre"], xmin=0, xmax=0.48, color="gray", 
               linestyle=":", label=f"Pre={h4_stats['p_engaged_pre']:.2f}")
    ax.axhline(h4_stats["p_engaged_post"], xmin=0.52, xmax=1, color="gray", 
               linestyle=":", label=f"Post={h4_stats['p_engaged_post']:.2f}")
    ax.set_xlabel("Trials from block boundary")
    ax.set_ylabel("P(Engaged State)")
    ax.set_title("Engagement Around Block Switches", fontweight="bold")
    ax.legend()
    sns.despine(ax=ax)
    
    plt.tight_layout()
    fig.savefig(OUT_DIR / "fig4_block_boundaries.png", dpi=FIGURE_DPI)
    plt.close()


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    print("=" * 70)
    print("RPE-based GLM-HMM Full Analysis")
    print(f"Date: {datetime.now(timezone.utc).isoformat()}")
    print(f"SSM package: {ssm.__file__}")
    print("=" * 70)
    
    # Load mouse list
    mice_file = DATA_DIR / "separate_mouse_data" / "mice_names.npz"
    if not mice_file.exists():
        raise FileNotFoundError(
            f"mice_names.npz not found at {mice_file}\n"
            "Run preprocessing first."
        )
    
    mice_names = mice_names_info(str(mice_file))
    if N_MICE is not None:
        mice_names = mice_names[:N_MICE]
    
    print(f"\nAnalyzing {len(mice_names)} mice...")
    print(f"Models: L0, M1, M2-Q, M2-Belief, M3-Q, M3-Belief")
    print(f"K={N_STATES} states, {N_FOLDS}-fold CV")
    print(f"Using {N_WORKERS} parallel workers\n")
    print(f"Live numeric outputs: {LIVE_DIR}")
    
    # Run CV analysis for all mice in parallel
    all_cv_results = []
    cv_worker = partial(_analyze_mouse_cv_worker, data_dir=DATA_DIR)
    
    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = {executor.submit(cv_worker, mouse): mouse for mouse in mice_names}
        for future in tqdm(as_completed(futures), total=len(futures), desc="CV fitting"):
            mouse = futures[future]
            try:
                result = future.result()
                if result is not None:
                    all_cv_results.append(result)

                    # Save per-mouse result immediately
                    save_json(LIVE_DIR / f"cv_mouse_{mouse}.json", result)

                    # Save aggregate CV results and partial summary immediately
                    save_json(LIVE_DIR / "cv_all_partial.json", all_cv_results)
                    summary_partial = build_summary_df(all_cv_results)
                    summary_partial.to_csv(LIVE_DIR / "summary_partial.csv", index=False)

                    mean_m1 = float(np.nanmean(result["ll_m1"])) if len(result["ll_m1"]) > 0 else np.nan
                    mean_m2q = float(np.nanmean(result["ll_m2_q"])) if len(result["ll_m2_q"]) > 0 else np.nan
                    print(f"[LIVE][CV] saved {mouse}: folds={len(result['ll_m1'])}, "
                          f"mean_m1={mean_m1:+.4f}, mean_m2q={mean_m2q:+.4f}")
                else:
                    save_json(LIVE_DIR / f"cv_mouse_{mouse}.json", {
                        "mouse": mouse,
                        "status": "skipped_or_failed",
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    })
                    print(f"[LIVE][CV] {mouse}: skipped_or_failed")
            except Exception as e:
                print(f"CV analysis failed for {mouse}: {e}")
                save_json(LIVE_DIR / f"cv_mouse_{mouse}.json", {
                    "mouse": mouse,
                    "status": "exception",
                    "error": str(e),
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                })
    
    print(f"\nSuccessfully analyzed {len(all_cv_results)} mice")
    
    # Run full-session analysis in parallel
    all_full_results = []
    full_worker = partial(_analyze_mouse_full_worker, data_dir=DATA_DIR)
    
    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = {executor.submit(full_worker, mouse): mouse for mouse in mice_names}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Full-session fitting"):
            mouse = futures[future]
            try:
                result = future.result()
                if result is not None:
                    all_full_results.append(result)
                    save_json(LIVE_DIR / f"full_mouse_{mouse}.json", result)
                    save_json(LIVE_DIR / "full_all_partial.json", all_full_results)
                    print(f"[LIVE][FULL] saved {mouse}: sessions_fitted={len(result['states_list'])}")
                else:
                    save_json(LIVE_DIR / f"full_mouse_{mouse}.json", {
                        "mouse": mouse,
                        "status": "skipped_or_failed",
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    })
                    print(f"[LIVE][FULL] {mouse}: skipped_or_failed")
            except Exception as e:
                print(f"Full-session analysis failed for {mouse}: {e}")
                save_json(LIVE_DIR / f"full_mouse_{mouse}.json", {
                    "mouse": mouse,
                    "status": "exception",
                    "error": str(e),
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                })
    
    # Build summary DataFrame
    summary_df = build_summary_df(all_cv_results)
    
    # Save summary
    summary_df.to_csv(OUT_DIR / "summary.csv", index=False)
    summary_df.to_csv(LIVE_DIR / "summary_final.csv", index=False)
    print(f"\nSummary saved to {OUT_DIR / 'summary.csv'}")
    
    # Run analyses
    h1_results = analysis_1_model_comparison(all_cv_results)
    save_json(LIVE_DIR / "analysis_1.json", h1_results)
    print(f"[LIVE][ANALYSIS] saved analysis_1.json")

    h1b_results = analysis_1b_belief_vs_q(all_cv_results)
    save_json(LIVE_DIR / "analysis_1b.json", h1b_results)
    print(f"[LIVE][ANALYSIS] saved analysis_1b.json")

    h2_results = analysis_2_transition_weights(all_full_results)
    save_json(LIVE_DIR / "analysis_2.json", h2_results)
    print(f"[LIVE][ANALYSIS] saved analysis_2.json")

    h3_df, h3_r, h3_p = analysis_3_individual_differences(all_full_results, all_cv_results)
    save_json(LIVE_DIR / "analysis_3.json", {
        "table": h3_df,
        "spearman_r": h3_r,
        "spearman_p": h3_p,
    })
    print(f"[LIVE][ANALYSIS] saved analysis_3.json")

    rpe_mat, eng_mat, h4_stats = analysis_4_block_boundaries(all_full_results)
    save_json(LIVE_DIR / "analysis_4.json", {
        "rpe_mat": rpe_mat,
        "eng_mat": eng_mat,
        "stats": h4_stats,
    })
    print(f"[LIVE][ANALYSIS] saved analysis_4.json")
    
    # Generate figures
    fig_model_comparison(summary_df, h1_results)
    fig_transition_weights(h2_results)
    fig_individual_differences(h3_df, h3_r, h3_p)
    if h4_stats is not None:
        fig_block_boundaries(rpe_mat, eng_mat, h4_stats)
    
    # Save all results
    results = {
        "config": {
            "N_STATES": N_STATES,
            "N_FOLDS": N_FOLDS,
            "N_EM_ITERS_CV": N_EM_ITERS_CV,
            "N_EM_ITERS_FULL": N_EM_ITERS_FULL,
            "N_PERMUTATIONS": N_PERMUTATIONS,
            "RPE_WINDOW": RPE_WINDOW,
            "TAU_FILTER": TAU_FILTER,
            "BOUNDARY_WINDOW": BOUNDARY_WINDOW,
            "baseline_model": "L0 training-set choice-frequency baseline",
        },
        "analysis_1": h1_results,
        "analysis_1b": h1b_results,
        "analysis_2": h2_results,
        "analysis_3": {"spearman_r": h3_r, "spearman_p": h3_p},
        "analysis_4": h4_stats,
    }

    save_json(OUT_DIR / "results.json", results)
    save_json(LIVE_DIR / "results_final.json", results)
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"Outputs saved to: {OUT_DIR}")
    for f in sorted(OUT_DIR.glob("*")):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
