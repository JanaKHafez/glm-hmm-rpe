#!/usr/bin/env python3
"""Inspect individual IBL GLM-HMM fit outputs.

Usage examples:
  python inspect_indiv_results.py --animal DY_011
  python inspect_indiv_results.py --animal DY_011 --K 3 --fold 0 --iter 0

This script prints:
- where it found the saved .npz
- EM objective trace summary (final LP, last-10 delta, monotonicity)
- parameter shapes (init probs, transition matrices/weights, observation weights)
- cross-validation bits/trial (if diff_folds_fit.npz exists)

It is robust to params being saved as a ragged object array (dtype=object).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

def _detect_repo_root(start: Path) -> Path:
    """Walk upwards to find the project root.

    Heuristics: look for environment.yml and the results/ + data/ folders.
    This makes the script runnable from any subdirectory.
    """
    start = start.resolve()
    for p in [start] + list(start.parents):
        if (p / "environment.yml").exists() and (p / "results").exists() and (p / "data").exists():
            return p
    # Fallback: assume this file is inside the repo; go one level up.
    return start.parent


REPO_ROOT = _detect_repo_root(Path(__file__).resolve().parent)


@dataclass(frozen=True)
class FitSpec:
    animal: str
    K: int
    fold: int
    iter: int
    num_inputs: int
    prior_sigma: float
    transition_alpha: float


def _params_to_py(params):
    """Convert params saved as object ndarray back to nested python lists."""
    if isinstance(params, np.ndarray) and params.dtype == object:
        return params.tolist()
    return params


def find_fit_file(spec: FitSpec) -> Path:
    base = (
        REPO_ROOT
        / "results"
        / "model_indiv_ibl"
        / f"num_regress_obs_{spec.num_inputs}"
        / f"prior_sigma_{spec.prior_sigma}_transition_alpha_{spec.transition_alpha}"
        / spec.animal
        / "Model"
        / f"glmhmm_#state={spec.K}"
        / f"fld_num={spec.fold}"
        / f"iter_{spec.iter}"
        / f"glm_hmm_raw_parameters_itr_{spec.iter}.npz"
    )
    return base


def find_first_fit_file(
    animal: str,
    num_inputs: int,
    prior_sigma: float,
    transition_alpha: float,
    K: Optional[int],
    fold: Optional[int],
    iter: Optional[int],
) -> Optional[Path]:
    """Find the first fit file matching inputs; if K/fold/iter omitted, glob."""

    base = (
        REPO_ROOT
        / "results"
        / "model_indiv_ibl"
        / f"num_regress_obs_{num_inputs}"
        / f"prior_sigma_{prior_sigma}_transition_alpha_{transition_alpha}"
        / animal
        / "Model"
    )
    if not base.exists():
        return None

    k_glob = f"glmhmm_#state={K}" if K is not None else "glmhmm_#state=*"
    fold_glob = f"fld_num={fold}" if fold is not None else "fld_num=*"
    iter_glob = f"iter_{iter}" if iter is not None else "iter_*"

    pattern = str(base / k_glob / fold_glob / iter_glob / "glm_hmm_raw_parameters_itr_*.npz")
    matches = sorted(Path(p) for p in base.glob(f"{k_glob}/{fold_glob}/{iter_glob}/glm_hmm_raw_parameters_itr_*.npz"))
    # If the above glob doesn't work (on older Python), fall back to rglob by string pattern
    if not matches:
        matches = sorted(base.rglob("glm_hmm_raw_parameters_itr_*.npz"))
        # filter roughly
        filtered = []
        for m in matches:
            s = str(m)
            if K is not None and f"glmhmm_#state={K}" not in s:
                continue
            if fold is not None and f"fld_num={fold}" not in s:
                continue
            if iter is not None and f"iter_{iter}" not in s:
                continue
            filtered.append(m)
        matches = filtered

    if not matches:
        return None
    return matches[0]


def load_fit_npz(path: Path):
    d = np.load(str(path), allow_pickle=True)
    # expected order: [params, lls]
    params, lls = [d[k] for k in d.files]
    params = _params_to_py(params)
    return params, np.asarray(lls)


def summarize_lls(lls: np.ndarray) -> str:
    if lls.size == 0:
        return "No lls saved."

    final_lp = float(lls[-1])
    last10_delta = float(lls[-1] - lls[-10]) if lls.size >= 10 else float("nan")
    monotone = bool(np.all(np.diff(lls) >= -1e-8))
    return (
        f"n_em_iters={lls.size} final_LP={final_lp:.3f} "
        f"delta_last10={last10_delta:.3f} monotone_nondec={monotone}"
    )


def summarize_params(params) -> str:
    try:
        pi0 = np.array(params[0][0])
        log_Ps = np.array(params[1][0])
        trans_W = np.array(params[1][1])
        obs_W = np.array(params[2])
    except Exception as e:
        return f"Could not parse params structure: {type(e).__name__}: {e}"

    return (
        "params shapes:\n"
        f"  init_state_dist: {pi0.shape}\n"
        f"  log_transition_matrix: {log_Ps.shape}\n"
        f"  transition_weights: {trans_W.shape}\n"
        f"  obs_weights: {obs_W.shape}"
    )


def load_cv_diff(path_analysis_glm_hmm: Path) -> Optional[np.ndarray]:
    cv = path_analysis_glm_hmm / "diff_folds_fit.npz"
    if not cv.exists():
        return None
    d = np.load(str(cv), allow_pickle=True)
    diff = d[d.files[0]]
    return diff


def print_cv_summary(diff: np.ndarray):
    print("CV (diff_folds_fit) shape:", diff.shape)

    # Per `cross_validation_indiv_fit.py`:
    # - row 0: GLM
    # - rows 1-2: reserved (often zeros)
    # - GLM-HMM with K states is stored at row = 3 + (K-2)
    def row_label(i: int) -> str:
        if i == 0:
            return "GLM"
        if i in (1, 2):
            return "(reserved)"
        K = i - 1
        return f"GLM-HMM K={K}"

    print("CV bits/trial (per fold) and mean±std; NaN means missing fit:")
    for i in range(diff.shape[0]):
        row = diff[i, :]
        valid = np.isfinite(row)
        if not np.any(valid):
            mean = float("nan")
            std = float("nan")
        else:
            mean = float(np.mean(row[valid]))
            std = float(np.std(row[valid]))
        folds_str = " ".join([f"{x:.3f}" if np.isfinite(x) else "nan" for x in row.tolist()])
        print(f"  row {i} [{row_label(i)}] folds=[{folds_str}] mean±std={mean:.3f}±{std:.3f}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Inspect saved individual GLM-HMM results.")
    ap.add_argument("--animal", required=True, help="Animal name, e.g. DY_011")
    ap.add_argument("--K", type=int, default=None, help="Number of states (optional).")
    ap.add_argument("--fold", type=int, default=None, help="Fold index (optional).")
    ap.add_argument("--iter", type=int, default=None, help="Initialization index (optional).")
    ap.add_argument("--num_inputs", type=int, default=4)
    ap.add_argument("--prior_sigma", type=float, default=4.0)
    ap.add_argument("--transition_alpha", type=float, default=2.0)

    args = ap.parse_args()

    # Resolve fit path
    if args.K is not None and args.fold is not None and args.iter is not None:
        spec = FitSpec(
            animal=args.animal,
            K=args.K,
            fold=args.fold,
            iter=args.iter,
            num_inputs=args.num_inputs,
            prior_sigma=args.prior_sigma,
            transition_alpha=args.transition_alpha,
        )
        fit_path = find_fit_file(spec)
        if not fit_path.exists():
            print("Fit file not found:")
            print(" ", fit_path)
            return 2
    else:
        fit_path = find_first_fit_file(
            animal=args.animal,
            num_inputs=args.num_inputs,
            prior_sigma=args.prior_sigma,
            transition_alpha=args.transition_alpha,
            K=args.K,
            fold=args.fold,
            iter=args.iter,
        )
        if fit_path is None:
            print("No fit files found for:")
            print(f"  animal={args.animal}")
            print(f"  num_inputs={args.num_inputs} prior_sigma={args.prior_sigma} transition_alpha={args.transition_alpha}")
            print("Looked under:")
            print(
                " ",
                (
                    REPO_ROOT
                    / "results"
                    / "model_indiv_ibl"
                    / f"num_regress_obs_{args.num_inputs}"
                    / f"prior_sigma_{args.prior_sigma}_transition_alpha_{args.transition_alpha}"
                    / args.animal
                ),
            )
            return 2

    print("Loaded fit:")
    print(" ", fit_path)
    params, lls = load_fit_npz(fit_path)
    print(summarize_lls(lls))
    print(summarize_params(params))

    # CV summary if available
    animal_root = (
        REPO_ROOT
        / "results"
        / "model_indiv_ibl"
        / f"num_regress_obs_{args.num_inputs}"
        / f"prior_sigma_{args.prior_sigma}_transition_alpha_{args.transition_alpha}"
        / args.animal
    )
    diff = load_cv_diff(animal_root)
    if diff is None:
        print("No CV file found (diff_folds_fit.npz).")
        print("If you ran CV, it should be here:")
        print(" ", animal_root / "diff_folds_fit.npz")
    else:
        print_cv_summary(diff)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
