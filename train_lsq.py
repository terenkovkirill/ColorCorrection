from typing import List, Tuple

import numpy as np

from metrics import xyz_to_lab, deltaE2000, deltae_stats


def fit_ls_matrix(P: np.ndarray, X: np.ndarray, ridge: float = 0.0) -> np.ndarray:
    P = np.asarray(P, float)
    X = np.asarray(X, float)
    if P.ndim != 2 or P.shape[1] != 3:
        raise ValueError("P must be (n,3)")
    if X.ndim != 2 or X.shape != P.shape:
        raise ValueError("X must be (n,3) and same shape as P")

    PtP = P.T @ P
    if ridge > 0:
        PtP = PtP + ridge * np.eye(3)

    M = np.linalg.solve(PtP, P.T @ X)  # (3,3)
    return M


def predict_xyz(P: np.ndarray, M: np.ndarray) -> np.ndarray:
    return np.asarray(P, float) @ np.asarray(M, float)


def kfold_indices(n: int, k: int = 5, seed: int = 42) -> List[Tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    folds = np.array_split(idx, k)
    splits = []
    for i in range(k):
        test = folds[i]
        train = np.concatenate([folds[j] for j in range(k) if j != i])
        splits.append((train, test))
    return splits


def mean_abs_error(A: np.ndarray, B: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(A, float) - np.asarray(B, float))))


def run_5fold(P: np.ndarray, X: np.ndarray, ridge: float = 0.0, seed: int = 42) -> dict:
    maes = []
    for tr, te in kfold_indices(P.shape[0], k=5, seed=seed):
        M = fit_ls_matrix(P[tr], X[tr], ridge=ridge)
        Xp = predict_xyz(P[te], M)
        maes.append(mean_abs_error(Xp, X[te]))
    maes = np.asarray(maes, float)
    return {
        "fold_mae": maes,
        "mean": float(maes.mean()),
        "median": float(np.median(maes)),
        "min": float(maes.min()),
        "max": float(maes.max()),
    }

def run_5fold_deltaE2000(P: np.ndarray, X: np.ndarray, XYZ_white: np.ndarray,
                         ridge: float = 0.0, seed: int = 42) -> dict:
    fold_mean = []
    fold_p95 = []
    for tr, te in kfold_indices(P.shape[0], k=5, seed=seed):
        M = fit_ls_matrix(P[tr], X[tr], ridge=ridge)
        Xp = predict_xyz(P[te], M)

        Lab_p = xyz_to_lab(Xp, XYZ_white)
        Lab_t = xyz_to_lab(X[te], XYZ_white)
        dE = deltaE2000(Lab_p, Lab_t)

        st = deltae_stats(dE)
        fold_mean.append(st["mean"])
        fold_p95.append(st["p95"])

    fold_mean = np.asarray(fold_mean, float)
    fold_p95 = np.asarray(fold_p95, float)

    return {
        "fold_mean": fold_mean,
        "fold_p95": fold_p95,
        "mean_of_means": float(np.mean(fold_mean)),
        "median_of_means": float(np.median(fold_mean)),
        "mean_of_p95": float(np.mean(fold_p95)),
        "max_p95": float(np.max(fold_p95)),
    }