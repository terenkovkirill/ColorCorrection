from typing import List, Tuple

import numpy as np

from metrics import xyz_to_lab, deltaE2000, deltae_stats


VALID_METHODS = ("ls", "ls-p", "ls-rp")


def _check_rgb_matrix(P: np.ndarray) -> np.ndarray:
    P = np.asarray(P, float)
    if P.ndim != 2 or P.shape[1] != 3:
        raise ValueError("P must be (n,3)")
    return P


def feature_dim(method: str) -> int:
    method = method.lower()
    if method == "ls":
        return 3
    if method == "ls-p":
        return 10
    if method == "ls-rp":
        return 6
    raise ValueError(f"Unknown method '{method}'. Valid values: {VALID_METHODS}")


def expand_features(P: np.ndarray, method: str = "ls") -> np.ndarray:
    P = _check_rgb_matrix(P)
    method = method.lower()

    R = P[:, 0]
    G = P[:, 1]
    B = P[:, 2]

    if method == "ls":
        return P

    if method == "ls-p":
        return np.column_stack([
            R * R,
            G * G,
            B * B,
            R * G,
            R * B,
            G * B,
            R,
            G,
            B,
            np.ones_like(R),
        ])

    if method == "ls-rp":
        return np.column_stack([
            R,
            G,
            B,
            np.sqrt(np.clip(R * G, 0.0, None)),
            np.sqrt(np.clip(R * B, 0.0, None)),
            np.sqrt(np.clip(G * B, 0.0, None)),
        ])

    raise ValueError(f"Unknown method '{method}'. Valid values: {VALID_METHODS}")


def fit_ls_matrix(P: np.ndarray, X: np.ndarray, ridge: float = 0.0, method: str = "ls") -> np.ndarray:
    X = np.asarray(X, float)
    if X.ndim != 2 or X.shape[1] != 3:
        raise ValueError("X must be (n,3)")

    Phi = expand_features(P, method=method)
    if X.shape[0] != Phi.shape[0]:
        raise ValueError("X must have the same number of rows as P")

    if ridge > 0:
        d = Phi.shape[1]
        A = Phi.T @ Phi + ridge * np.eye(d)
        B = Phi.T @ X
        M = np.linalg.solve(A, B)
    else:
        M = np.linalg.pinv(Phi) @ X

    return M


def predict_xyz(P: np.ndarray, M: np.ndarray, method: str = "ls") -> np.ndarray:
    Phi = expand_features(P, method=method)
    M = np.asarray(M, float)
    if M.ndim != 2 or M.shape[0] != Phi.shape[1] or M.shape[1] != 3:
        raise ValueError(f"M must have shape ({Phi.shape[1]}, 3) for method '{method}'")
    return Phi @ M


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


def run_5fold(P: np.ndarray, X: np.ndarray, ridge: float = 0.0, seed: int = 42,
              method: str = "ls") -> dict:
    maes = []
    for tr, te in kfold_indices(P.shape[0], k=5, seed=seed):
        M = fit_ls_matrix(P[tr], X[tr], ridge=ridge, method=method)
        Xp = predict_xyz(P[te], M, method=method)
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
                         ridge: float = 0.0, seed: int = 42,
                         method: str = "ls") -> dict:
    fold_mean = []
    fold_p95 = []

    for tr, te in kfold_indices(P.shape[0], k=5, seed=seed):
        M = fit_ls_matrix(P[tr], X[tr], ridge=ridge, method=method)
        Xp = predict_xyz(P[te], M, method=method)

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