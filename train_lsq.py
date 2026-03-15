from typing import List, Tuple

import numpy as np
from scipy.optimize import minimize

from metrics import xyz_to_lab, deltaE2000, deltae_stats


VALID_METHODS = ("ls", "ls-p", "ls-rp", "ls-opt", "ls-p-opt", "ls-rp-opt")
VALID_OPT_LOSSES = ("lab", "de2000")


def _base_method(method: str) -> str:
    method = method.lower()
    if method not in VALID_METHODS:
        raise ValueError(f"Unknown method '{method}'. Valid values: {VALID_METHODS}")
    return method[:-4] if method.endswith("-opt") else method


def _is_opt_method(method: str) -> bool:
    return method.lower().endswith("-opt")


def _check_rgb_matrix(P: np.ndarray) -> np.ndarray:
    P = np.asarray(P, float)
    if P.ndim != 2 or P.shape[1] != 3:
        raise ValueError("P must be (n, 3)")
    return P


def expand_features(P: np.ndarray, method: str = "ls") -> np.ndarray:
    """
    Feature maps:
      ls      -> [R, G, B]
      ls-p    -> [R^2, G^2, B^2, RG, RB, GB, R, G, B, 1]
      ls-rp   -> [R, G, B, sqrt(RG), sqrt(RB), sqrt(GB)]
    """
    P = _check_rgb_matrix(P)
    base = _base_method(method)

    R = P[:, 0]
    G = P[:, 1]
    B = P[:, 2]

    if base == "ls":
        return P

    if base == "ls-p":
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

    if base == "ls-rp":
        return np.column_stack([
            R,
            G,
            B,
            np.sqrt(np.clip(R * G, 0.0, None)),
            np.sqrt(np.clip(R * B, 0.0, None)),
            np.sqrt(np.clip(G * B, 0.0, None)),
        ])

    raise ValueError(f"Unknown method '{method}'. Valid values: {VALID_METHODS}")


def _fit_closed_form_from_features(Phi: np.ndarray, X: np.ndarray, ridge: float = 0.0) -> np.ndarray:
    Phi = np.asarray(Phi, float)
    X = np.asarray(X, float)

    if Phi.ndim != 2:
        raise ValueError("Phi must be 2D")
    if X.ndim != 2 or X.shape[1] != 3:
        raise ValueError("X must be (n, 3)")
    if Phi.shape[0] != X.shape[0]:
        raise ValueError("Phi and X must have the same number of rows")

    d = Phi.shape[1]

    if ridge > 0:
        A = Phi.T @ Phi + ridge * np.eye(d)
        B = Phi.T @ X
        return np.linalg.solve(A, B)

    return np.linalg.pinv(Phi) @ X


def _lab_triplet_norm(Lab_pred: np.ndarray, Lab_true: np.ndarray) -> np.ndarray:
    D = np.asarray(Lab_pred, float) - np.asarray(Lab_true, float)
    return np.sqrt(np.sum(D * D, axis=1))


def _objective(
    params: np.ndarray,
    Phi: np.ndarray,
    X: np.ndarray,
    XYZ_white: np.ndarray,
    opt_loss: str,
) -> float:
    d = Phi.shape[1]
    M = np.asarray(params, float).reshape(d, 3)
    Xp = Phi @ M

    Lab_p = xyz_to_lab(Xp, XYZ_white)
    Lab_t = xyz_to_lab(X, XYZ_white)

    if opt_loss == "lab":
        return float(np.mean(_lab_triplet_norm(Lab_p, Lab_t)))

    if opt_loss == "de2000":
        return float(np.mean(deltaE2000(Lab_p, Lab_t)))

    raise ValueError(f"Unknown opt_loss '{opt_loss}'. Valid values: {VALID_OPT_LOSSES}")


def fit_ls_matrix(
    P: np.ndarray,
    X: np.ndarray,
    ridge: float = 0.0,
    method: str = "ls",
    XYZ_white: np.ndarray | None = None,
    opt_loss: str = "lab",
    opt_maxiter: int = 3000,
    opt_xatol: float = 1e-6,
    opt_fatol: float = 1e-6,
) -> np.ndarray:
    X = np.asarray(X, float)
    if X.ndim != 2 or X.shape[1] != 3:
        raise ValueError("X must be (n, 3)")

    opt_loss = opt_loss.lower()
    if opt_loss not in VALID_OPT_LOSSES:
        raise ValueError(f"Unknown opt_loss '{opt_loss}'. Valid values: {VALID_OPT_LOSSES}")

    Phi = expand_features(P, method=method)
    if Phi.shape[0] != X.shape[0]:
        raise ValueError("P and X must have the same number of rows")

    M0 = _fit_closed_form_from_features(Phi, X, ridge=ridge)

    if not _is_opt_method(method):
        return M0

    if XYZ_white is None:
        raise ValueError("XYZ_white is required for optimisation-based methods")

    result = minimize(
        _objective,
        x0=M0.reshape(-1),
        args=(Phi, X, np.asarray(XYZ_white, float), opt_loss),
        method="Nelder-Mead",
        options={
            "maxiter": int(opt_maxiter),
            "xatol": float(opt_xatol),
            "fatol": float(opt_fatol),
        },
    )

    return np.asarray(result.x, float).reshape(Phi.shape[1], 3)


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


def run_5fold_deltaE2000(
    P: np.ndarray,
    X: np.ndarray,
    XYZ_white: np.ndarray,
    ridge: float = 0.0,
    seed: int = 42,
    method: str = "ls",
    opt_loss: str = "lab",
    opt_maxiter: int = 3000,
    opt_xatol: float = 1e-6,
    opt_fatol: float = 1e-6,
) -> dict:
    fold_mean = []
    fold_max = []
    fold_median = []
    fold_p95 = []

    for tr, te in kfold_indices(P.shape[0], k=5, seed=seed):
        M = fit_ls_matrix(
            P[tr],
            X[tr],
            ridge=ridge,
            method=method,
            XYZ_white=XYZ_white,
            opt_loss=opt_loss,
            opt_maxiter=opt_maxiter,
            opt_xatol=opt_xatol,
            opt_fatol=opt_fatol,
        )
        Xp = predict_xyz(P[te], M, method=method)

        Lab_p = xyz_to_lab(Xp, XYZ_white)
        Lab_t = xyz_to_lab(X[te], XYZ_white)
        dE = deltaE2000(Lab_p, Lab_t)
        st = deltae_stats(dE)

        fold_mean.append(st["mean"])
        fold_max.append(st["max"])
        fold_median.append(st["median"])
        fold_p95.append(st["p95"])

    fold_mean = np.asarray(fold_mean, float)
    fold_max = np.asarray(fold_max, float)
    fold_median = np.asarray(fold_median, float)
    fold_p95 = np.asarray(fold_p95, float)

    return {
        "mean": float(np.mean(fold_mean)),
        "max": float(np.mean(fold_max)),
        "median": float(np.mean(fold_median)),
        "p95": float(np.mean(fold_p95)),
    }