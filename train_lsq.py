from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np
from scipy.optimize import minimize

from metrics import xyz_to_lab, deltaE2000, deltae_stats


VALID_METHODS = (
    "ls",
    "ls-p",
    "ls-rp",
    "ls-rp3",
    "ls-rp4",
    "ls-opt",
    "ls-p-opt",
    "ls-rp-opt",
    "ls-rp3-opt",
    "ls-rp4-opt",
)
VALID_OPT_LOSSES = ("lab", "de2000")
DEFAULT_RIDGE_GRID = (0.0, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4)


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


def _safe_sqrt_prod(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.sqrt(np.clip(a * b, 0.0, None))


def _safe_fourthroot_prod(*arrs: np.ndarray) -> np.ndarray:
    prod = np.ones_like(arrs[0], dtype=float)
    for arr in arrs:
        prod = prod * arr
    return np.power(np.clip(prod, 0.0, None), 0.25)


def _safe_cuberoot_prod(*arrs: np.ndarray) -> np.ndarray:
    prod = np.ones_like(arrs[0], dtype=float)
    for arr in arrs:
        prod = prod * arr
    return np.cbrt(np.clip(prod, 0.0, None))


def expand_features(P: np.ndarray, method: str = "ls") -> np.ndarray:
    """
    Feature maps:
      ls       -> [R, G, B]
      ls-p     -> [R^2, G^2, B^2, RG, RB, GB, R, G, B, 1]
      ls-rp    -> [R, G, B, sqrt(RG), sqrt(RB), sqrt(GB)]
      ls-rp3   -> ls-rp + third-order root-polynomial terms
                  [(R^2G)^(1/3), (R^2B)^(1/3), (G^2R)^(1/3), (G^2B)^(1/3),
                   (B^2R)^(1/3), (B^2G)^(1/3), (RGB)^(1/3)]
      ls-rp4   -> ls-rp3 + fourth-order root-polynomial terms
                  [(R^3G)^(1/4), (R^3B)^(1/4), (G^3R)^(1/4), (G^3B)^(1/4),
                   (B^3R)^(1/4), (B^3G)^(1/4),
                   (R^2G^2)^(1/4), (R^2B^2)^(1/4), (G^2B^2)^(1/4),
                   (R^2GB)^(1/4), (RG^2B)^(1/4), (RGB^2)^(1/4)]
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
            _safe_sqrt_prod(R, G),
            _safe_sqrt_prod(R, B),
            _safe_sqrt_prod(G, B),
        ])

    if base == "ls-rp3":
        return np.column_stack([
            R,
            G,
            B,
            _safe_sqrt_prod(R, G),
            _safe_sqrt_prod(R, B),
            _safe_sqrt_prod(G, B),
            _safe_cuberoot_prod(R, R, G),
            _safe_cuberoot_prod(R, R, B),
            _safe_cuberoot_prod(G, G, R),
            _safe_cuberoot_prod(G, G, B),
            _safe_cuberoot_prod(B, B, R),
            _safe_cuberoot_prod(B, B, G),
            _safe_cuberoot_prod(R, G, B),
        ])

    if base == "ls-rp4":
        return np.column_stack([
            # order 1
            R,
            G,
            B,

            # order 2 root-polynomial
            _safe_sqrt_prod(R, G),
            _safe_sqrt_prod(R, B),
            _safe_sqrt_prod(G, B),

            # order 3 root-polynomial
            _safe_cuberoot_prod(R, R, G),
            _safe_cuberoot_prod(R, R, B),
            _safe_cuberoot_prod(G, G, R),
            _safe_cuberoot_prod(G, G, B),
            _safe_cuberoot_prod(B, B, R),
            _safe_cuberoot_prod(B, B, G),
            _safe_cuberoot_prod(R, G, B),

            # order 4 root-polynomial
            _safe_fourthroot_prod(R, R, R, G),
            _safe_fourthroot_prod(R, R, R, B),
            _safe_fourthroot_prod(G, G, G, R),
            _safe_fourthroot_prod(G, G, G, B),
            _safe_fourthroot_prod(B, B, B, R),
            _safe_fourthroot_prod(B, B, B, G),

            _safe_fourthroot_prod(R, R, G, G),
            _safe_fourthroot_prod(R, R, B, B),
            _safe_fourthroot_prod(G, G, B, B),

            _safe_fourthroot_prod(R, R, G, B),
            _safe_fourthroot_prod(R, G, G, B),
            _safe_fourthroot_prod(R, G, B, B),
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
    ridge: float = 0.0,
) -> float:
    d = Phi.shape[1]
    M = np.asarray(params, float).reshape(d, 3)
    Xp = Phi @ M

    Lab_p = xyz_to_lab(Xp, XYZ_white)
    Lab_t = xyz_to_lab(X, XYZ_white)

    if opt_loss == "lab":
        loss = float(np.mean(_lab_triplet_norm(Lab_p, Lab_t)))
    elif opt_loss == "de2000":
        loss = float(np.mean(deltaE2000(Lab_p, Lab_t)))
    else:
        raise ValueError(f"Unknown opt_loss '{opt_loss}'. Valid values: {VALID_OPT_LOSSES}")

    if ridge > 0:
        loss += float(ridge) * float(np.mean(M * M))

    return loss


def _evaluate_deltae_mean(
    P_train: np.ndarray,
    X_train: np.ndarray,
    P_val: np.ndarray,
    X_val: np.ndarray,
    XYZ_white: np.ndarray,
    *,
    ridge: float,
    method: str,
    opt_loss: str,
    opt_maxiter: int,
    opt_xatol: float,
    opt_fatol: float,
    opt_multistart: int,
    opt_perturb_scale: float,
    seed: int,
) -> float:
    rng = np.random.default_rng(seed)
    M = fit_ls_matrix(
        P_train,
        X_train,
        ridge=ridge,
        method=method,
        XYZ_white=XYZ_white,
        opt_loss=opt_loss,
        opt_maxiter=opt_maxiter,
        opt_xatol=opt_xatol,
        opt_fatol=opt_fatol,
        opt_multistart=opt_multistart,
        opt_perturb_scale=opt_perturb_scale,
        rng=rng,
    )
    Xp = predict_xyz(P_val, M, method=method)
    Lab_p = xyz_to_lab(Xp, XYZ_white)
    Lab_t = xyz_to_lab(X_val, XYZ_white)
    return float(np.mean(deltaE2000(Lab_p, Lab_t)))


def select_ridge_nested_cv(
    P: np.ndarray,
    X: np.ndarray,
    XYZ_white: np.ndarray,
    *,
    method: str,
    ridge_grid: Sequence[float] = DEFAULT_RIDGE_GRID,
    inner_folds: int = 3,
    seed: int = 42,
    opt_loss: str = "lab",
    opt_maxiter: int = 3000,
    opt_xatol: float = 1e-6,
    opt_fatol: float = 1e-6,
    opt_multistart: int = 1,
    opt_perturb_scale: float = 0.05,
) -> float:
    if inner_folds < 2:
        raise ValueError("inner_folds must be >= 2")

    candidates = [float(r) for r in ridge_grid]
    if not candidates:
        raise ValueError("ridge_grid must contain at least one value")

    splits = kfold_indices(P.shape[0], k=inner_folds, seed=seed)
    best_ridge = candidates[0]
    best_score = np.inf

    for i, ridge in enumerate(candidates):
        fold_scores = []
        for j, (tr, va) in enumerate(splits):
            fold_scores.append(
                _evaluate_deltae_mean(
                    P[tr],
                    X[tr],
                    P[va],
                    X[va],
                    XYZ_white,
                    ridge=ridge,
                    method=method,
                    opt_loss=opt_loss,
                    opt_maxiter=opt_maxiter,
                    opt_xatol=opt_xatol,
                    opt_fatol=opt_fatol,
                    opt_multistart=opt_multistart,
                    opt_perturb_scale=opt_perturb_scale,
                    seed=seed + 1000 * (i + 1) + j,
                )
            )

        score = float(np.mean(fold_scores))
        if (score < best_score - 1e-15) or (abs(score - best_score) <= 1e-15 and ridge < best_ridge):
            best_score = score
            best_ridge = ridge

    return float(best_ridge)


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
    opt_multistart: int = 1,
    opt_perturb_scale: float = 0.05,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    X = np.asarray(X, float)
    if X.ndim != 2 or X.shape[1] != 3:
        raise ValueError("X must be (n, 3)")

    opt_loss = opt_loss.lower()
    if opt_loss not in VALID_OPT_LOSSES:
        raise ValueError(f"Unknown opt_loss '{opt_loss}'. Valid values: {VALID_OPT_LOSSES}")
    if opt_multistart < 1:
        raise ValueError("opt_multistart must be >= 1")
    if opt_perturb_scale < 0:
        raise ValueError("opt_perturb_scale must be >= 0")

    Phi = expand_features(P, method=method)
    if Phi.shape[0] != X.shape[0]:
        raise ValueError("P and X must have the same number of rows")

    M0 = _fit_closed_form_from_features(Phi, X, ridge=ridge)

    if not _is_opt_method(method):
        return M0

    if XYZ_white is None:
        raise ValueError("XYZ_white is required for optimisation-based methods")

    if rng is None:
        rng = np.random.default_rng(42)

    x0 = np.asarray(M0, float).reshape(-1)
    scale = np.maximum(np.abs(x0), 1.0)
    starts = [x0]
    for _ in range(opt_multistart - 1):
        noise = rng.normal(loc=0.0, scale=opt_perturb_scale, size=x0.shape) * scale
        starts.append(x0 + noise)

    best_x = x0.copy()
    best_fun = _objective(best_x, Phi, X, np.asarray(XYZ_white, float), opt_loss, ridge=ridge)

    for start in starts:
        result = minimize(
            _objective,
            x0=start,
            args=(Phi, X, np.asarray(XYZ_white, float), opt_loss, ridge),
            method="Nelder-Mead",
            options={
                "maxiter": int(opt_maxiter),
                "xatol": float(opt_xatol),
                "fatol": float(opt_fatol),
            },
        )

        cand_x = np.asarray(result.x if result.x is not None else start, float)
        cand_fun = _objective(cand_x, Phi, X, np.asarray(XYZ_white, float), opt_loss, ridge=ridge)

        if cand_fun < best_fun:
            best_fun = cand_fun
            best_x = cand_x

    return best_x.reshape(Phi.shape[1], 3)


def predict_xyz(P: np.ndarray, M: np.ndarray, method: str = "ls") -> np.ndarray:
    Phi = expand_features(P, method=method)
    M = np.asarray(M, float)

    if M.ndim != 2 or M.shape[0] != Phi.shape[1] or M.shape[1] != 3:
        raise ValueError(f"M must have shape ({Phi.shape[1]}, 3) for method '{method}'")

    return Phi @ M


def kfold_indices(n: int, k: int = 5, seed: int = 42) -> List[Tuple[np.ndarray, np.ndarray]]:
    if k < 2:
        raise ValueError("k must be >= 2")
    if n < k:
        raise ValueError("n must be >= k")

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
    opt_multistart: int = 1,
    opt_perturb_scale: float = 0.05,
    auto_ridge: bool = False,
    ridge_grid: Sequence[float] = DEFAULT_RIDGE_GRID,
    ridge_inner_folds: int = 3,
) -> dict:
    fold_mean = []
    fold_max = []
    fold_median = []
    fold_p95 = []
    selected_ridges: list[float] = []

    for fold_id, (tr, te) in enumerate(kfold_indices(P.shape[0], k=5, seed=seed)):
        ridge_to_use = float(ridge)
        if auto_ridge:
            ridge_to_use = select_ridge_nested_cv(
                P[tr],
                X[tr],
                XYZ_white,
                method=method,
                ridge_grid=ridge_grid,
                inner_folds=ridge_inner_folds,
                seed=seed + 10_000 + fold_id,
                opt_loss=opt_loss,
                opt_maxiter=opt_maxiter,
                opt_xatol=opt_xatol,
                opt_fatol=opt_fatol,
                opt_multistart=opt_multistart,
                opt_perturb_scale=opt_perturb_scale,
            )
        selected_ridges.append(ridge_to_use)

        rng = np.random.default_rng(seed + 20_000 + fold_id)
        M = fit_ls_matrix(
            P[tr],
            X[tr],
            ridge=ridge_to_use,
            method=method,
            XYZ_white=XYZ_white,
            opt_loss=opt_loss,
            opt_maxiter=opt_maxiter,
            opt_xatol=opt_xatol,
            opt_fatol=opt_fatol,
            opt_multistart=opt_multistart,
            opt_perturb_scale=opt_perturb_scale,
            rng=rng,
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
        "selected_ridges": [float(r) for r in selected_ridges],
    }