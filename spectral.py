from __future__ import annotations

import gzip
import re
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd


@dataclass
class SpectralData:
    wl: np.ndarray          # (m,)
    values: np.ndarray      # (m, k)


def _trapz_weights(wl: np.ndarray) -> np.ndarray:
    wl = np.asarray(wl, dtype=float)
    if wl.ndim != 1 or wl.size < 2:
        raise ValueError("wl must be 1D with at least 2 samples.")
    dw = np.diff(wl)
    w = np.zeros_like(wl)
    w[0] = dw[0] / 2
    w[-1] = dw[-1] / 2
    w[1:-1] = (dw[:-1] + dw[1:]) / 2
    return w  # (m,)


def _interp_to(wl_target: np.ndarray, wl_src: np.ndarray, y_src: np.ndarray) -> np.ndarray:
    wl_target = np.asarray(wl_target, float)
    wl_src = np.asarray(wl_src, float)
    y_src = np.asarray(y_src, float)
    if y_src.ndim == 1:
        return np.interp(wl_target, wl_src, y_src)
    out = np.empty((wl_target.size, y_src.shape[1]), float)
    for j in range(y_src.shape[1]):
        out[:, j] = np.interp(wl_target, wl_src, y_src[:, j])
    return out


def align_to_common_wl(base_wl: np.ndarray, data: SpectralData) -> SpectralData:
    return SpectralData(
        wl=np.asarray(base_wl, float),
        values=_interp_to(base_wl, data.wl, data.values),
    )


def load_cie_d65_csv(path: str) -> SpectralData:
    
    """Load CIE D65 spectral power distribution from CSV."""

    df = pd.read_csv(path)
    cols = [c.lower() for c in df.columns]

    wl_candidates = [i for i, c in enumerate(cols) if c in ("wavelength", "wl", "lambda", "nm")]
    wl_col = df.columns[wl_candidates[0]] if wl_candidates else df.columns[0]

    other_cols = [c for c in df.columns if c != wl_col]
    if len(other_cols) < 1:
        raise ValueError(f"Could not find D65 column in {path}. Columns: {list(df.columns)}")
    d65_col = other_cols[0]

    wl = df[wl_col].to_numpy(float)
    d65 = df[d65_col].to_numpy(float).reshape(-1, 1)
    return SpectralData(wl=wl, values=d65)


def load_cie_cmf_csv(path: str) -> SpectralData:
    
    """Load CIE 1931 2° CMF from CSV (x̄,ȳ,z̄)."""

    df = pd.read_csv(path)
    cols = [c.lower() for c in df.columns]

    wl_candidates = [i for i, c in enumerate(cols) if c in ("wavelength", "wl", "lambda", "nm")]
    wl_col = df.columns[wl_candidates[0]] if wl_candidates else df.columns[0]

    def find_col(keys: List[str]) -> Optional[str]:
        for c in df.columns:
            lc = str(c).lower()
            if any(k in lc for k in keys):
                return c
        return None

    xcol = find_col(["xbar", "x_bar", "x̄"]) or find_col(["x"])
    ycol = find_col(["ybar", "y_bar", "ȳ"]) or find_col(["y"])
    zcol = find_col(["zbar", "z_bar", "z̄"]) or find_col(["z"])

    if xcol is None or ycol is None or zcol is None:
        non_wl = [c for c in df.columns if c != wl_col]
        if len(non_wl) < 3:
            raise ValueError(f"Could not find CMF columns in {path}. Columns: {list(df.columns)}")
        xcol, ycol, zcol = non_wl[0], non_wl[1], non_wl[2]

    wl = df[wl_col].to_numpy(float)
    xyz = df[[xcol, ycol, zcol]].to_numpy(float)
    return SpectralData(wl=wl, values=xyz)


def load_nikon_xlsx(path: str) -> SpectralData:
    
    """Load camera spectral sensitivities from nikon.xlsx."""

    df = pd.read_excel(path, sheet_name=0)  

    if df.shape[1] < 4:
        raise ValueError(f"nikon.xlsx has too few columns: {list(df.columns)}")

    wl_col = df.columns[0]

    r_col = None
    for c in df.columns:
        if "ground truth" in str(c).strip().lower():
            r_col = c
            break
    if r_col is None:
        raise ValueError(f"Could not find 'Ground truth' column in nikon.xlsx headers: {list(df.columns)}")

    first_row = df.iloc[0].astype(str).str.strip().str.lower()

    g_col = None
    b_col = None
    for c in df.columns:
        v = str(first_row[c]).strip().lower()
        if v == "green":
            g_col = c
        if v == "blue":
            b_col = c

    if g_col is None:
        for c in df.columns:
            if "green" in str(c).strip().lower():
                g_col = c
                break
    if b_col is None:
        for c in df.columns:
            if "blue" in str(c).strip().lower():
                b_col = c
                break

    if g_col is None or b_col is None:
        cols = list(df.columns)
        ri = cols.index(r_col)
        if g_col is None and ri + 1 < len(cols):
            g_col = cols[ri + 1]
        if b_col is None and ri + 2 < len(cols):
            b_col = cols[ri + 2]

    if g_col is None or b_col is None:
        raise ValueError(f"Could not identify Green/Blue columns. Headers: {list(df.columns)}")

    sub = df[[wl_col, r_col, g_col, b_col]].copy()

    for c in [wl_col, r_col, g_col, b_col]:
        sub[c] = pd.to_numeric(sub[c], errors="coerce")

    sub = sub.dropna(subset=[wl_col, r_col, g_col, b_col])

    wl = sub[wl_col].to_numpy(dtype=float)
    rgb = sub[[r_col, g_col, b_col]].to_numpy(dtype=float)

    return SpectralData(wl=wl, values=rgb)


def _read_gz_text_lines(path_gz: str) -> List[str]:
    with gzip.open(path_gz, "rt", encoding="utf-8", errors="replace") as f:
        return f.read().splitlines()


def load_sfu_reflect_db_reflect_gz(path_gz: str) -> SpectralData:
    
    """Load SFU reflectance database (.reflect.gz) into (wl, spectra)."""

    raw_lines = _read_gz_text_lines(path_gz)

    header = None
    for ln in raw_lines[:400]:
        if ln.strip().startswith("#!"):
            header = ln.strip()
            break
    if header is None:
        raise ValueError("SFU header line starting with '#!' not found in reflect file.")

    m_n = re.search(r"n\s*=\s*([0-9]+)", header)
    m_o = re.search(r"o\s*=\s*([0-9]*\.?[0-9]+)", header)
    m_s = re.search(r"s\s*=\s*([0-9]*\.?[0-9]+)", header)
    if not (m_n and m_o and m_s):
        raise ValueError(f"Could not parse n/o/s from header: {header}")

    n_wl = int(m_n.group(1))
    o = float(m_o.group(1))
    s = float(m_s.group(1))

    wl = o + s * np.arange(n_wl, dtype=float)  # (n_wl,)

    nums: list[float] = []
    header_found = False
    for ln in raw_lines:
        if not header_found:
            if ln.strip().startswith("#!"):
                header_found = True
            continue

        ln2 = re.split(r"(#|%|//)", ln)[0].strip()
        if not ln2:
            continue

        parts = re.split(r"[,\s]+", ln2)
        for p in parts:
            if not p:
                continue
            try:
                nums.append(float(p))
            except ValueError:
                pass

    if len(nums) < n_wl:
        raise ValueError("Not enough numeric data found after header.")

    n_spectra = len(nums) // n_wl
    if n_spectra <= 0:
        raise ValueError("Could not determine number of spectra (n_spectra <= 0).")

    total = n_spectra * n_wl
    arr = np.asarray(nums[:total], dtype=float).reshape(n_spectra, n_wl)  # (n_spectra, n_wl)

    return SpectralData(wl=wl, values=arr.T)


def compute_rgb_xyz(
    reflectances: SpectralData,     # (m, n)
    camera_rgb_sens: SpectralData,  # (m, 3)
    cmf_xyz: SpectralData,          # (m, 3)
    illuminant: SpectralData,       # (m, 1)
    exposure: float = 1.0,
    normalize_by_white: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    
    """Integrate spectra to camera RGB and CIE XYZ (optionally white-normalized)."""

    wl = reflectances.wl
    cam = align_to_common_wl(wl, camera_rgb_sens)
    xyz = align_to_common_wl(wl, cmf_xyz)
    ill = align_to_common_wl(wl, illuminant)

    S = np.asarray(reflectances.values, float)         # (m, n)
    Q = np.asarray(cam.values, float)                  # (m, 3)
    Xbar = np.asarray(xyz.values, float)               # (m, 3)
    E = np.asarray(ill.values[:, 0:1], float)          # (m, 1)

    w = _trapz_weights(wl).reshape(-1, 1)              # (m, 1)

    ES = E * S                                         # (m, n)

    RGB = (ES.T @ (w * Q)) * exposure                  # (n, 3)
    XYZ = (ES.T @ (w * Xbar)) * exposure               # (n, 3)

    if normalize_by_white:
        ones = np.ones((wl.size, 1), dtype=float)      # perfect reflector
        E1 = E * ones                                  # (m,1)
        RGB_w = ((E1.T @ (w * Q)) * exposure).reshape(3,)   # (3,)
        XYZ_w = ((E1.T @ (w * Xbar)) * exposure).reshape(3,) # (3,)

        RGB_w = np.where(RGB_w == 0, 1.0, RGB_w)
        XYZ_w = np.where(XYZ_w == 0, 1.0, XYZ_w)

        RGB = RGB / RGB_w
        XYZ = XYZ / XYZ_w

    return RGB, XYZ


def compute_white_xyz(wl: np.ndarray, cmf_xyz: SpectralData, illuminant: SpectralData) -> np.ndarray:
    
    """XYZ white point for a perfect reflector under the given illuminant."""

    xyz = align_to_common_wl(wl, cmf_xyz).values   # (m,3)
    ill = align_to_common_wl(wl, illuminant).values[:, 0:1]  # (m,1)
    w = _trapz_weights(wl).reshape(-1, 1)          # (m,1)
    XYZ_w = (ill.T @ (w * xyz)).reshape(3,)
    # avoid zeros
    XYZ_w = np.where(XYZ_w == 0, 1.0, XYZ_w)
    return XYZ_w