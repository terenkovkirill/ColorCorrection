import numpy as np


def _f_lab(t: np.ndarray) -> np.ndarray:
    delta = 6 / 29
    return np.where(t > delta**3, np.cbrt(t), (t / (3 * delta**2)) + (4 / 29))


def xyz_to_lab(XYZ: np.ndarray, XYZ_white: np.ndarray) -> np.ndarray:

    """XYZ -> CIE Lab (D65 or other reference white)."""

    XYZ = np.asarray(XYZ, float)
    W = np.asarray(XYZ_white, float).reshape(1, 3)

    W = np.where(W == 0, 1.0, W)

    x = XYZ[:, 0:1] / W[:, 0:1]
    y = XYZ[:, 1:2] / W[:, 1:2]
    z = XYZ[:, 2:3] / W[:, 2:3]

    fx, fy, fz = _f_lab(x), _f_lab(y), _f_lab(z)

    L = (116 * fy) - 16
    a = 500 * (fx - fy)
    b = 200 * (fy - fz)
    return np.concatenate([L, a, b], axis=1)


def deltaE2000(Lab1: np.ndarray, Lab2: np.ndarray,
              kL: float = 1.0, kC: float = 1.0, kH: float = 1.0) -> np.ndarray:
    
    """CIEDE2000 between two Lab arrays, returns (n,)."""

    Lab1 = np.asarray(Lab1, float)
    Lab2 = np.asarray(Lab2, float)

    L1, a1, b1 = Lab1[:, 0], Lab1[:, 1], Lab1[:, 2]
    L2, a2, b2 = Lab2[:, 0], Lab2[:, 1], Lab2[:, 2]

    C1 = np.sqrt(a1**2 + b1**2)
    C2 = np.sqrt(a2**2 + b2**2)
    Cbar = (C1 + C2) / 2

    Cbar7 = Cbar**7
    G = 0.5 * (1 - np.sqrt(Cbar7 / (Cbar7 + 25**7)))

    a1p = (1 + G) * a1
    a2p = (1 + G) * a2

    C1p = np.sqrt(a1p**2 + b1**2)
    C2p = np.sqrt(a2p**2 + b2**2)

    h1p = np.degrees(np.arctan2(b1, a1p)) % 360
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360

    dLp = L2 - L1
    dCp = C2p - C1p

    dhp = h2p - h1p
    dhp = np.where(dhp > 180, dhp - 360, dhp)
    dhp = np.where(dhp < -180, dhp + 360, dhp)
    dhp = np.where((C1p * C2p) == 0, 0, dhp)

    dHp = 2 * np.sqrt(C1p * C2p) * np.sin(np.radians(dhp / 2))

    Lbarp = (L1 + L2) / 2
    Cbarp = (C1p + C2p) / 2

    hsum = h1p + h2p
    hbarp = np.where((C1p * C2p) == 0, hsum, 0)

    cond = np.abs(h1p - h2p) > 180
    hbarp = np.where((C1p * C2p) == 0, hbarp, (hsum / 2))
    hbarp = np.where((C1p * C2p) == 0, hbarp,
                     np.where(cond, (hsum + 360) / 2, hsum / 2))
    hbarp = hbarp % 360

    T = (1
         - 0.17 * np.cos(np.radians(hbarp - 30))
         + 0.24 * np.cos(np.radians(2 * hbarp))
         + 0.32 * np.cos(np.radians(3 * hbarp + 6))
         - 0.20 * np.cos(np.radians(4 * hbarp - 63)))

    dtheta = 30 * np.exp(-((hbarp - 275) / 25)**2)
    RC = 2 * np.sqrt((Cbarp**7) / (Cbarp**7 + 25**7))
    RT = -np.sin(np.radians(2 * dtheta)) * RC

    SL = 1 + (0.015 * (Lbarp - 50)**2) / np.sqrt(20 + (Lbarp - 50)**2)
    SC = 1 + 0.045 * Cbarp
    SH = 1 + 0.015 * Cbarp * T

    dE = np.sqrt(
        (dLp / (kL * SL))**2 +
        (dCp / (kC * SC))**2 +
        (dHp / (kH * SH))**2 +
        RT * (dCp / (kC * SC)) * (dHp / (kH * SH))
    )
    return dE


def deltae_stats(dE: np.ndarray) -> dict:
    dE = np.asarray(dE, float)
    return {
        "mean": float(np.mean(dE)),
        "median": float(np.median(dE)),
        "p95": float(np.percentile(dE, 95)),
        "max": float(np.max(dE)),
        "min": float(np.min(dE)),
    }