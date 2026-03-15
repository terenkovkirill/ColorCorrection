import argparse
import numpy as np

from spectral import (
    load_sfu_reflect_db_reflect_gz,
    load_nikon_xlsx,
    load_cie_d65_csv,
    load_cie_cmf_csv,
    compute_rgb_xyz,
    compute_white_xyz,
)

from train_lsq import (
    fit_ls_matrix,
    predict_xyz,
    run_5fold,
    run_5fold_deltaE2000,
    mean_abs_error,
    feature_dim,
)

from metrics import (
    xyz_to_lab,
    deltaE2000,
    deltae_stats,
)


METHOD_CHOICES = ("ls", "ls-p", "ls-rp")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reflect_gz", required=True, help="Path to reflect_db.reflect.gz")
    ap.add_argument("--nikon_xlsx", required=True, help="Path to nikon.xlsx (D5100 sensitivities)")
    ap.add_argument("--cie_d65_csv", required=True, help="Path to CIE_std_illum_D65.csv")
    ap.add_argument("--cie_cmf_csv", required=True, help="Path to CIE_xyz_1931_2deg.csv")
    ap.add_argument("--method", choices=METHOD_CHOICES, default="ls",
                    help="Colour correction method: ls, ls-p, or ls-rp")
    ap.add_argument("--exposure", type=float, default=1.0, help="Exposure multiplier (default 1.0)")
    ap.add_argument("--ridge", type=float, default=0.0, help="Optional ridge regularization (default 0)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for 5-fold split")
    args = ap.parse_args()

    print("Loading reflectances:", args.reflect_gz)
    refl = load_sfu_reflect_db_reflect_gz(args.reflect_gz)
    print(f"  reflectances: wl={refl.wl.size}, spectra={refl.values.shape[1]}")

    print("Loading Nikon sensitivities:", args.nikon_xlsx)
    nikon = load_nikon_xlsx(args.nikon_xlsx)
    print(f"  nikon: wl={nikon.wl.size}, cols={nikon.values.shape[1]}")

    print("Loading CIE D65:", args.cie_d65_csv)
    d65 = load_cie_d65_csv(args.cie_d65_csv)
    print(f"  d65: wl={d65.wl.size}")

    print("Loading CIE CMF 1931 2°:", args.cie_cmf_csv)
    cmf = load_cie_cmf_csv(args.cie_cmf_csv)
    print(f"  cmf: wl={cmf.wl.size}, cols={cmf.values.shape[1]}")

    print("Computing RGB/XYZ by spectral integration...")
    P, X = compute_rgb_xyz(
        reflectances=refl,
        camera_rgb_sens=nikon,
        cmf_xyz=cmf,
        illuminant=d65,
        exposure=args.exposure,
    )
    print(f"  P (RGB): {P.shape}, X (XYZ): {X.shape}")
    print(f"  method: {args.method}, feature_dim: {feature_dim(args.method)}")

    XYZ_white = compute_white_xyz(refl.wl, cmf, d65)

    de_stats = run_5fold_deltaE2000(
        P, X, XYZ_white,
        ridge=args.ridge,
        seed=args.seed,
        method=args.method,
    )
    print(f"\n5-fold CV (DeltaE2000, method={args.method}):")
    print("  fold_mean:", np.array2string(de_stats["fold_mean"], precision=6))
    print("  fold_p95 :", np.array2string(de_stats["fold_p95"], precision=6))
    print("  mean(mean):", de_stats["mean_of_means"])
    print("  median(mean):", de_stats["median_of_means"])
    print("  mean(p95):", de_stats["mean_of_p95"])
    print("  max(p95):", de_stats["max_p95"])

    stats = run_5fold(P, X, ridge=args.ridge, seed=args.seed, method=args.method)
    print(f"\n5-fold CV (XYZ MAE, method={args.method}):")
    print("  fold_mae:", np.array2string(stats["fold_mae"], precision=6))
    print("  mean   :", stats["mean"])
    print("  median :", stats["median"])
    print("  min    :", stats["min"])
    print("  max    :", stats["max"])

    M = fit_ls_matrix(P, X, ridge=args.ridge, method=args.method)
    print(f"\nFinal regression matrix M ({feature_dim(args.method)}x3) for method={args.method}:")
    print(M)

    Xp_all = predict_xyz(P, M, method=args.method)
    print("\nTrain-set XYZ MAE:", mean_abs_error(Xp_all, X))

    Lab_p_all = xyz_to_lab(Xp_all, XYZ_white)
    Lab_t_all = xyz_to_lab(X, XYZ_white)
    dE_all = deltaE2000(Lab_p_all, Lab_t_all)
    st_all = deltae_stats(dE_all)

    print("\nTrain-set DeltaE2000 stats:")
    print("  mean  :", st_all["mean"])
    print("  median:", st_all["median"])
    print("  p95   :", st_all["p95"])
    print("  max   :", st_all["max"])


if __name__ == "__main__":
    main()