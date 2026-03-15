import argparse
import numpy as np

from spectral import (
    load_sfu_reflect_db_reflect_gz,
    load_nikon_xlsx,
    load_cie_d65_csv,
    load_cie_cmf_csv,
    compute_rgb_xyz,
)

from train_lsq import VALID_METHODS, run_5fold_deltaE2000


DISPLAY_NAMES = {
    "ls": "LS",
    "ls-p": "LS-P",
    "ls-rp": "LS-RP",
    "ls-opt": "LS-Opt",
    "ls-p-opt": "LS-P-Opt",
    "ls-rp-opt": "LS-RP-Opt",
}


def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--reflect_gz", required=True)
    ap.add_argument("--nikon_xlsx", required=True)
    ap.add_argument("--cie_d65_csv", required=True)
    ap.add_argument("--cie_cmf_csv", required=True)

    ap.add_argument("--method", choices=VALID_METHODS, default="ls")
    ap.add_argument("--all_methods", action="store_true")

    ap.add_argument("--opt_loss", choices=("lab", "de2000"), default="lab")
    ap.add_argument("--opt_maxiter", type=int, default=3000)
    ap.add_argument("--opt_xatol", type=float, default=1e-6)
    ap.add_argument("--opt_fatol", type=float, default=1e-6)

    ap.add_argument("--exposure", type=float, default=1.0)
    ap.add_argument("--ridge", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--decimals", type=int, default=2)

    args = ap.parse_args()

    refl = load_sfu_reflect_db_reflect_gz(args.reflect_gz)
    nikon = load_nikon_xlsx(args.nikon_xlsx)
    d65 = load_cie_d65_csv(args.cie_d65_csv)
    cmf = load_cie_cmf_csv(args.cie_cmf_csv)

    P, X = compute_rgb_xyz(
        reflectances=refl,
        camera_rgb_sens=nikon,
        cmf_xyz=cmf,
        illuminant=d65,
        exposure=args.exposure,
    )

    XYZ_white = np.ones(3, dtype=float)

    methods = list(VALID_METHODS) if args.all_methods else [args.method]

    width_method = 12
    width_num = 10

    print(f"{'Methods':<{width_method}}{'Mean':>{width_num}}{'Max':>{width_num}}{'Med':>{width_num}}{'95%':>{width_num}}")

    fmt = f">{width_num}.{args.decimals}f"

    for method in methods:
        st = run_5fold_deltaE2000(
            P,
            X,
            XYZ_white,
            ridge=args.ridge,
            seed=args.seed,
            method=method,
            opt_loss=args.opt_loss,
            opt_maxiter=args.opt_maxiter,
            opt_xatol=args.opt_xatol,
            opt_fatol=args.opt_fatol,
        )

        print(
            f"{DISPLAY_NAMES[method]:<{width_method}}"
            f"{format(st['mean'], fmt)}"
            f"{format(st['max'], fmt)}"
            f"{format(st['median'], fmt)}"
            f"{format(st['p95'], fmt)}"
        )


if __name__ == "__main__":
    main()