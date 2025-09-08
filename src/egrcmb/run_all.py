import argparse
from .cli import run_cmb, build_argparser

def main():
    ap = argparse.ArgumentParser(description="Run full CMB TT pipeline (plots+AR1+MCMC).")
    ap.add_argument("--data", default="data/COM_PowerSpect_CMB-TT-full_R3.01.txt")
    ap.add_argument("--outdir", default="out_cmb")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--starts", type=int, default=16)
    args0 = ap.parse_args()

    ap2 = build_argparser()
    args = ap2.parse_args([])
    # Defaults for “everything on”
    args.data = args0.data
    args.outdir = args0.outdir
    args.seed = args0.seed
    args.starts = args0.starts
    args.plot = True
    args.ar1 = True
    args.ar1_calibrate = True
    args.band_scatter = True
    args.mcmc = True
    args.ppc_samples = 200
    run_cmb(args)

if __name__ == "__main__":
    main()
