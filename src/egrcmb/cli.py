from __future__ import annotations
import argparse, json, math, sys, warnings
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
plt.style.use("dark_background")

from .constants import *
from .elm import EntropyLedgerMatrix
from .model import EGRFreed7TT, build_param_dict_and_bounds, build_model_function
from .fit import run_fit, ledger_info_overhead
from .ar1 import (calibrate_epsilon_ar1, residual_acf, whitened_residual_stats,
                  ledger_band_overhead, fit_ar1_rho)
from .mcmc_utils import mcmc_run, posterior_predictive_band
from .diagnostics import (gof_and_normality, aic_bic, save_fit_table,
                          qqplot_normal, corr_heatmap_from_chain)

# IO helpers kept local (simple and explicit)
import hashlib, csv
from datetime import datetime, timezone

SUMMARY_COLS = [
    "ID","n_points_used","ell_min","ell_max","Dl_max","model_name",
    "epsilon_model_pinned","chi2_red","p_chi2","AIC","BIC","Qexp",
    "R_gate","phi_ac","A_ac","A_pk","k_pk","sigma_pk","S_resp",
    "rho_AR1","acf_lags_over_2_over_sqrtN",
    "deltaH_mix_total_nats","deltaH_mix_per_mode_nats"
]

def ensure_dirs(outdir: Path):
    figs = outdir / "figs"; tables = outdir / "tables"
    figs.mkdir(parents=True, exist_ok=True); tables.mkdir(parents=True, exist_ok=True)
    return outdir, figs, tables

def file_sha256(path: Path) -> str:
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1<<16), b""): h.update(chunk)
        return h.hexdigest()
    except Exception: return ""

def write_run_json(args: argparse.Namespace, outdir: Path, data_path: Path) -> None:
    info = {"timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "args": vars(args), "data_sha256": file_sha256(data_path),
            "versions": {"python": sys.version, "numpy": np.__version__,
                         "matplotlib": plt.matplotlib.__version__}}
    (outdir/"run.json").write_text(json.dumps(info, indent=2))

def summarize_and_save(row: dict, summary_path: Path) -> None:
    write_header = not summary_path.exists()
    if not write_header:
        try:
            with open(summary_path, "r", newline="") as f:
                first = f.readline()
            write_header = not all(h in first for h in ("epsilon_model_pinned","R_gate","S_resp","p_chi2","AIC","BIC"))
        except Exception:
            write_header = True
    mode = "w" if write_header else "a"
    with open(summary_path, mode, newline="") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_COLS)
        if write_header: w.writeheader()
        w.writerow({k: row.get(k, "") for k in SUMMARY_COLS})

# ---------------- Pipeline runner ----------------
def run_cmb(args: argparse.Namespace) -> None:
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    outdir, figs_dir, tables_dir = ensure_dirs(outdir)

    # ---------- data ----------
    data_path = Path(args.data)
    try:
        raw = np.loadtxt(data_path)
    except Exception as e:
        print(f"Error loading data file '{args.data}': {e}"); sys.exit(1)

    ell         = raw[:, 0]
    Dl_data     = raw[:, 1]
    Dl_err_min  = raw[:, 2]
    Dl_err_plus = raw[:, 3]
    Dl_err_planck = 0.5 * (Dl_err_min + Dl_err_plus)

    if args.ell_min is not None:
        m = ell >= args.ell_min
        ell, Dl_data, Dl_err_planck = ell[m], Dl_data[m], Dl_err_planck[m]
    if args.ell_max is not None:
        m = ell <= args.ell_max
        ell, Dl_data, Dl_err_planck = ell[m], Dl_data[m], Dl_err_planck[m]

    # ---------- optional MG/Wilson kernel ----------
    mg_cfg, mg_accept, kernel = None, None, None
    if args.use_mg and args.mg_config:
        try:
            from .mg_kernel import IRKernel
            mg_cfg_path = Path(args.mg_config)
            if mg_cfg_path.exists():
                mg_cfg = json.loads(mg_cfg_path.read_text())
                kernel = IRKernel(**mg_cfg)
                mg_accept = {
                    "gr_symbol": bool(kernel.gr_principal_symbol_ok()),
                    "shapiro_pos": bool(kernel.eikonal_shapiro_delay_positive()),
                }
                print(f"[mg] kernel loaded from {mg_cfg_path}  acceptance={mg_accept}")
            else:
                print(f"[mg] warning: --mg-config not found: {mg_cfg_path}")
        except Exception as e:
            print(f"[mg] warning: failed to init kernel: {e}")

    # ---------- run.json (provenance) ----------
    write_run_json(args, outdir, data_path)
    try:
        meta = json.loads((outdir / "run.json").read_text())
        meta["mg_kernel"] = {"enabled": bool(kernel is not None), "config": mg_cfg, "acceptance": mg_accept}
        (outdir / "run.json").write_text(json.dumps(meta, indent=2))
    except Exception:
        pass

    # ---------- optional ELM diagnostics ----------
    if args.elm_enable:
        elm_dir = outdir / "elm"; elm_dir.mkdir(parents=True, exist_ok=True)
        elm = EntropyLedgerMatrix(mode=args.elm_mode, alpha=args.elm_alpha, beta=args.elm_beta)
        bank = elm.bank()
        def _save_matrix(name, M):
            np.save(elm_dir / f"{name}.npy", M)
            try:
                plt.figure(figsize=(3,3)); plt.imshow(M.real, cmap="gray", interpolation="nearest")
                plt.axis("off"); plt.tight_layout(); plt.savefig(elm_dir / f"{name}_real.png", dpi=200); plt.close()
                plt.figure(figsize=(3,3)); plt.imshow(M.imag, cmap="gray", interpolation="nearest")
                plt.axis("off"); plt.tight_layout(); plt.savefig(elm_dir / f"{name}_imag.png", dpi=200); plt.close()
            except Exception:
                pass
        for nm, M in bank.items(): _save_matrix(nm, M)
        gates = [g.strip().upper() for g in args.elm_gates.split(",") if g.strip()]
        if gates:
            for base_name, M0 in bank.items():
                M = M0.copy()
                for g in gates:
                    if g == "AND": M = elm.transform_and(M, phase_scale=args.elm_phase_scale)
                    elif g == "XOR": M = elm.transform_xor(M, scale=args.elm_scale)
                    elif g == "NOT": M = elm.transform_not(M, scale=args.elm_scale)
                tag = "_".join(gates).lower(); _save_matrix(f"{base_name}__{tag}", M)

    # ---------- build model (kernel actually attached here) ----------
    model_obj  = EGRFreed7TT(mg_kernel=kernel)
    model_func = build_model_function(model_obj)
    param_order, p0, lb, ub = build_param_dict_and_bounds()

    # ---------- epsilon sweep (optional) ----------
    sweep_table, best = [], None
    if args.sweep.strip():
        sweep_vals = [float(x.strip()) for x in args.sweep.split(",") if x.strip()]
        print("\n=== ε_model sweep ==="); print("ε_model   χ²_red      Q_exp     ΔH_mix/N (nats)")
        for s in sweep_vals:
            res = run_fit(s, ell, Dl_data, Dl_err_planck, model_func, p0, lb, ub,
                          nstarts=args.starts, seed=args.seed)
            if res is None:
                print(f"{s:7.4f}  (fit failed)"); continue
            print(f"{s:7.4f}  {res['chi2_red']:7.3f}  {res['Qexp']:10.6f}  {res['deltaH_pp_nats']:10.6f}")
            sweep_table.append({
                "epsilon_model": res["epsilon_model"],
                "chi2_red": res["chi2_red"],
                "Qexp": res["Qexp"],
                "deltaH_pp_nats": res["deltaH_pp_nats"],
            })
            if best is None or abs(res["chi2_red"] - 1.0) < abs(best["chi2_red"] - 1.0):
                best = res
        if best:
            print(f"\nBest by |χ²_red-1|: ε_model={best['epsilon_model']:.6f}  "
                  f"χ²_red={best['chi2_red']:.3f}  Q_exp={best['Qexp']:.6f}")

    # ---------- pinned run ----------
    pinned = run_fit(args.epsilon_model, ell, Dl_data, Dl_err_planck, model_func, p0, lb, ub,
                     nstarts=args.starts, seed=args.seed)
    if pinned is None:
        print("Pinned run failed."); sys.exit(2)

    R_gate, phi_ac, A_ac, A_pk, k_pk, sigma_pk, S_resp = pinned["popt"]
    Qexp = pinned["Qexp"]
    _, _, rvec = whitened_residual_stats(Dl_data, pinned["preds"], pinned["err"])
    dof = max(len(ell) - len(pinned["popt"]), 1)
    gof = gof_and_normality(rvec, dof)
    AIC, BIC = aic_bic(gof["chi2"], k=len(pinned["popt"]), N=len(ell))

    # quick visibility of μ(k) if kernel is active
    if kernel is not None:
        kvec = ell / 14000.0
        mu_vec = np.fromiter((model_obj.mu_k(1.0, float(ki)) for ki in kvec), dtype=float, count=len(kvec))
        print(f"[mg] mu(k) stats  mean={np.mean(mu_vec):.4f}  min={np.min(mu_vec):.4f}  max={np.max(mu_vec):.4f}")

    # ---------- AR(1) diagnostics ----------
    ar1_block = {}
    if args.ar1:
        z = (Dl_data - pinned["preds"]) / pinned["err"]
        rho_star, chi2_star = fit_ar1_rho(z)
        redchi_ar1 = chi2_star / dof
        print(f"\nAR(1) residual model: ρ* = {rho_star:+.3f},  χ²_red(AR1) = {redchi_ar1:.3f}")
        ar1_block = {"rho_star": float(rho_star), "redchi_ar1": float(redchi_ar1)}

    ar1_cal = {}
    if args.ar1_calibrate:
        s_star, rho_s, rc_s = calibrate_epsilon_ar1(Dl_data, pinned["preds"], Dl_err_planck,
                                                    s_init=args.epsilon_model)
        Hn, Hn_pm, Hb, Hb_pm = ledger_info_overhead(Dl_err_planck, Dl_data, s_star)
        print("\n[AR(1) diagnostic re-calibration]")
        print(f"  ε_model^AR1 = {s_star:.4f}  (pinned {args.epsilon_model:.4f})")
        print(f"  ρ*          = {rho_s:+.3f}")
        print(f"  χ²_red(AR1) = {rc_s:.3f}")
        ar1_cal = {"epsilon_star": float(s_star), "rho_star": float(rho_s), "redchi_ar1": float(rc_s),
                   "total_nats": float(Hn), "per_mode_nats": float(Hn_pm),
                   "total_bits": float(Hb), "per_mode_bits": float(Hb_pm)}

    # ---------- residual ACF, save tables/plots ----------
    acf = residual_acf(rvec, 40)
    thr = 2/np.sqrt(len(rvec))
    nz_lags = (np.where(np.abs(acf[1:]) > thr)[0] + 1).tolist()

    save_fit_table(tables_dir / "fit_points.csv", ell, Dl_data, pinned["preds"], pinned["err"], rvec)
    qqplot_normal(figs_dir / "cmbtt_qq.png", rvec)

    row = {
        "ID":"CMBTT",
        "n_points_used": int(len(ell)),
        "ell_min": float(np.min(ell)),
        "ell_max": float(np.max(ell)),
        "Dl_max": float(np.max(Dl_data)),
        "model_name": "EGR-Freed7",
        "epsilon_model_pinned": float(pinned["epsilon_model"]),
        "chi2_red": float(pinned["chi2_red"]),
        "p_chi2": float(gof["p_chi2"]),
        "AIC": float(AIC),
        "BIC": float(BIC),
        "Qexp": float(Qexp),
        "R_gate": float(R_gate), "phi_ac": float(phi_ac), "A_ac": float(A_ac),
        "A_pk": float(A_pk), "k_pk": float(k_pk), "sigma_pk": float(sigma_pk), "S_resp": float(S_resp),
        "rho_AR1": float(ar1_block.get("rho_star", np.nan)),
        "acf_lags_over_2_over_sqrtN": ";".join(str(x) for x in nz_lags[:12]),
        "deltaH_mix_total_nats": float(pinned["deltaH_tot_nats"]),
        "deltaH_mix_per_mode_nats": float(pinned["deltaH_pp_nats"])
    }
    summarize_and_save(row, tables_dir / "summary.csv")

    posthoc = {
        "sweep": sweep_table,
        "gof": gof,
        "info_criteria": {"AIC": float(AIC), "BIC": float(BIC)},
        "ledger_info": {
            "total_nats": pinned["deltaH_tot_nats"], "per_mode_nats": pinned["deltaH_pp_nats"],
            "total_bits": pinned["deltaH_tot_bits"], "per_mode_bits": pinned["deltaH_pp_bits"]
        },
        "acf": {"threshold": float(thr), "lags_over_threshold": nz_lags},
        "param_order": ["R_gate","phi_ac","A_ac","A_pk","k_pk","sigma_pk","S_resp"],
        "hit_bounds": pinned["hit_bounds"],
        "mg_kernel": {"enabled": bool(kernel is not None), "config": mg_cfg, "acceptance": mg_accept}
    }
    if ar1_block: posthoc["AR1"] = ar1_block
    if ar1_cal:   posthoc["AR1_calibrate"] = ar1_cal
    if args.band_scatter:
        bands = ledger_band_overhead(ell, Dl_err_planck, Dl_data, args.epsilon_model)
        posthoc["band_ledger_info"] = [
            {"ell_min": int(a), "ell_max": int(b), "H_nats": float(H), "H_per_mode_nats": float(Hp)}
            for (a,b,H,Hp) in bands
        ]
    (tables_dir / "posthoc.json").write_text(json.dumps(posthoc, indent=2))

    # ---------- plots ----------
    if args.plot:
        plt.figure(figsize=(10,6))
        plt.errorbar(ell, Dl_data, yerr=pinned["err"], fmt='.', color='white', alpha=0.7, label="Data")
        plt.plot(ell, pinned["preds"], 'r-', label=f"EGR-Freed7 (ε_model={pinned['epsilon_model']:.4f})")
        plt.yscale("log"); plt.xlabel(r"$\ell$"); plt.ylabel(r"$D_\ell\ [\mu {\rm K}^2]$")
        plt.title(r"CMB TT — EGR-Freed7 (pinned $\varepsilon_{\rm model}$)")
        plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
        plt.savefig(figs_dir / "cmbtt_fit.png", dpi=200)

        plt.figure(figsize=(10,4))
        plt.axhline(0, lw=1, alpha=0.6)
        plt.plot(ell, rvec, '.', ms=2)
        plt.xlabel(r"$\ell$"); plt.ylabel("whitened residual"); plt.title("Whitened residuals")
        plt.grid(alpha=0.3); plt.tight_layout(); plt.savefig(figs_dir / "cmbtt_residuals.png", dpi=200)

    # ---------- MCMC (optional) ----------
    if args.mcmc:
        _, _, lb2, ub2 = build_param_dict_and_bounds()
        sampler, chain = mcmc_run(pinned["popt"], ell, Dl_data, pinned["err"],
                                  lb2, ub2, model_func,
                                  walkers=args.walkers, steps=args.steps, burn=args.burn, seed=args.seed)
        if chain is not None and chain.size:
            names = ["R_gate","phi_ac","A_ac","A_pk","k_pk","sigma_pk","S_resp"]
            med = np.median(chain, axis=0)
            lo  = np.quantile(chain, 0.16, axis=0)
            hi  = np.quantile(chain, 0.84, axis=0)
            qexp_samples = chain[:,0] * chain[:,-1]
            mcmc_out = {"params": {"names": names,
                                   "median": med.tolist(), "p16": lo.tolist(), "p84": hi.tolist(),
                                   "Qexp_median": float(np.median(qexp_samples)),
                                   "Qexp_p16": float(np.quantile(qexp_samples, 0.16)),
                                   "Qexp_p84": float(np.quantile(qexp_samples, 0.84))}}
            try: mcmc_out["acceptance_fraction"] = float(np.mean(sampler.acceptance_fraction))
            except Exception: pass
            try: mcmc_out["tau"] = [float(x) for x in sampler.get_autocorr_time(tol=0)]
            except Exception: pass

            try:
                import corner
                fig = corner.corner(chain, labels=names, quantiles=[0.16,0.5,0.84],
                                    show_titles=True, title_fmt=".3f")
                fig.suptitle("MCMC — EGR-Freed7 (fixed ε_model)", fontsize=14)
                fig.savefig(figs_dir / "cmbtt_corner.png", dpi=160)
            except Exception:
                pass

            C = corr_heatmap_from_chain(figs_dir / "cmbtt_corr.png", chain, names)
            if C is not None: mcmc_out["corr"] = {"matrix": C.tolist(), "names": names}

            q16,q50,q84 = posterior_predictive_band(chain, model_func, ell, n_draws=args.ppc_samples)
            if args.plot:
                plt.figure(figsize=(10,6))
                plt.errorbar(ell, Dl_data, yerr=pinned["err"], fmt='.', color='white', alpha=0.6, label="Data")
                plt.fill_between(ell, q16, q84, alpha=0.25, label="PPC 68% band")
                plt.plot(ell, q50, '-', label="PPC median")
                plt.yscale("log"); plt.xlabel(r"$\ell$"); plt.ylabel(r"$D_\ell\ [\mu {\rm K}^2]$")
                plt.title("Posterior Predictive — EGR-Freed7 (fixed ε_model)")
                plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
                plt.savefig(figs_dir / "cmbtt_ppc.png", dpi=200)

            ph = json.loads((tables_dir / "posthoc.json").read_text())
            ph["mcmc"] = mcmc_out
            (tables_dir / "posthoc.json").write_text(json.dumps(ph, indent=2))

    if args.plot:
        plt.show()

    print("\n=== Pinned run (saved) ===")
    print(f" outdir : {outdir}")
    print(f" Q_exp  : {Qexp:.6f}")
    print(f" χ²_red : {pinned['chi2_red']:.3f}  |  p(χ²,dof) = {gof['p_chi2']:.3g}")
    print(" figures:", [str(p) for p in outdir.joinpath('figs').glob('*.png')])
    print(" tables :", [str(p) for p in outdir.joinpath('tables').glob('*.*')])


def build_argparser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="COM_PowerSpect_CMB-TT-full_R3.01.txt")
    ap.add_argument("--outdir", default="out_cmb")
    ap.add_argument("--epsilon-model", type=float, default=0.075, dest="epsilon_model")
    ap.add_argument("--model-scatter", type=float, dest="epsilon_model", help=argparse.SUPPRESS)
    ap.add_argument("--sweep", type=str, default="")
    ap.add_argument("--ar1", action="store_true")
    ap.add_argument("--ar1-calibrate", action="store_true")
    ap.add_argument("--band-scatter", action="store_true")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--mcmc", action="store_true")
    ap.add_argument("--walkers", type=int, default=64)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--burn", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ppc-samples", type=int, default=200)
    ap.add_argument("--ell-min", type=int, default=None)
    ap.add_argument("--ell-max", type=int, default=None)
    ap.add_argument("--starts", type=int, default=16)

    ap.add_argument("--elm-enable", action="store_true")
    ap.add_argument("--elm-mode", choices=["flat","bounded"], default="bounded")
    ap.add_argument("--elm-alpha", type=float, default=0.35)
    ap.add_argument("--elm-beta", type=float, default=0.35)
    ap.add_argument("--elm-gates", default="")
    ap.add_argument("--elm-phase-scale", type=float, default=1.0)
    ap.add_argument("--elm-scale", type=float, default=1.0)

    # MG / Wilsons (only logged unless you wire it in the solver)
    ap.add_argument("--mg-config", default=None,
                    help="Path to JSON with IR-EFT Wilson mapping (see configs/wilson_baseline.json)")
    ap.add_argument("--use-mg", action="store_true",
                    help="Enable modified-gravity kernel for scalar sector (mu/eta).")
    return ap

def main():
    ap = build_argparser()
    args = ap.parse_args()
    run_cmb(args)
