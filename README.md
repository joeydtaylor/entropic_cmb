# EGR-Freed7 — CMB TT, end-to-end, Docker-only

Pinned-ε CMB TT fitter with ΔIC diagnostics, AR(1) residual modeling, optional ε-sweep, MCMC on physics parameters, and posterior-predictive bands. **Zero local setup. One image. One command.**

---

## Quickstart

```bash
# from repo root
docker build -t egr-cmb .

# Full run on Planck TT file (plots + AR(1) + band-entropy + MCMC)
docker run --rm \
  -v "$PWD/data:/work/data" \
  -v "$PWD/out_cmb:/work/out_cmb" \
  egr-cmb egr-cmb-run-all \
    --data /work/data/COM_PowerSpect_CMB-TT-full_R3.01.txt \
    --outdir /work/out_cmb
````

**Artifacts** (all reproducible):

```
out_cmb/
  figs/
    cmbtt_fit.png          # best-fit D_ℓ (ε_model pinned)
    cmbtt_residuals.png    # whitened residuals vs ℓ
    cmbtt_qq.png           # Q–Q plot of whitened residuals
    cmbtt_corner.png       # (if --mcmc) corner plot
    cmbtt_corr.png         # (if --mcmc) posterior corr heatmap
    cmbtt_ppc.png          # (if --mcmc --plot) PPC median + 68% band
  tables/
    summary.csv            # recap row for pinned ε run
    fit_points.csv         # per-ℓ: ℓ, D_ℓ, model, σ_eff, z
    posthoc.json           # sweep/IC/AR1/ledger/MCMC metadata
  run.json                 # args, data SHA256, library versions
```

---

## Data

Place the Planck TT 4-column file (ℓ, D\_ℓ, σ₋, σ₊) under `data/` (or mount it):

```
data/COM_PowerSpect_CMB-TT-full_R3.01.txt
```

Optional ℓ-windowing: `--ell-min`, `--ell-max`.

---

## Model (EGR-Freed7)

* **Uncertainty (pinned mix proxy)**
  $\sigma_{\rm eff}^2(\ell) = \sigma_{\rm Planck}^2(\ell) + (\varepsilon_{\rm model}\, D_\ell)^2$

* **Gate scale**
  $Q_{\rm exp} = R_{\rm gate}\, S_{\rm resp}$

* **Fitted physics parameters**
  $[R_{\rm gate}, \phi_{\rm ac}, A_{\rm ac}, A_{\rm pk}, k_{\rm pk}, \sigma_{\rm pk}, S_{\rm resp}]$

* **Diagnostics**
  χ²\_red, p(χ²;dof), AIC/BIC, bound-hits, residual ACF exceedances, information-ledger (total/per-mode).

* **Optionals**
  ε-sweep, AR(1) residual modeling (ρ\* and χ²\_red(AR1)), MCMC on physics parameters (ε pinned), PPC band.

---

## Common runs

**Everything on (recommended)**

```bash
docker run --rm \
  -v "$PWD/data:/work/data" \
  -v "$PWD/out_cmb:/work/out_cmb" \
  egr-cmb egr-cmb-run-all \
    --data /work/data/COM_PowerSpect_CMB-TT-full_R3.01.txt \
    --outdir /work/out_cmb
```

**Pinned run (manual flags)**

```bash
docker run --rm \
  -v "$PWD/data:/work/data" \
  -v "$PWD/out_cmb:/work/out_cmb" \
  egr-cmb egr-cmb \
    --data /work/data/COM_PowerSpect_CMB-TT-full_R3.01.txt \
    --outdir /work/out_cmb \
    --plot --ar1 --ar1-calibrate --band-scatter --mcmc \
    --ppc-samples 200 --starts 16 --seed 42
```

**ε-sweep** (compare χ²\_red vs ε; prints Q\_exp and ΔH/N)

```bash
docker run --rm \
  -v "$PWD/data:/work/data" \
  -v "$PWD/out_cmb:/work/out_cmb" \
  egr-cmb egr-cmb \
    --data /work/data/COM_PowerSpect_CMB-TT-full_R3.01.txt \
    --outdir /work/out_cmb \
    --sweep "0.04,0.06,0.075,0.09,0.12"
```

**ℓ-window** (Planck-like)

```bash
... egr-cmb ... --ell-min 2 --ell-max 2500
```

---

## CLI knobs (most useful)

* `--epsilon-model 0.075` : fixed ε used in σ\_eff (can sweep via `--sweep`).
* `--plot` : save fit, residuals, Q–Q plots.
* `--ar1` : AR(1) residual model; reports ρ\* and χ²\_red(AR1).
* `--ar1-calibrate` : pick ε\* s.t. χ²\_red(AR1)≈1; records ΔH totals/per-mode.
* `--band-scatter` : bandwise ΔH (nats/mode).
* `--mcmc` : emcee on physics parameters (ε fixed); `--walkers/--steps/--burn`.
* `--ppc-samples 200` : draws for PPC band.
* `--starts 16` : TRF multi-start (soft\_l1).
* `--ell-min/--ell-max` : restrict ℓ range.
* `--seed` : RNG seed for reproducibility.

---

## Determinism

* `run.json` logs args, data SHA-256, and versions.
* The RNG seed (`--seed`) is used for multi-start jitters and MCMC.
* `summary.csv` schema is guarded: if the header changes, it rotates automatically.

---

## Repository layout

```
src/egrcmb/
  cli.py         # CLI and full pipeline
  run_all.py     # “everything on” wrapper
  constants.py   # numerics + pinned physics constants
  elm.py         # Entropy Ledger Matrix (diagnostics; optional)
  model.py       # EGRFreed7TT + bounds/builders
  fit.py         # error model, multi-start fitting, ledger totals
  ar1.py         # AR(1), ACF, band-entropy
  mcmc_utils.py  # emcee wrapper, PPC band
  diagnostics.py # GoF, AIC/BIC, tables, QQ, corr heatmap
data/
  COM_PowerSpect_CMB-TT-full_R3.01.txt   # mount or place here
out_cmb/                                  # artifacts (gitignored)
Dockerfile
pyproject.toml
requirements.txt
README.md
```

---

## Troubleshooting

* **No outputs**: mount `out_cmb` so the container can write:

  ```
  -v "$PWD/out_cmb:/work/out_cmb"
  ```
* **Data load error**: file must have 4 numeric columns (ℓ, D\_ℓ, σ₋, σ₊).
* **MCMC heavy**: reduce `--steps`, omit `--mcmc` for quick passes.
* **ACF threshold**: JSON reports the level `2/√N` and lags that exceed it.
