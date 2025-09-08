import subprocess, sys, json
from pathlib import Path

def test_smoke(tmp_path: Path):
    out = tmp_path/"out_cmb"; out.mkdir(parents=True, exist_ok=True)
    # Make a tiny fake Planck-like 4-col file (ℓ,Dℓ,σ-,σ+)
    dat = tmp_path/"fake_tt.txt"
    import numpy as np
    ell = np.arange(2, 60)
    D   = 1000.0/(1+0.01*ell)
    s   = 0.05*D
    np.savetxt(dat, np.c_[ell, D, s, s])
    cmd = [sys.executable, "-m", "egrcmb.run_all", "--data", str(dat), "--outdir", str(out)]
    subprocess.run(cmd, check=True)
    assert (out/"tables"/"summary.csv").exists()
    assert (out/"tables"/"posthoc.json").exists()
