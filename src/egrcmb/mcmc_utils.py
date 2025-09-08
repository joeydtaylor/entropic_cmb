import numpy as np
try:
    import emcee, corner  # noqa: F401
    HAS_EMCEE = True
except Exception:
    HAS_EMCEE = False

def mcmc_run(popt_seed, ell, Dl, err, lb, ub, model_func, walkers=64, steps=1500, burn=300, seed=42):
    if not HAS_EMCEE:
        print("emcee/corner not installed; skipping MCMC.")
        return None, None
    ndim = len(popt_seed)
    rng = np.random.default_rng(seed)
    pos = []
    for _ in range(walkers):
        jitter = popt_seed * (1.0 + 0.02*rng.standard_normal(ndim))
        pos.append(np.clip(jitter, lb, ub))

    def log_prior(theta):
        for v,l,u in zip(theta, lb, ub):
            if v < l or v > u: return -np.inf
        return 0.0

    def log_like(theta):
        m = model_func(ell, *theta); r = (Dl - m)/err
        return -0.5*np.sum(r**2)

    def log_prob(theta):
        lp = log_prior(theta)
        if not np.isfinite(lp): return -np.inf
        return lp + log_like(theta)

    sampler = emcee.EnsembleSampler(walkers, ndim, log_prob)
    sampler.run_mcmc(pos, steps, progress=True)
    chain = sampler.get_chain(discard=burn, flat=True)
    return sampler, chain

def posterior_predictive_band(chain, model_func, ell, n_draws=200, rng_seed=123):
    rng = np.random.default_rng(rng_seed)
    idx = rng.integers(0, chain.shape[0], size=min(n_draws, chain.shape[0]))
    preds = np.array([model_func(ell, *chain[i]) for i in idx])
    return (np.percentile(preds, 16, axis=0),
            np.percentile(preds, 50, axis=0),
            np.percentile(preds, 84, axis=0))
