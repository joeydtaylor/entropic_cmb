import numpy as np
from .constants import *

class EGRFreed7TT:
    def __init__(self, mg_kernel=None):
        self.mg_kernel = mg_kernel  # optional IR kernel; not used unless you call mu_k/eta_k
        self.degrees_of_freedom = BASE_DEGREES_OF_FREEDOM
        self.rotation_shear     = differential_rotation_shear_parameter_fixed
        self.vector_amp         = 12.0 * PI
        self.coupling_amp_fixed = coupling_amp_fixed
        self.J_amp_fixed        = J_AMPLITUDE
        self.sph_harm_amplitude = sph_harm_amplitude_fixed
        self.acoustic_amplitude = acoustic_amplitude_fixed
        self.acoustic_scale     = acoustic_scale_fixed
        self.ell_silk_fixed     = ELL_SILK_FIXED
        self.ell_recomb_fixed   = PROTON_ELECTRON_MASS_RATIO

    def mu_k(self, a: float, k: float) -> float:
        # placeholder; returns GR unless you decide to use it in sourcing
        return 1.0 if self.mg_kernel is None else self.mg_kernel.mu_k(a, k)

    def eta_k(self, a: float, k: float) -> float:
        return 1.0 if self.mg_kernel is None else self.mg_kernel.eta_k(a, k)

    def _j_invariant_contribution(self, ell):
        scale_safe = coupling_scale + 1e-10
        q = np.exp(-2.0 * PI * ell / scale_safe)
        j_term = abs((1.0 / q) + 744.0 + 196884.0 * q)
        j_norm = np.log1p(j_term) / np.log(1e5)
        k = ell / scale_safe
        mod = 1.0 + 0.1 * np.cos(2.0*PI * k / (scale_safe / 10.0))
        return self.J_amp_fixed * j_norm * mod

    def _vector_mode_contribution(self, ell, Qexp):
        ts = 2.0*PI*(1.0-1.0/PHI**2)*self.rotation_shear
        x = ell/(ts+1e-10)
        rise = x/PI
        plateau = 1.0/(1.0+np.finfo(float).eps*np.log1p(x))
        mod = Qexp/(Qexp+1.0)
        return self.vector_amp * rise * (1.0 + plateau) * mod

    def _scalar_mode_contribution(self, ell, Qexp):
        trans = 1.0 - np.exp(-scalar_scale*ell/100.0)
        h_scale = Qexp/(Qexp+1e-10)
        return scalar_amplitude_fixed * trans / -2.0 * h_scale

    def _tensor_mode_contribution(self, ell):
        ell_max = max(np.max(ell),1e-10)
        ell_scaled = ell/ell_max
        growth = 1.0 - np.exp(-PHI*ell_scaled)
        silk_damp = np.exp(-((ell/ELL_SILK_FIXED)**PHI))
        recomb_damp = 1.0/(1.0+(ell/self.ell_recomb_fixed)**PHI)
        return self.degrees_of_freedom * growth * silk_damp * recomb_damp

    def _coupling_mode_contribution(self, ell):
        x = ell/(coupling_scale + 1e-10)
        trans = 1.0 - np.exp(-x)
        shadow = np.exp(-x/10.0)/(1.0 + x**PHI)
        return self.coupling_amp_fixed * (trans + 0.1*shadow) / -1.0

    def _resonance_mode_contribution(self, ell):
        amplitude = (self.coupling_amp_fixed*10.0) + self.acoustic_amplitude
        freq = self.rotation_shear
        ell_max = max(np.max(ell),1e-10)
        ell_scaled = ell/ell_max
        wave = np.sin(freq*ell_scaled+PI/resonance_phase_shift)
        damp = 1.0/(1.0+resonance_damp_slope*ell_scaled)
        return amplitude*wave/damp

    def _acoustic_modulation(self, ell, Qexp, phi_ac, A_ac, A_pk, k_pk, sigma_pk, S_resp):
        return self._acoustic_shaping(
            ell, self.acoustic_amplitude, self.acoustic_scale, Qexp,
            k_qp_factor_fixed, k_si_factor_fixed, damping_exponent_fixed,
            phi_ac, A_ac, A_pk, k_pk, sigma_pk, S_resp
        )

    def _acoustic_shaping(self, ell, amplitude, scale, Qexp,
                          k_qp_factor, k_si_factor, damping_exponent,
                          wave_freq, wave_ampl_scaling, peak_amp,
                          peak_center, peak_width, S_resp):
        ell = np.asarray(ell, dtype=float)
        k = ell / 14000.0
        k_qp = k_qp_factor * np.sqrt(amplitude) * (Qexp/0.3)**0.25
        k_si = k_si_factor * (scale**0.25) * np.sqrt(amplitude)
        template = np.exp(-((k/k_qp)**damping_exponent))
        wave = np.cos(2.0*PI*k/k_si + PI/wave_freq)
        peak1 = peak_amp * np.exp(-((k - peak_center)**2)/(peak_width+1e-15))
        entropy_mod = 1.0 + S_resp * np.exp(-k/50.0)
        return (1.0 + wave_ampl_scaling * template * (wave + peak1)) * entropy_mod

    def predict_Dl(self, ell, R_gate, phi_ac, A_ac, A_pk, k_pk, sigma_pk, S_resp):
        """
        TT template with optional IR-kernel modulation.
        If self.mg_kernel is None → GR (no modulation).
        If present → multiply the lensed template by μ(k) with k = ℓ / 14000.
        """
        Qexp = R_gate * S_resp

        # base components (unchanged)
        vec = self._vector_mode_contribution(ell, Qexp)
        sca = self._scalar_mode_contribution(ell, Qexp)
        base_vs = vec * (1.0 + sca)
        ten = self._tensor_mode_contribution(ell)
        modulated = base_vs * (1.0 + ten)
        coup = self._coupling_mode_contribution(ell)
        res  = self._resonance_mode_contribution(ell)
        combo_cr = 1.0 + res * coup
        j_bg  = 1.0 + self._j_invariant_contribution(ell)
        acoustic = self._acoustic_modulation(ell, Qexp, phi_ac, A_ac, A_pk, k_pk, sigma_pk, S_resp)
        combined = modulated * combo_cr * j_bg * acoustic

        # lensing + foregrounds (as before)
        lens = np.exp(-ell * (ell + 1.0) / (2.0 * lens_scale**2))
        lensed_model = sph_harm_amplitude_fixed * combined * lens
        ell_safe = np.where(ell == 0, 1.0, ell)
        foreground = fg_amp_fixed * (ell_safe**fg_slope)

        # --- NEW: IR kernel modulation (active only if mg_kernel is set) ---
        if self.mg_kernel is not None:
            # map ℓ to comoving k; keep consistent with your acoustic shaping
            k = np.asarray(ell, dtype=float) / 14000.0
            # μ(k) evaluated at a≈1 for TT sensitivity; vectorize safely
            mu_vec = np.fromiter((self.mu_k(1.0, float(ki)) for ki in k), dtype=float, count=len(k))
            mu_vec = np.clip(mu_vec, 0.7, 1.3)
            # gentle guard against pathological configs
            lensed_model = lensed_model * mu_vec

        return lensed_model + foreground

def build_param_dict_and_bounds():
    param_order = ["R_gate","phi_ac","A_ac","A_pk","k_pk","sigma_pk","S_resp"]
    p0 = np.array([1.0/(PHI**2), PHI**2, 1.0/(PHI**2), PHI, 0.02, np.sqrt(2)/100, 0.95], float)
    lb = np.array([PHI*1e-8, 1.2, 0.0, 0.0, 0.0, 1e-6, PHI*1e-8], float)
    ub = np.array([1.0, 4.0, 1.0, 2.0, 0.10, 0.05, 1.5], float)
    return param_order, p0, lb, ub

def build_model_function(model_obj: EGRFreed7TT):
    def model_func(ell, R_gate, phi_ac, A_ac, A_pk, k_pk, sigma_pk, S_resp):
        return model_obj.predict_Dl(ell, R_gate, phi_ac, A_ac, A_pk, k_pk, sigma_pk, S_resp)
    return model_func
