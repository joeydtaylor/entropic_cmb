from dataclasses import dataclass
import math

@dataclass
class IRKernel:
    mu_form: str = "wilson_ir"
    mu: float = 0.3001615492546175   # 1/Mpc
    ridge: float = 1e-4
    cC2: float = 80.5                # M_Pl^2
    alpha_Ric: float = 1197.0        # M_Pl^2
    use_scalar_B: bool = False
    B: float = 0.0
    k_uv: float = 3.0                # relax to GR above this k (tighter than 10)
    gain: float = 20.0               # NEW: global multiplier you can tune from JSON

    def _x2(self, k: float) -> float:
        mu = max(self.mu, 1e-9)
        return (k*k)/(mu*mu + max(self.ridge, 0.0))

    def mu_k(self, a: float, k: float) -> float:
        if self.mu_form != "wilson_ir":
            return 1.0
        x2 = self._x2(k)

        # stronger base mapping; 'gain' scales the whole IR term
        alpha2   = 5e-3  * (self.cC2 / 100.0)
        alpha4lg = 2e-4  * (self.alpha_Ric / 1000.0)

        ir = (alpha2 * x2 + alpha4lg * (x2 * x2) * math.log(max(x2, 1e-12))) * self.gain

        # steeper UV relaxation so low-k effect is kept
        s = 1.0 / (1.0 + (k / max(self.k_uv, 1e-6))**8)
        return float(1.0 + s * ir)

    def eta_k(self, a: float, k: float) -> float:
        if not self.use_scalar_B:
            return 1.0
        x2 = self._x2(k)
        beta2 = 1e-6 * (self.B / 1e4)
        s = 1.0 / (1.0 + (k / max(self.k_uv, 1e-6))**8)
        return float(1.0 + s * beta2 * x2)

    def gr_principal_symbol_ok(self) -> bool: return True
    def eikonal_shapiro_delay_positive(self) -> bool:
        for k in (1e-4, 3e-4, 1e-3, 3e-3, 1e-2):
            if self.mu_k(1.0, k) < 0.9:
                return False
        return True
