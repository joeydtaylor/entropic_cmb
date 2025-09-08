import numpy as np

class EntropyLedgerMatrix:
    def __init__(self, mode="bounded", alpha=0.35, beta=0.35, rdc_scale=1.0, pac_scale=1.0):
        if mode not in {"bounded","flat"}: raise ValueError("mode must be 'bounded' or 'flat'")
        self.mode, self.alpha, self.beta = mode, float(alpha), float(beta)
        self.rdc_scale = float(max(rdc_scale, 1e-12)); self.pac_scale = float(max(pac_scale, 1e-12))
        self.phi = (1.0 + np.sqrt(5.0))/2.0
        self.E_plus, self.E_minus = self.phi, 1.0/self.phi
        self.RETURN_RATE, self.BORROW_RATE = 1.0/self.phi, 1.0/(self.phi**2)
        self.TRANSACTION_COST = self.RETURN_RATE - self.BORROW_RATE

    @staticmethod
    def _tanh(x): return float(np.tanh(x))
    @staticmethod
    def _sigm(x): return float(1.0/(1.0+np.exp(-x)))

    def _weight(self, r_idx, c_idx):
        if self.mode == "flat": return 1.0
        r = float(np.hypot(r_idx, c_idx)) + 1e-9
        rdc = 1.0 + self.alpha * self._tanh((1.0/r)/self.rdc_scale)
        phase = 2.0*np.pi*(r_idx+c_idx)/8.0
        pac = 1.0 + self.beta * (2.0*self._sigm(np.cos(phase)/self.pac_scale) - 1.0)
        return float(np.clip(rdc*pac, 1e-6, 1e6))

    def _build(self, pattern_fn):
        M = np.zeros((8,8), dtype=complex)
        for r in range(8):
            for c, base in pattern_fn(r):
                M[r,c] = complex(base)*self._weight(r,c)
        M[np.abs(M)<1e-12]=0.0
        return M

    def _pat_hanging_left(self):
        def p(row):
            out=[]; L=8-row
            for i in range(L):
                sign = 1 if (i%2==0) else -1
                energy = self.E_plus if sign>0 else self.E_minus
                out.append((i, sign*energy))
            return out
        return p

    def _pat_hanging_right_inverted(self):
        def p(row):
            out=[]; L=8-row
            for i in range(L):
                sign = 1 if (i%2==0) else -1
                energy = self.E_minus if sign>0 else self.E_plus
                col = 7 - i
                out.append((col, sign*energy))
            return out
        return p

    def matrix_hanging_left(self): return self._build(self._pat_hanging_left())
    def matrix_hanging_right_inverted(self): return self._build(self._pat_hanging_right_inverted())
    def bank(self): return {"hanging_left": self.matrix_hanging_left(),
                            "hanging_right_inverted": self.matrix_hanging_right_inverted()}
    # Optional gates (unchanged)
    def transform_and(self, M, phase_scale=1.0):
        q = np.array([[ self.phi, 1.0, self.phi**-1, 0.0],
                      [ 0.0, self.phi, 1.0, self.phi**-1],
                      [ self.phi**-1, 0.0, self.phi, 1.0],
                      [ 1.0, self.phi**-1, 0.0, self.phi]], dtype=complex)
        G8 = np.kron(q, np.eye(2, dtype=complex))
        return M.astype(complex) @ (np.exp(1j*np.pi/self.phi*phase_scale)*G8)
    def transform_xor(self, M, scale=1.0):
        H = (1.0/np.sqrt(2.0))*np.array([[1.0,1.0],[1.0,-1.0]], dtype=complex)
        H8 = np.kron(np.kron(H,H),H); return M.astype(complex) @ (H8*(np.pi/self.phi)*scale)
    def transform_not(self, M, scale=1.0):
        X = np.array([[0.0,1.0],[1.0,0.0]], dtype=complex)
        X8 = np.kron(np.kron(np.eye(2, dtype=complex), np.eye(2, dtype=complex)), X)
        return M.astype(complex) @ (X8*(np.pi/self.phi)*scale)
    def apply_transformation(self, gate, M, **kw):
        if gate=="S_AND": return self.transform_and(M, **kw)
        if gate=="S_XOR": return self.transform_xor(M, **kw)
        if gate=="S_NOT": return self.transform_not(M, **kw)
        raise ValueError(f"Unsupported gate: {gate}")
