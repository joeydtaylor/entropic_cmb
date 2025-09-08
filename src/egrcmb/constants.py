import numpy as np

# Numerics / style
PI  = np.pi
PHI = (1 + np.sqrt(5.0)) / 2.0
MAXFEV_FIT = 40_000

# Pinned constants (identical to your script)
GROUP_ORDER_TRIVIAL = 1.0
GROUP_ORDER_V4      = 4.0
GROUP_ORDER_S4      = 24.0
GROUP_ORDER_D6STAR  = 36.0
BASE_DIMENSIONS          = 10
BASE_DIMENSIONS_SQUARED  = BASE_DIMENSIONS**2
BASE_DIMENSIONS_CUBED    = BASE_DIMENSIONS**3
BASE_DEGREES_OF_FREEDOM  = GROUP_ORDER_D6STAR * BASE_DIMENSIONS_SQUARED
PROTON_ELECTRON_MASS_RATIO = 1836.15267

J_AMPLITUDE        = GROUP_ORDER_TRIVIAL / GROUP_ORDER_S4
coupling_amp_fixed = GROUP_ORDER_TRIVIAL / GROUP_ORDER_V4
coupling_scale     = BASE_DIMENSIONS_SQUARED
sph_harm_base = GROUP_ORDER_TRIVIAL / (GROUP_ORDER_V4 * PI)
sph_harm_amplitude_fixed = sph_harm_base * BASE_DIMENSIONS
scalar_scale = GROUP_ORDER_TRIVIAL / PHI
scalar_amplitude_fixed = (2.0 / GROUP_ORDER_TRIVIAL) - ((2.0 / GROUP_ORDER_TRIVIAL) / BASE_DIMENSIONS_CUBED)
lens_scale = BASE_DIMENSIONS_CUBED
acoustic_amplitude_fixed = PHI / (BASE_DIMENSIONS + GROUP_ORDER_TRIVIAL)
acoustic_scale_fixed     = ((BASE_DIMENSIONS + GROUP_ORDER_TRIVIAL) * PI) / (BASE_DIMENSIONS * BASE_DIMENSIONS_CUBED)
fg_amp_fixed = BASE_DIMENSIONS_CUBED / (2.0 / GROUP_ORDER_TRIVIAL)  # 500
fg_slope     = -acoustic_amplitude_fixed
differential_rotation_shear_parameter_fixed = (scalar_amplitude_fixed**3 * PI / 3.0)
ELL_SILK_FIXED = fg_amp_fixed * (PI**2)
resonance_damp_slope   = 3 / GROUP_ORDER_V4
resonance_phase_shift  = np.sqrt(128.0)
k_qp_factor_fixed      = GROUP_ORDER_TRIVIAL / 3.0
k_si_factor_fixed      = np.sin(np.deg2rad(13.5))
damping_exponent_fixed = PHI + GROUP_ORDER_TRIVIAL
