"""
Hyperparameter likelihood for EOS inference via piecewise polytrope.

Strategy
--------
Each event i has an injected chirp mass Mc_i (from equal-mass merger m1=m2=m_i)
and a Fisher-matrix posterior that is Gaussian in (Mc, log λ̃).  Because the
Mc uncertainty is negligibly small, we fix m = m_i and evaluate the EOS-predicted
log λ̃ at that mass.  The per-event log-likelihood is then:

    log L_i = log N(log λ̃_EOS(m_i) | μ_i[log λ̃], σ_i[log λ̃])

The total log-likelihood is the sum over all N events.

Performance
-----------
`CreateSimNeutronStarFamily` (the TOV integrator) dominates runtime at ~10 ms
per call.  We build the EOS family once per likelihood call, then evaluate λ
directly at each of the N injected masses from the already-built family.
This reduces `CreateSimNeutronStarFamily` from O(N) to O(1) per likelihood call.

Note on pressure units
----------------------
bilby's `polytrope_or_causal_params_to_lambda_1_lambda_2` takes pressures in
log10(dyn cm⁻²) (CGS) and subtracts 1 internally before passing to LALSim.
The raw LALSim wrappers take log10(Pa) (SI) directly.  We store pressures in
CGS and subtract 1 only at the LALSim call sites inside `_build_family`.
"""

import numpy as np
from scipy.stats import norm

from bilby.core.likelihood import Likelihood
from bilby.gw.conversion import (
    lambda_from_mass_and_family,
    solar_mass,
    gravitational_constant,
    speed_of_light,
)
from bilby.gw.utils import (
    lalsim_SimNeutronStarEOS3PieceDynamicPolytrope,
    lalsim_SimNeutronStarEOS3PDViableFamilyCheck,
    lalsim_CreateSimNeutronStarFamily,
    lalsim_SimNeutronStarEOSMaxPseudoEnthalpy,
    lalsim_SimNeutronStarEOSSpeedOfSoundGeometerized,
    lalsim_SimNeutronStarFamMinimumMass,
    lalsim_SimNeutronStarMaximumMass,
)


def _build_family(p1, lp1_cgs, p2, lp2_cgs, p3, causal):
    """
    Validate EOS parameters and return the LALSim NS family object.

    Steps
    -----
    1. Viability pre-filter (cheap).
    2. Build piecewise-polytrope EOS object (analytic, fast).
    3. Integrate TOV equations to get mass-radius-tidal family (expensive, ~10 ms).
    4. Speed-of-sound causality check.

    Parameters
    ----------
    p1, p2, p3 : float
        Adiabatic indices.
    lp1_cgs, lp2_cgs : float
        Dividing pressures in log10(dyn cm⁻²).
    causal : int

    Returns
    -------
    family : LALSim family object, or None if EOS is unphysical.
    """
    lp1_si = lp1_cgs - 1.0
    lp2_si = lp2_cgs - 1.0

    if lalsim_SimNeutronStarEOS3PDViableFamilyCheck(p1, lp1_si, p2, lp2_si, p3, causal) != 0:
        return None

    eos    = lalsim_SimNeutronStarEOS3PieceDynamicPolytrope(p1, lp1_si, p2, lp2_si, p3)
    family = lalsim_CreateSimNeutronStarFamily(eos)   

    max_h = lalsim_SimNeutronStarEOSMaxPseudoEnthalpy(eos)
    if lalsim_SimNeutronStarEOSSpeedOfSoundGeometerized(max_h, eos) > 1.1:
        return None

    return family


class EOSHyperparameterLikelihood(Likelihood):
    """
    Parameters
    ----------
    cov_matrices : (N, 2, 2) array
        Per-event Fisher covariance in (Mc, log λ̃) space.
    means : (N, 2) array
        Per-event injection means [Mc, log λ̃].
    mass_min, mass_max : float
        Component-mass prior bounds [M☉].
    log10_pressure1_cgs, log10_pressure2_cgs : float
        Dividing pressures in log10(dyn cm⁻²) (CGS).
    causal : bool
    n_samples : int
        Unused; kept for API compatibility.
    """

    def __init__(
        self,
        cov_matrices,
        means,
        mass_min,
        mass_max,
        log10_pressure1_cgs,
        log10_pressure2_cgs,
        causal=False,
        n_samples=None,
    ):
        if len(cov_matrices) != len(means):
            raise ValueError(
                f"cov_matrices (len {len(cov_matrices)}) and means "
                f"(len {len(means)}) must have equal length."
            )
        super().__init__()

        self.mass_min = mass_min
        self.mass_max = mass_max
        self._lp1_cgs = log10_pressure1_cgs   # CGS; SI conversion happens in _build_family
        self._lp2_cgs = log10_pressure2_cgs
        self._causal  = int(causal)

        means        = np.asarray(means, dtype=float)
        cov_matrices = np.asarray(cov_matrices, dtype=float)

        # Equal-mass component masses: Mc = m · 2^{-1/5}  ⟹  m = Mc · 2^{1/5}
        self._m_injected       = means[:, 0] * 2.0 ** (1.0 / 5.0)   # (N,)
        self._log_lam_t_means  = means[:, 1]                          # (N,)
        self._log_lam_t_sigmas = np.sqrt(cov_matrices[:, 1, 1])       # (N,)

    # ------------------------------------------------------------------

    def log_likelihood(self, parameters=None):
        if parameters is None:
            parameters = self.parameters

        family = _build_family(
            parameters["param1"], self._lp1_cgs,
            parameters["param2"], self._lp2_cgs,
            parameters["param3"], self._causal,
        )
        if family is None:
            return -np.inf

        m_min_eos = lalsim_SimNeutronStarFamMinimumMass(family) / solar_mass
        m_max_eos = lalsim_SimNeutronStarMaximumMass(family) / solar_mass

        # Reject if any injected mass falls outside the EOS-supported range
        if np.any(self._m_injected < m_min_eos) or np.any(self._m_injected > m_max_eos):
            return -np.inf

        # Evaluate λ directly at each injected mass from the already-built family
        log_lam_t = np.array([
            np.log(lambda_from_mass_and_family(m, family))
            for m in self._m_injected
        ])

        return np.sum(
            norm.logpdf(log_lam_t, loc=self._log_lam_t_means, scale=self._log_lam_t_sigmas)
        )

    def noise_log_likelihood(self):
        return np.nan
