"""
Hyperparameter likelihood for empirical relation inference.

Goal
----
Infer posteriors on Omega = (alpha, beta, gamma), the parameters of the
quadratic empirical relation:

    f_peak [kHz] = 10^alpha * lam_t^2 - 10^beta * lam_t + gamma

connecting tidal deformability to post-merger peak frequency, using N BNS
merger events with equal component masses.

Method: Hierarchical inference via converse likelihood
------------------------------------------------------
For each hyperparameter sample Omega and each event i, draw M samples
log_lam^(k) from a Gaussian centred on the injected log_lam (i.e. the
inspiral posterior mean), then:

  1. Exponentiate to get lam_t^(k).
  2. Map to f_peak^(k) via the empirical relation.
  3. Evaluate the product of inspiral and post-merger Gaussian likelihoods
     in log-space for numerical stability.
  4. Use logsumexp over k for the per-event MC log-likelihood estimate.

Multiplication of the two likelihood factors per event is valid under
conditional independence of inspiral and post-merger data given lam_t.

Units
-----
- lam_t  : dimensionless
- f_peak : kHz
- All means and variances for lam_t are in log-space (natural log).
- All means and variances for f_peak are in kHz.
"""

import numpy as np
from scipy.special import logsumexp
from scipy.stats import norm

from bilby.core.likelihood import Likelihood


class EMRelationLikelihood(Likelihood):
    """
    Converse likelihood for the empirical relation f_peak = g(lam_t; Omega).

    Subclasses bilby.core.likelihood.Likelihood. The sampler populates
    self.parameters at each iteration; log_likelihood() reads from it directly.

    Parameters
    ----------
    means_log_lam : (N,) array
        Per-event injected log(lam_t) values (inspiral posterior means).
    vars_log_lam : (N,) array
        Per-event log(lam_t) posterior variances from Fisher matrix.
    means_fp : (N,) array
        Per-event post-merger f_peak posterior means [kHz].
    vars_fp : (N,) array
        Per-event post-merger f_peak posterior variances [kHz^2].
    M : int
        Number of Monte Carlo samples per event per likelihood call.
        Default: 10.
    """

    def __init__(self, means_log_lam, vars_log_lam, means_fp, vars_fp, M=10):
        # super().__init__(parameters={"alpha": None, "beta": None, "gamma": None})
        super().__init__()

        self._means_log_lam = np.asarray(means_log_lam, dtype=float)          # (N,)
        self._stds_log_lam  = np.sqrt(np.asarray(vars_log_lam, dtype=float))  # (N,)
        self._means_fp      = np.asarray(means_fp, dtype=float)                # (N,)
        self._stds_fp       = np.sqrt(np.asarray(vars_fp, dtype=float))        # (N,)
        self._M             = int(M)

        N = len(self._means_log_lam)
        if not all(len(a) == N for a in [
            self._stds_log_lam, self._means_fp, self._stds_fp
        ]):
            raise ValueError("All input arrays must have the same length N.")

    # ------------------------------------------------------------------

    def log_likelihood(self, parameters=None):
        """
        Evaluate the total log-likelihood across all N events.

        Reads alpha, beta, gamma from self.parameters, which bilby populates
        at each sampler iteration.

        Returns
        -------
        float
            Sum of per-event log-likelihoods.
        """
        alpha = parameters["alpha"]
        beta  = parameters["beta"]
        # gamma = parameters["gamma"]
        M     = self._M

        # Draw log_lam samples: shape (N, M)
        # Centred on injected means with inspiral posterior widths.
        log_lam_samples = np.random.normal(
            loc   = self._means_log_lam[:, None],  # (N, 1) -> broadcasts to (N, M)
            scale = self._stds_log_lam[:, None],
            size  = (len(self._means_log_lam), M),
        )

        # Exponentiate to get lam_t: shape (N, M)
        lam_t_samples = np.exp(log_lam_samples)

        # Map to f_peak via empirical relation: shape (N, M)
        # a          = 10.0 ** alpha
        # b          = 10.0 ** beta
        # fp_samples = a * lam_t_samples ** 2 - b * lam_t_samples + gamma

        # linear modeL: y = -10^\alpha * x + beta
        fp_samples = - (10 ** alpha) * lam_t_samples + beta


        # Inspiral log-likelihood factor: log N(log_lam^(k) | mu_i, sigma_i^2)
        # shape (N, M)
        log_w_insp = norm.logpdf(
            log_lam_samples,
            loc   = self._means_log_lam[:, None],
            scale = self._stds_log_lam[:, None],
        )

        # Post-merger log-likelihood factor: log N(f_peak^(k) | mu_i, sigma_i^2)
        # shape (N, M)
        log_w_pm = norm.logpdf(
            fp_samples,
            loc   = self._means_fp[:, None],
            scale = self._stds_fp[:, None],
        )

        # Combined log weight: shape (N, M)
        # log_w = log_w_pm
        log_w = log_w_insp + log_w_pm

        # Per-event MC log-likelihood: logsumexp over M samples, subtract log M
        # shape (N,)
        log_like_per_event = logsumexp(log_w, axis=1) - np.log(M)

        return float(np.sum(log_like_per_event))

    def noise_log_likelihood(self):
        return np.nan