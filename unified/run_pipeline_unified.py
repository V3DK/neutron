"""
Grand unified BNS inference pipeline
=====================================
Stages
------
Pre-merger arm:
  pre:prior_kde     – rejection-sample viable EOS 5-tuples, fit KDE
                      → viable_gamma_samples.npy, bounds.npy
  pre:event_level   – Fisher-matrix runs (inspiral waveform)
                      → event_data_pre.npz, M_tot_draws.npy
  pre:hierarchical  – emcee on EOSHyperparameterLikelihood
                      → outdir_pre/eos_hierarchical_result.pkl
  pre:ppd           – posterior predictive λ(m) plot with 90% CI
                      → ppd_pre.png

Post-merger arm:
  post:event_level  – Fisher-matrix runs (post-merger waveform)
                      → event_data_post.npz
                      [requires M_tot_draws.npy from pre:event_level]
  post:hierarchical – emcee on EMRelationLikelihood (converse likelihood)
                      → outdir_post/unif_hierarchical_result.json
  post:ppd          – posterior predictive f_peak(Λ̃) plot
                      → ppd_post.png

Multi-run comparison:
  multi:ppd         – overlay pre-merger PPD envelopes from multiple run
                      directories onto one plot, and plot ΔΛ₉₀ vs N
                      → multi_ppd.png, delta_lambda_vs_N.png
                        (both written to --multi-out-dir)
  multi:post_ppd    – overlay post-merger f_peak(Λ̃) PPD envelopes from
                      multiple run directories onto one plot, and plot
                      minimum detectable Δn vs N with DD2F-SF reference lines
                      → multi_post_ppd.png, delta_n_vs_N.png
                        (both written to --multi-out-dir)


  multi:ppd requires completed pre:hierarchical in each --run-dirs entry.

Mass conventions
----------------
  Pre-merger  : component mass m ∈ [MMIN / 2, MMAX / 2]  (M☉)
                equal-mass: m1 = m2 = M_tot / 2
  Post-merger : total binary mass M_tot ∈ [MMIN, MMAX]  (M☉)
                same M_tot draws as pre-merger; pm.get_params takes M_tot.

Storage conventions (all under RUN_DIR)
----------------------------------------
  viable_gamma_samples.npy          – (n_kde_samples, 5)
  bounds.npy                        – (5, 2)
  M_tot_draws.npy                   – (n_events,)      pre-merger mass draws (z≤2)
  M_tot_draws_post.npy              – (n_events,)      post-merger mass draws (z≤0.2)
  event_data_pre.npz                – keys: covs (N,2,2), means (N,2)
  event_data_pre_post.npz           – keys: covs (N,2,2), means (N,2)  [z≤0.2 events]
  event_data_post.npz               – keys: covs (N,), means (N,)
  outdir_pre/eos_hierarchical_result.pkl
  outdir_post/unif_hierarchical_result.json
  ppd_pre.png
  ppd_post.png

Usage
-----
  python run_pipeline.py [--run-dir PATH] [--stages STAGE [STAGE ...]]
                         [--n-events N] [--n-kde-samples N]
                         [--nwalkers-pre N]  [--nsteps-pre N]
                         [--nwalkers-post N] [--nsteps-post N]
                         [--npool N] [--mc-samples M]
                         [--n-ppd-samples N] [--seed N]
                         [--force]
                         [--run-dirs PATH [PATH ...]]
                         [--run-labels LABEL [LABEL ...]]
                         [--multi-out-dir PATH]

  --stages  subset of:
              pre:prior_kde  pre:event_level  pre:hierarchical  pre:ppd
              post:event_level  post:hierarchical  post:ppd
              multi:ppd multi:post_ppd
            (default: all single-run stages, in dependency order;
             multi:ppd must be requested explicitly)
  --force   re-run every requested stage even if output exists

  For multi:ppd:
  --run-dirs    list of run directories, one per N value
  --run-labels  corresponding legend labels (e.g. "N=50" "N=200" "N=500")
  --multi-out-dir  directory for multi_ppd.png (default: current dir)
"""

import argparse
import os
import sys

import bilby
import bilby.gw.conversion as conv
import numpy as np
from bilby.core.prior.joint import JointPrior
from bilby.core.utils import random
from bilby.gw.conversion import (
    lambda_from_mass_and_family,
    polytrope_or_causal_params_to_lambda_1_lambda_2,
    solar_mass,
)
from bilby.gw.utils import (
    lalsim_SimNeutronStarFamMinimumMass,
    lalsim_SimNeutronStarMaximumMass,
)
from gwbench import (
    M_of_Mc_eta,
    Network,
    f_isco_Msolar,
    injections_CBC_params_redshift,
)
from scipy.stats import gaussian_kde

import lal
lal.swig_redirect_standard_output_error(False)
import lalsimulation as lalsim

# ── local library modules (must live alongside this script) ──────────────────
from converse_likelihood_pre  import EOSHyperparameterLikelihood, _build_family
from converse_likelihood_post import EMRelationLikelihood
from KDEPrior import KDEJointDist
import pm_waveform_np as pm

import matplotlib 
matplotlib.rcParams.update({
    "text.usetex":        True,
    "font.family":        "serif",
    "font.serif":         ["Computer Modern Roman"],
})
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Physical / injection constants
# ---------------------------------------------------------------------------
EOS = "MPA1"

# Total binary mass bounds [M☉]; component mass m = M_tot / 2 for equal-mass systems
MMIN = 2.5
MMAX = 3.0

MIN_MAX_MASS = 2.0   # EOS viability floor on M_max [M☉]

# Shared injection cosmology / spin
COSMO_DICT_PRE  = {"zmin": 0, "zmax": 2.0, "sampler": "uniform_comoving_volume_inversion"}
COSMO_DICT_POST = {"zmin": 0, "zmax": 0.2, "sampler": "uniform_comoving_volume_inversion"}
SPIN_DICT  = {"dim": 1, "geom": "cartesian", "chi_lo": 0, "chi_hi": 0}

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------
# pm_waveform_np.py must be resolved as an absolute path for gwbench
_HERE = os.path.dirname(os.path.abspath(__file__))
PM_WAVEFORM_PATH = os.path.join(_HERE, "pm_waveform_np.py")


def _path(run_dir: str, *parts: str) -> str:
    return os.path.join(run_dir, *parts)


# ===========================================================================
# EOS viability helper (shared by pre-merger stages)
# ===========================================================================
def is_viable(g0: float, g1: float, g2: float, lp1: float, lp2: float) -> bool:
    """True iff EOS is physical and supports M_max ≥ MIN_MAX_MASS."""
    family = _build_family(g0, lp1, g1, lp2, g2)
    if family is None:
        return False
    return lalsim_SimNeutronStarMaximumMass(family) / solar_mass >= MIN_MAX_MASS


# ===========================================================================
# Shared injection draws
# ===========================================================================
def _make_injection_draws(n_events: int, seed: int, cosmo_dict: dict):
    """
    Return (injections_data, M_tot_draws).

    injections_data is the gwbench tuple; indices 8-13 carry extrinsic params.
    M_tot_draws is shape (n_events,), drawn from U[MMIN, MMAX].

    Parameters
    ----------
    cosmo_dict : cosmological / redshift distribution dict passed to
                 injections_CBC_params_redshift.  Pass COSMO_DICT_PRE for the
                 pre-merger arm (zmax = 2) or COSMO_DICT_POST for the
                 post-merger arm (zmax = 0.2).
    """
    mass_dict = {"dist": "uniform", "mmin": MMIN, "mmax": MMAX}
    injections_data = injections_CBC_params_redshift(
        cosmo_dict, mass_dict, SPIN_DICT,
        redshifted=1, num_injs=n_events, seed=seed, file_path=None,
    )
    rng = np.random.default_rng(seed=seed)
    M_tot_draws = rng.uniform(MMIN, MMAX, size=n_events)
    return injections_data, M_tot_draws


# ===========================================================================
# Stage pre:1 – prior KDE
# ===========================================================================
def _check_candidate(args):
    g0, g1, g2, lp1, lp2 = args
    return (g0, g1, g2, lp1, lp2) if is_viable(g0, g1, g2, lp1, lp2) else None


def stage_pre_prior_kde(run_dir: str, n_samples: int, n_pool: int = 1) -> tuple:
    """
    Rejection-sample n_samples viable EOS 5-tuples, fit KDE, derive bounds.

    Saves
    -----
    viable_gamma_samples.npy  shape (n_samples, 5)
    bounds.npy                shape (5, 2)
    """
    import time
    from multiprocessing import Pool

    samples_path = _path(run_dir, "viable_gamma_samples.npy")
    bounds_path  = _path(run_dir, "bounds.npy")

    print(f"\n[pre:prior_kde] Rejection-sampling {n_samples} viable EOS 5-tuples "
          f"using {n_pool} process(es) ...")

    viable        = []
    n_total       = 0
    BATCH         = n_pool * 200
    estimate_done = {100: False, 1000: False}
    t_start       = time.perf_counter()

    def _print_estimate(n_viable, n_tried, label):
        elapsed   = time.perf_counter() - t_start
        rate      = n_viable / elapsed if elapsed > 0 else float("inf")
        remaining = (n_samples - n_viable) / rate if rate > 0 else float("inf")
        accept    = n_viable / n_tried
        print(
            f"[pre:prior_kde]  {label}: {n_viable} accepted / {n_tried} tried  "
            f"(acceptance {accept:.2%})  |  elapsed {elapsed:.1f}s  |  "
            f"estimated remaining {remaining/60:.1f} min"
        )

    with Pool(processes=n_pool) as pool:
        while len(viable) < n_samples:
            candidates = [
                (
                    float(np.random.uniform(1.0, 5.0)),
                    float(np.random.uniform(1.0, 5.0)),
                    float(np.random.uniform(1.0, 5.0)),
                    float(np.random.uniform(33.0, 36.0)),
                    float(np.random.uniform(33.0, 36.0)),
                )
                for _ in range(BATCH)
            ]
            n_total += BATCH
            results  = pool.map(_check_candidate, candidates)
            for r in results:
                if r is not None and len(viable) < n_samples:
                    viable.append(r)

            n_acc = len(viable)
            for milestone, done in list(estimate_done.items()):
                if not done and n_acc >= milestone:
                    _print_estimate(n_acc, n_total, f"After {milestone} accepted")
                    estimate_done[milestone] = True

            print(
                f"[pre:prior_kde]  {n_acc}/{n_samples} accepted "
                f"({n_acc/n_samples:.1%})  |  tried {n_total}",
                end="\r", flush=True,
            )

    print()
    viable = np.array(viable[:n_samples])
    elapsed_total = time.perf_counter() - t_start
    print(f"[pre:prior_kde] Done in {elapsed_total/60:.1f} min  |  "
          f"acceptance: {n_samples / n_total:.3%}")

    np.save(samples_path, viable)
    bounds = np.column_stack([viable.min(axis=0), viable.max(axis=0)])
    np.save(bounds_path, bounds)
    print(f"[pre:prior_kde] Saved → {samples_path}, {bounds_path}")

    kde = gaussian_kde(viable.T, bw_method="scott")
    return kde, bounds


def _load_pre_prior_kde(run_dir: str) -> tuple:
    viable = np.load(_path(run_dir, "viable_gamma_samples.npy"))
    bounds = np.load(_path(run_dir, "bounds.npy"))
    kde    = gaussian_kde(viable.T, bw_method="scott")
    return kde, bounds


# ===========================================================================
# Stage pre:2 – event-level (inspiral Fisher)
# ===========================================================================
def stage_pre_event_level(run_dir: str, n_events: int, seed: int, n_cores: int) -> tuple:
    """
    Fisher-matrix runs for the inspiral waveform.  Uses COSMO_DICT_PRE (zmax=2).

    Saves
    -----
    event_data_pre.npz   keys: covs (N,2,2), means (N,2)  [Mc, log_lam_t]
    M_tot_draws.npy      shape (N,)                        [pre-merger mass draws]
    """
    event_path  = _path(run_dir, "event_data_pre.npz")
    mtot_path   = _path(run_dir, "M_tot_draws.npy")

    print(f"\n[pre:event_level] Running {n_events} Fisher-matrix events ...")

    injections_data, M_tot_draws = _make_injection_draws(n_events, seed, COSMO_DICT_PRE)

    eos_lalsim = lalsim.SimNeutronStarEOSByName(EOS)
    fam        = lalsim.CreateSimNeutronStarFamily(eos_lalsim)

    pre_wf_model_name    = "lal_bns"
    pre_wf_other_var_dic = {"approximant": "IMRPhenomD_NRTidalv2"}
    deriv_symbs_string     = "Mc DL tc phic iota lam_t ra dec psi"
    ana_deriv_symbs_string = "DL tc phic ra dec psi"
    conv_cos = ("dec", "iota")
    conv_log = ("DL", "lam_t")

    covs  = []
    means = []

    for inj_id in range(n_events):
        M_tot = M_tot_draws[inj_id]
        m     = M_tot / 2   # equal-mass

        lam_1 = lambda_from_mass_and_family(m, fam)
        lam_2 = lambda_from_mass_and_family(m, fam)

        inj_params = {
            "Mc"          : conv.component_masses_to_chirp_mass(m, m),
            "eta"         : conv.component_masses_to_symmetric_mass_ratio(m, m),
            "chi1x"       : injections_data[2][inj_id],
            "chi1y"       : injections_data[3][inj_id],
            "chi1z"       : injections_data[4][inj_id],
            "chi2x"       : injections_data[5][inj_id],
            "chi2y"       : injections_data[6][inj_id],
            "chi2z"       : injections_data[7][inj_id],
            "DL"          : injections_data[8][inj_id],
            "tc"          : 0.0,
            "phic"        : 0.0,
            "iota"        : injections_data[9][inj_id],
            "ra"          : injections_data[10][inj_id],
            "dec"         : injections_data[11][inj_id],
            "psi"         : injections_data[12][inj_id],
            "z"           : injections_data[13][inj_id],
            "lam_t"       : conv.lambda_1_lambda_2_to_lambda_tilde(lam_1, lam_2, m, m),
            "delta_lam_t" : conv.lambda_1_lambda_2_to_delta_lambda_tilde(lam_1, lam_2, m, m),
        }

        f_lo = 1.0
        f_hi = f_isco_Msolar(M_of_Mc_eta(inj_params["Mc"], inj_params["eta"]))
        df   = 2.0 ** -4
        f    = np.arange(f_lo, f_hi + df, df)
        # print(f"[pre:event_level | inj {inj_id:02d}] M_tot={M_tot:.4f}  f_hi={f_hi:.2f} Hz")
        print(f"[pre:event_level | inj {inj_id:02d}] M_tot={M_tot:.4f}  z={inj_params["z"]:.2f}")

        net = Network("E", logger_name="PRE", logger_level="WARNING")
        net.set_net_vars(
            wf_model_name=pre_wf_model_name, wf_other_var_dic=pre_wf_other_var_dic,
            f=f, inj_params=inj_params,
            deriv_symbs_string=deriv_symbs_string,
            conv_cos=conv_cos, conv_log=conv_log, use_rot=1,
            ana_deriv_symbs_string=ana_deriv_symbs_string,
        )
        net.calc_errors(
            only_net=1, derivs="num",
            step=1e-6, method="central", order=2,
            gen_derivs=None, num_cores=n_cores,
        )

        keep = ["Mc", "log_lam_t"]
        idx  = [list(net.deriv_variables).index(p) for p in keep]
        covs.append(net.cov[np.ix_(idx, idx)])
        means.append(np.array([
            net.inj_params["Mc"],
            np.log(net.inj_params["lam_t"]),
        ]))

    covs  = np.array(covs)
    means = np.array(means)
    np.savez(event_path, covs=covs, means=means)
    np.save(mtot_path, M_tot_draws)
    print(f"\n[pre:event_level] Saved → {event_path}, {mtot_path}  ({len(means)} events)")
    return covs, means


def _load_pre_event_data(run_dir: str) -> tuple:
    data = np.load(_path(run_dir, "event_data_pre.npz"))
    return data["covs"].astype(np.float64), data["means"]


def _load_pre_post_event_data(run_dir: str) -> tuple:
    """Load the z≤0.2 inspiral Fisher data produced by post:event_level."""
    data = np.load(_path(run_dir, "event_data_pre_post.npz"))
    return data["covs"].astype(np.float64), data["means"]


# ===========================================================================
# Stage pre:3 – hierarchical (EOS)
# ===========================================================================
def stage_pre_hierarchical(
    run_dir:   str,
    covs:      np.ndarray,
    means:     np.ndarray,
    kde:       gaussian_kde,
    bounds:    np.ndarray,
    n_walkers: int = 64,
    n_steps:   int = 5000,
    n_pool:    int = 10,
):
    """
    emcee hierarchical inference on EOS parameters.
    bilby writes result to RUN_DIR/outdir_pre/eos_hierarchical_result.pkl.
    """
    outdir = _path(run_dir, "outdir_pre")
    bilby.utils.check_directory_exists_and_if_not_mkdir(outdir)

    print(f"\n[pre:hierarchical] Building likelihood and priors ...")

    bounds_dict = {
        "param1":   (bounds[0, 0], bounds[0, 1]),
        "param2":   (bounds[1, 0], bounds[1, 1]),
        "param3":   (bounds[2, 0], bounds[2, 1]),
        "log10_p1": (bounds[3, 0], bounds[3, 1]),
        "log10_p2": (bounds[4, 0], bounds[4, 1]),
    }
    dist = KDEJointDist(
        names=["param1", "param2", "param3", "log10_p1", "log10_p2"],
        kde=kde,
        bounds=bounds_dict,
    )
    priors = {
        "param1":   JointPrior(dist=dist, name="param1",   latex_label=r"$\Gamma_0$"),
        "param2":   JointPrior(dist=dist, name="param2",   latex_label=r"$\Gamma_1$"),
        "param3":   JointPrior(dist=dist, name="param3",   latex_label=r"$\Gamma_2$"),
        "log10_p1": JointPrior(dist=dist, name="log10_p1", latex_label=r"$\log_{10}p_1$"),
        "log10_p2": JointPrior(dist=dist, name="log10_p2", latex_label=r"$\log_{10}p_2$"),
    }

    hp_likelihood = EOSHyperparameterLikelihood(
        cov_matrices=covs, means=means,
        mass_min=MMIN / 2, mass_max=MMAX / 2,
        param_bounds=bounds_dict,
    )

    lo = bounds[:, 0]
    hi = bounds[:, 1]
    init_pos = []
    while len(init_pos) < n_walkers:
        batch     = kde.resample(n_walkers * 10).T
        in_bounds = np.all((batch >= lo) & (batch <= hi), axis=1)
        for candidate in batch[in_bounds]:
            if len(init_pos) >= n_walkers:
                break
            if is_viable(candidate[0], candidate[1], candidate[2],
                         candidate[3], candidate[4]):
                init_pos.append(candidate)
    init_pos = np.array(init_pos)

    print(f"[pre:hierarchical] Running emcee: {n_walkers} walkers × {n_steps} steps ...")
    result = bilby.run_sampler(
        likelihood=hp_likelihood,
        priors=priors,
        sampler="emcee",
        nwalkers=n_walkers,
        nsteps=n_steps,
        pos0=init_pos,
        outdir=outdir,
        label="eos_hierarchical",
        npool=n_pool,
    )
    result.plot_corner()
    print(f"[pre:hierarchical] Result saved → {outdir}/")
    return result


def _load_pre_hierarchical(run_dir: str):
    pkl = _path(run_dir, "outdir_pre", "eos_hierarchical_result.pkl")
    return bilby.core.result.read_in_result(filename=pkl)


# ===========================================================================
# PPD computation helper (shared by pre:ppd and multi:ppd)
# ===========================================================================
def _compute_pre_ppd_arrays(
    result,
    n_ppd_samples: int,
    n_mass_points: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Draw EOS posterior samples and evaluate Λ(m) on a mass grid.

    Returns
    -------
    m_plot      : (n_mass_points,)          component mass grid [M☉]
    lam_ppd     : (n_valid_draws, n_mass_points)  Λ values; rows with any NaN
                  are dropped before percentile computation — callers receive
                  the full NaN-free array.
    lam_true    : (n_valid_true,)           MPA1 truth on the same grid
    m_true      : (n_valid_true,)           mass grid trimmed to MPA1 support
    """
    m_plot    = np.linspace(MMIN / 2, MMAX / 2, n_mass_points)
    posterior = result.posterior
    draws     = posterior.sample(n_ppd_samples, replace=True)[
        ["param1", "param2", "param3", "log10_p1", "log10_p2"]
    ]

    lam_rows = []
    for _, row in draws.iterrows():
        family = _build_family(row.param1, row.log10_p1, row.param2, row.log10_p2, row.param3)
        if family is None:
            continue
        m_min = lalsim_SimNeutronStarFamMinimumMass(family) / solar_mass
        m_max = lalsim_SimNeutronStarMaximumMass(family) / solar_mass
        # Fill Λ where the EOS has support; NaN outside
        lam_row = np.full(n_mass_points, np.nan)
        valid   = (m_plot >= m_min) & (m_plot <= m_max)
        if not valid.any():
            continue
        lam_row[valid] = np.array(
            [lambda_from_mass_and_family(m, family) for m in m_plot[valid]]
        )
        lam_rows.append(lam_row)

    lam_ppd = np.array(lam_rows)   # (n_draws, n_mass_points), may contain NaN

    # MPA1 truth
    eos_true    = lalsim.SimNeutronStarEOSByName(EOS)
    family_true = lalsim.CreateSimNeutronStarFamily(eos_true)
    m_min_t     = lalsim_SimNeutronStarFamMinimumMass(family_true) / solar_mass
    m_max_t     = lalsim_SimNeutronStarMaximumMass(family_true) / solar_mass
    mask_t      = (m_plot >= m_min_t) & (m_plot <= m_max_t)
    m_true      = m_plot[mask_t]
    lam_true    = np.array([lambda_from_mass_and_family(m, family_true) for m in m_true])

    return m_plot, lam_ppd, lam_true, m_true


def _ppd_percentiles(
    lam_ppd: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute NaN-aware 5th / 50th / 95th percentiles column-wise.

    Columns with fewer than 2 non-NaN draws are set to NaN in all outputs
    so the plot does not extrapolate into unsupported mass regions.

    Returns
    -------
    lam_lo, lam_med, lam_hi : each (n_mass_points,)
    """
    with np.errstate(all="ignore"):
        n_valid  = np.sum(~np.isnan(lam_ppd), axis=0)
        lam_lo   = np.where(n_valid >= 2, np.nanpercentile(lam_ppd,  5, axis=0), np.nan)
        lam_med  = np.where(n_valid >= 2, np.nanpercentile(lam_ppd, 50, axis=0), np.nan)
        lam_hi   = np.where(n_valid >= 2, np.nanpercentile(lam_ppd, 95, axis=0), np.nan)
    return lam_lo, lam_med, lam_hi


# ===========================================================================
# Stage pre:4 – PPD plot (λ vs m) with 90% credible envelope
# ===========================================================================
def stage_pre_ppd(run_dir: str, result, n_ppd_samples: int = 500, n_mass_points: int = 200):
    """
    Draw EOS posterior samples, evaluate Λ(m), plot 90% CL envelope and
    median, save ppd_pre.png.

    The envelope is computed column-wise (per mass point) using NaN-aware
    percentiles so that draws whose EOS support does not reach a given mass
    are excluded at that point rather than extrapolated.
    """
    plot_path = _path(run_dir, "ppd_pre.png")
    print(f"\n[pre:ppd] Drawing {n_ppd_samples} posterior samples ...")

    m_plot, lam_ppd, lam_true, m_true = _compute_pre_ppd_arrays(
        result, n_ppd_samples, n_mass_points
    )
    lam_lo, lam_med, lam_hi = _ppd_percentiles(lam_ppd)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.fill_between(m_plot, lam_lo, lam_hi,
                    alpha=0.35, color="steelblue", label="PPD 90% CI")
    ax.plot(m_plot, lam_med,
            color="steelblue", lw=2, label="PPD median")
    ax.plot(m_true, lam_true,
            color="crimson", lw=2, label=f"Injection truth ({EOS})")

    ax.set_xlabel(r"$m\ [M_\odot]$", fontsize=13)
    ax.set_ylabel(r"$\Lambda$", fontsize=13)
    ax.set_xlim(MMIN / 2, MMAX / 2)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=11)
    ax.set_title(f"Inspiral PPD: {n_ppd_samples} posterior draws, 90% CI", fontsize=13)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"[pre:ppd] Saved → {plot_path}")


# ===========================================================================
# Stage multi:ppd – overlay pre-merger PPDs from multiple run directories
# ===========================================================================

# Colour cycle for successive N values (colourblind-friendly)
_MULTI_COLORS = [
    "#1f77b4",  # blue
    "#ff7f0e",  # orange
    "#2ca02c",  # green
    "#9467bd",  # purple
    "#8c564b",  # brown
]


def stage_multi_ppd(
    run_dirs:      list[str],
    labels:        list[str],
    out_dir:       str,
    n_ppd_samples: int = 500,
    n_mass_points: int = 200,
    force:         bool = False,
):
    """
    Load the pre:hierarchical result from each run directory, compute the
    90% CI PPD envelope for each, and produce two figures:

      1. multi_ppd.png            – overlaid Λ(m) envelopes for all N
      2. delta_lambda_vs_N.png    – 90% CI width ΔΛ₉₀ at m = 1.35 M☉ vs N

    N is read from M_tot_draws.npy in each run directory (authoritative source),
    not from labels or CLI arguments.  Run directories are processed in the
    order supplied but the width figure is plotted sorted by ascending N.

    Parameters
    ----------
    run_dirs      : pipeline run directories; each must contain
                    outdir_pre/eos_hierarchical_result.pkl and
                    M_tot_draws.npy
    labels        : legend label for each run directory in the PPD overlay
    out_dir       : directory where both figures are written
    n_ppd_samples : posterior draws per run directory
    n_mass_points : mass grid resolution
    force         : overwrite existing outputs

    Saves
    -----
    <out_dir>/multi_ppd.png
    <out_dir>/delta_lambda_vs_N.png
    """
    # Reference mass for ΔΛ₉₀ scalar [M☉]
    M_REF = 1.35

    ppd_path   = os.path.join(out_dir, "multi_ppd.png")
    width_path = os.path.join(out_dir, "delta_lambda_vs_N.png")

    need_ppd   = force or not os.path.exists(ppd_path)
    need_width = force or not os.path.exists(width_path)

    if not need_ppd and not need_width:
        print(
            f"\n[multi:ppd] Both outputs exist (use --force to regenerate):\n"
            f"  {ppd_path}\n  {width_path}"
        )
        return

    if len(run_dirs) != len(labels):
        sys.exit(
            f"ERROR: [multi:ppd] --run-dirs ({len(run_dirs)} entries) and "
            f"--run-labels ({len(labels)} entries) must have the same length."
        )

    os.makedirs(out_dir, exist_ok=True)

    # ── mass grid and reference column index ─────────────────────────────────
    m_plot  = np.linspace(MMIN / 2, MMAX / 2, n_mass_points)
    if M_REF < m_plot[0] or M_REF > m_plot[-1]:
        sys.exit(
            f"ERROR: [multi:ppd] M_REF = {M_REF} M☉ lies outside the mass grid "
            f"[{m_plot[0]:.3f}, {m_plot[-1]:.3f}] M☉.  Adjust MMIN/MMAX or M_REF."
        )
    ref_idx = int(np.argmin(np.abs(m_plot - M_REF)))
    print(f"\n[multi:ppd] ΔΛ₉₀ evaluated at m = {m_plot[ref_idx]:.4f} M☉ "
          f"(grid index {ref_idx}, requested {M_REF} M☉)")

    # ── MPA1 truth (computed once) ────────────────────────────────────────────
    eos_true    = lalsim.SimNeutronStarEOSByName(EOS)
    family_true = lalsim.CreateSimNeutronStarFamily(eos_true)
    m_min_t     = lalsim_SimNeutronStarFamMinimumMass(family_true) / solar_mass
    m_max_t     = lalsim_SimNeutronStarMaximumMass(family_true) / solar_mass
    mask_t      = (m_plot >= m_min_t) & (m_plot <= m_max_t)
    m_true      = m_plot[mask_t]
    lam_true    = np.array([lambda_from_mass_and_family(m, family_true) for m in m_true])

    # ── per-run-dir loop ──────────────────────────────────────────────────────
    # Collect (N, ΔΛ₉₀, color, label) for the width figure
    width_records: list[tuple[int, float, str, str]] = []

    fig_ppd, ax_ppd = plt.subplots(figsize=(9, 6))

    for i, (run_dir, label) in enumerate(zip(run_dirs, labels)):
        # ── prerequisite checks ───────────────────────────────────────────────
        pkl_path  = _path(run_dir, "outdir_pre", "eos_hierarchical_result.pkl")
        mtot_path = _path(run_dir, "M_tot_draws.npy")

        for fpath, stage in [(pkl_path, "pre:hierarchical"), (mtot_path, "pre:event_level")]:
            if not os.path.exists(fpath):
                sys.exit(
                    f"ERROR: [multi:ppd] Required file not found: {fpath}. "
                    f"Run {stage} for '{run_dir}' first."
                )

        # ── read N from disk ──────────────────────────────────────────────────
        N = len(np.load(mtot_path))

        print(f"\n[multi:ppd] '{label}' | N = {N} | loading {pkl_path} ...")
        result = bilby.core.result.read_in_result(filename=pkl_path)

        print(f"[multi:ppd] '{label}' | computing PPD ({n_ppd_samples} draws) ...")
        _, lam_ppd, _, _ = _compute_pre_ppd_arrays(result, n_ppd_samples, n_mass_points)
        lam_lo, lam_med, lam_hi = _ppd_percentiles(lam_ppd)

        color = _MULTI_COLORS[i % len(_MULTI_COLORS)]

        # ── PPD overlay ───────────────────────────────────────────────────────
        ax_ppd.fill_between(m_plot, lam_lo, lam_hi, alpha=0.25, color=color, label=label)
        # ax_ppd.plot(m_plot, lam_med, color=color, lw=2, label=label)

        # ── ΔΛ₉₀ at M_REF ────────────────────────────────────────────────────
        lo_val = lam_lo[ref_idx]
        hi_val = lam_hi[ref_idx]
        if np.isnan(lo_val) or np.isnan(hi_val):
            print(
                f"[multi:ppd] WARNING: envelope is NaN at m = {M_REF} M☉ for '{label}'. "
                "This run dir will be omitted from the width figure."
            )
        else:
            delta_lam = hi_val - lo_val
            width_records.append((N, delta_lam, color, label))
            print(f"[multi:ppd] '{label}' | ΔΛ₉₀({M_REF} M☉) = {delta_lam:.1f}")

    # ── finalise PPD overlay figure ───────────────────────────────────────────
    ax_ppd.plot(m_true, lam_true,
                color="crimson", lw=1.5, ls="--", label=f"Injection truth ({EOS})")
    ax_ppd.set_xlabel(r"$m\ [M_\odot]$", fontsize=13)
    ax_ppd.set_ylabel(r"$\Lambda$", fontsize=13)
    ax_ppd.set_xlim(MMIN / 2, MMAX / 2)
    ax_ppd.set_ylim(bottom=0)
    ax_ppd.legend(fontsize=11)
    # ax_ppd.set_title("Inspiral PPD: 90% CL envelopes", fontsize=13)
    fig_ppd.tight_layout()

    if need_ppd:
        fig_ppd.savefig(ppd_path, dpi=150)
        print(f"\n[multi:ppd] Saved → {ppd_path}")
    else:
        print(f"\n[multi:ppd] Skipping {ppd_path} (exists; use --force to overwrite)")
    plt.close(fig_ppd)

    # ── ΔΛ₉₀ vs N figure ─────────────────────────────────────────────────────
    if not width_records:
        print("[multi:ppd] WARNING: no valid width records; skipping delta_lambda_vs_N.png")
        return

    if need_width:
        # Sort ascending by N so x-axis is monotone regardless of --run-dirs order
        width_records.sort(key=lambda r: r[0])
        ns     = [r[0] for r in width_records]
        widths = [r[1] for r in width_records]
        colors = [r[2] for r in width_records]

        fig_w, ax_w = plt.subplots(figsize=(8, 6))
        ax_w.plot(ns, widths, color="dimgray", lw=3, zorder=1)
        for n_val, w_val, c in zip(ns, widths, colors):
            ax_w.scatter(n_val, w_val, color=c, s=129, zorder=2)

        ax_w.axhline(y=558, color="dimgray", lw=3.0, ls="--", zorder=1, label = "GW170817")
        ax_w.set_xlabel(r"$N$", fontsize=26)
        ax_w.set_ylabel(
            r"$\Delta\Lambda_{90}$", fontsize=26)
        ax_w.tick_params(axis="both", labelsize=22)
        ax_w.set_xlim(left=0)
        ax_w.set_ylim(bottom=0)
        ax_w.legend(fontsize = 26, loc = "upper right")
        fig_w.tight_layout()
        fig_w.savefig(width_path, dpi=150)
        plt.close(fig_w)
        print(f"[multi:ppd] Saved → {width_path}")
    else:
        print(f"[multi:ppd] Skipping {width_path} (exists; use --force to overwrite)")


# ===========================================================================
# Stage post:1 – event-level (post-merger Fisher)
# ===========================================================================
def stage_post_event_level(run_dir: str, n_events: int, seed: int, n_cores: int) -> tuple:
    """
    Fisher-matrix runs for both the inspiral and post-merger waveforms, using
    COSMO_DICT_POST (zmax=0.2).  Both Fisher calculations operate on the same
    set of injection draws so that the pre- and post-merger likelihoods in
    post:hierarchical are anchored to identical events.

    Does NOT depend on pre:event_level outputs; it generates its own
    independent mass draws (M_tot_draws_post.npy) from the z≤0.2 population.

    Saves
    -----
    M_tot_draws_post.npy     shape (N,)        post-merger mass draws (z≤0.2)
    event_data_pre_post.npz  keys: covs (N,2,2), means (N,2)
                             inspiral Fisher results for the z≤0.2 events
    event_data_post.npz      keys: covs (N,), means (N,)
                             post-merger Fisher results for the z≤0.2 events
    """
    mtot_post_path   = _path(run_dir, "M_tot_draws_post.npy")
    pre_post_path    = _path(run_dir, "event_data_pre_post.npz")
    post_event_path  = _path(run_dir, "event_data_post.npz")

    print(f"\n[post:event_level] Running {n_events} events (zmax=0.2) — "
          f"inspiral + post-merger Fisher ...")

    injections_data, M_tot_draws = _make_injection_draws(n_events, seed, COSMO_DICT_POST)

    np.save(mtot_post_path, M_tot_draws)
    print(f"[post:event_level] Saved → {mtot_post_path}")

    # ── shared waveform / derivative settings ────────────────────────────────
    # Inspiral
    pre_wf_model_name    = "lal_bns"
    pre_wf_other_var_dic = {"approximant": "IMRPhenomD_NRTidalv2"}
    pre_deriv_symbs      = "Mc DL tc phic iota lam_t ra dec psi"
    pre_ana_deriv_symbs  = "DL tc phic ra dec psi"
    pre_conv_cos         = ("dec", "iota")
    pre_conv_log         = ("DL", "lam_t")

    # Post-merger
    post_f_lo = 1.0
    post_f_hi = 8000.0
    post_df   = 2.0 ** -4
    post_f    = np.arange(post_f_lo, post_f_hi + post_df, post_df)

    post_wf_model_name    = "pm_waveform"
    post_wf_other_var_dic = None
    post_user_waveform    = PM_WAVEFORM_PATH

    post_deriv_symbs = (
        "fpeak_ts_ zeta_drift_ t_star_ "
        "f_spiral_ f_2m0_ f_2p0_ "
        "A_peak_ A_spiral_ A_2m0_ A_2p0_ "
        "tau_peak_ tau_spiral_ tau_2m0_ tau_2p0_ "
        "phi_peak_ phi_spiral_ phi_2m0_ phi_2p0_ "
        "N_ "
        "DL tc phic iota"
    )
    post_ana_deriv_symbs = "DL tc phic"
    post_conv_cos        = ("dec", "iota")
    post_conv_log        = ("DL",)

    eos_lalsim = lalsim.SimNeutronStarEOSByName(EOS)
    fam        = lalsim.CreateSimNeutronStarFamily(eos_lalsim)

    pre_covs   = []
    pre_means  = []
    post_covs  = []
    post_means = []

    for inj_id in range(n_events):
        M_tot = M_tot_draws[inj_id]
        m     = M_tot / 2   # equal-mass

        print(f"[post:event_level | inj {inj_id:02d}] M_tot={M_tot:.4f}")

        # ── inspiral Fisher ───────────────────────────────────────────────────
        lam_1 = lambda_from_mass_and_family(m, fam)
        lam_2 = lambda_from_mass_and_family(m, fam)

        pre_inj_params = {
            "Mc"          : conv.component_masses_to_chirp_mass(m, m),
            "eta"         : conv.component_masses_to_symmetric_mass_ratio(m, m),
            "chi1x"       : injections_data[2][inj_id],
            "chi1y"       : injections_data[3][inj_id],
            "chi1z"       : injections_data[4][inj_id],
            "chi2x"       : injections_data[5][inj_id],
            "chi2y"       : injections_data[6][inj_id],
            "chi2z"       : injections_data[7][inj_id],
            "DL"          : injections_data[8][inj_id],
            "tc"          : 0.0,
            "phic"        : 0.0,
            "iota"        : injections_data[9][inj_id],
            "ra"          : injections_data[10][inj_id],
            "dec"         : injections_data[11][inj_id],
            "psi"         : injections_data[12][inj_id],
            "z"           : injections_data[13][inj_id],
            "lam_t"       : conv.lambda_1_lambda_2_to_lambda_tilde(lam_1, lam_2, m, m),
            "delta_lam_t" : conv.lambda_1_lambda_2_to_delta_lambda_tilde(lam_1, lam_2, m, m),
        }

        f_lo = 1.0
        f_hi = f_isco_Msolar(M_of_Mc_eta(pre_inj_params["Mc"], pre_inj_params["eta"]))
        df   = 2.0 ** -4
        f    = np.arange(f_lo, f_hi + df, df)

        net_pre = Network("E", logger_name="POST_PRE", logger_level="WARNING")
        net_pre.set_net_vars(
            wf_model_name=pre_wf_model_name, wf_other_var_dic=pre_wf_other_var_dic,
            f=f, inj_params=pre_inj_params,
            deriv_symbs_string=pre_deriv_symbs,
            conv_cos=pre_conv_cos, conv_log=pre_conv_log, use_rot=1,
            ana_deriv_symbs_string=pre_ana_deriv_symbs,
        )
        net_pre.calc_errors(
            only_net=1, derivs="num",
            step=1e-6, method="central", order=2,
            gen_derivs=None, num_cores=n_cores,
        )

        keep = ["Mc", "log_lam_t"]
        idx  = [list(net_pre.deriv_variables).index(p) for p in keep]
        pre_covs.append(net_pre.cov[np.ix_(idx, idx)])
        pre_means.append(np.array([
            net_pre.inj_params["Mc"],
            np.log(net_pre.inj_params["lam_t"]),
        ]))

        # ── post-merger Fisher ────────────────────────────────────────────────
        p = pm.get_params(M_tot)

        post_inj_params = {
            "fpeak_ts_"   : p["fpeak_ts"],
            "zeta_drift_" : p["zeta_drift"],
            "t_star_"     : p["t_star"],
            "f_spiral_"   : p["f_spiral"],
            "f_2m0_"      : p["f_2m0"],
            "f_2p0_"      : p["f_2p0"],
            "A_peak_"     : p["A_peak"],
            "A_spiral_"   : p["A_spiral"],
            "A_2m0_"      : p["A_2m0"],
            "A_2p0_"      : p["A_2p0"],
            "tau_peak_"   : p["tau_peak"],
            "tau_spiral_" : p["tau_spiral"],
            "tau_2m0_"    : p["tau_2m0"],
            "tau_2p0_"    : p["tau_2p0"],
            "phi_peak_"   : p["phi_peak"],
            "phi_spiral_" : p["phi_spiral"],
            "phi_2m0_"    : p["phi_2m0"],
            "phi_2p0_"    : p["phi_2p0"],
            "N_"          : p["N"],
            "s_"          : p["s"],
            "DL"          : injections_data[8][inj_id],
            "tc"          : 0.0,
            "phic"        : 0.0,
            "iota"        : injections_data[9][inj_id],
            "ra"          : injections_data[10][inj_id],
            "dec"         : injections_data[11][inj_id],
            "psi"         : injections_data[12][inj_id],
            "z"           : injections_data[13][inj_id],
            "Mc"          : None,
            "eta"         : None,
        }

        net_post = Network("E", logger_name="POST", logger_level="WARNING")
        net_post.set_net_vars(
            wf_model_name=post_wf_model_name,
            wf_other_var_dic=post_wf_other_var_dic,
            user_waveform=post_user_waveform,
            f=post_f,
            inj_params=post_inj_params,
            deriv_symbs_string=post_deriv_symbs,
            ana_deriv_symbs_string=post_ana_deriv_symbs,
            conv_cos=post_conv_cos,
            conv_log=post_conv_log,
            use_rot=0,
        )
        net_post.calc_errors(
            only_net=1, derivs="num",
            step=1e-6, method="central", order=2,
            gen_derivs=None, num_cores=n_cores,
        )

        keep    = "fpeak_ts_"
        idx     = net_post.deriv_variables.index(keep)
        post_covs.append(net_post.cov[idx, idx])
        post_means.append(net_post.inj_params[keep])

    pre_covs   = np.array(pre_covs)
    pre_means  = np.array(pre_means)
    post_covs  = np.array(post_covs)
    post_means = np.array(post_means)

    np.savez(pre_post_path,   covs=pre_covs,  means=pre_means)
    np.savez(post_event_path, covs=post_covs, means=post_means)
    print(
        f"\n[post:event_level] Saved → {pre_post_path}, {post_event_path}"
        f"  ({len(post_means)} events)"
    )
    return pre_covs, pre_means, post_covs, post_means


def _load_post_event_data(run_dir: str) -> tuple:
    """Load pre_post and post event data produced by post:event_level."""
    pre_post = np.load(_path(run_dir, "event_data_pre_post.npz"))
    post     = np.load(_path(run_dir, "event_data_post.npz"))
    return (
        pre_post["covs"].astype(np.float64), pre_post["means"],
        post["covs"].astype(np.float64),     post["means"],
    )


# ===========================================================================
# Stage post:2 – hierarchical (empirical relation)
# ===========================================================================
def stage_post_hierarchical(
    run_dir:          str,
    pre_post_covs:    np.ndarray,
    pre_post_means:   np.ndarray,
    post_covs:        np.ndarray,
    post_means:       np.ndarray,
    n_walkers:        int = 32,
    n_steps:          int = 5000,
    n_pool:           int = 10,
    mc_samples:       int = 10,
):
    """
    emcee hierarchical inference on empirical-relation parameters (α, β).
    bilby writes result to RUN_DIR/outdir_post/unif_hierarchical_result.json.

    Parameters
    ----------
    pre_post_covs, pre_post_means : inspiral Fisher results from the z≤0.2
        event population (event_data_pre_post.npz), produced by post:event_level.
        These supply the log_lam_t constraints that are paired with the
        post-merger f_peak measurements from the same events.
    post_covs, post_means : post-merger Fisher results (event_data_post.npz).
    """
    outdir = _path(run_dir, "outdir_post")
    bilby.utils.check_directory_exists_and_if_not_mkdir(outdir)

    print(f"\n[post:hierarchical] Building likelihood and priors ...")

    # Inspiral posteriors (z≤0.2 population) supply log_lam_t constraints
    means_log_lam = pre_post_means[:, 1]
    vars_log_lam  = pre_post_covs[:, 1, 1]

    likelihood = EMRelationLikelihood(
        means_log_lam=means_log_lam,
        vars_log_lam=vars_log_lam,
        means_fp=post_means,
        vars_fp=post_covs,
        M=mc_samples,
    )

    # linear model
    priors = {
        "alpha": bilby.core.prior.Uniform(
            minimum=-5, maximum=-0, name="alpha", latex_label=r"$\alpha$"),
        "beta":  bilby.core.prior.Uniform(
            minimum=0,  maximum=5, name="beta",  latex_label=r"$\beta$"),
    }

    BOUNDS = np.array([
        [priors[p].minimum, priors[p].maximum]
        for p in ["alpha", "beta"]
    ])

    p0_centre = BOUNDS.mean(axis=1)
    p0_scale  = (BOUNDS[:, 1] - BOUNDS[:, 0]) * 0.05
    rng       = np.random.default_rng(seed=42)
    init_pos  = np.clip(
        p0_centre + rng.standard_normal((n_walkers, 2)) * p0_scale,
        BOUNDS[:, 0], BOUNDS[:, 1],
    )

    print(f"[post:hierarchical] Running emcee: {n_walkers} walkers × {n_steps} steps ...")
    result = bilby.run_sampler(
        likelihood=likelihood,
        priors=priors,
        sampler="emcee",
        nwalkers=n_walkers,
        nsteps=n_steps,
        pos0=init_pos,
        outdir=outdir,
        label="unif_hierarchical",
        npool=n_pool,
    )
    result.plot_corner()
    print(f"[post:hierarchical] Result saved → {outdir}/")
    return result


def _load_post_hierarchical(run_dir: str):
    pkl = _path(run_dir, "outdir_post", "unif_hierarchical_result.json")
    return bilby.core.result.read_in_result(filename=pkl)


# ===========================================================================
# DD2F-SF phase-transition reference data (module-level constants)
# ===========================================================================
# Model indices follow Bauswein+2018 Table 1; model 2 is excluded because it
# uses a different hadronic baseline.
# _PT_MODEL_NUMBERS  = [1, 3, 4, 5, 6, 7]
_PT_MODEL_NUMBERS_PLOT  = [4, 7]
_PT_DELTA_N_PLOT        = np.array([0.082, 0.030])

_PT_DELTA_N        = np.array([0.106, 0.094, 0.082, 0.108, 0.121, 0.030])  # fm⁻³
_PT_F_PEAKS        = np.array([3.54,  3.58,  3.36,  3.59,  3.67,  3.33])   # kHz
_PT_F_PEAK_DD2F    = 3.098   # kHz  – hadronic DD2F baseline at lam_comp
_PT_DELTA_F_PEAKS  = _PT_F_PEAKS - _PT_F_PEAK_DD2F                          # kHz
# _PT_LAM_COMP       = 531.14  # fixed Λ̃ anchor (1.35 M☉ star)
_PT_LAM_COMP       = 631  # fixed Λ̃ anchor (MPA1 1.35 M☉ star)

# OLS slope through origin: Δf_peak = a · Δn
_PT_A_LINEAR = (_PT_DELTA_N @ _PT_DELTA_F_PEAKS) / (_PT_DELTA_N @ _PT_DELTA_N)

# Each model's Δn expressed as a detectable-threshold reference [fm⁻³]
# (identical to _PT_DELTA_N, kept separately for semantic clarity in plots)
# _PT_DELTA_N_REFS = _PT_DELTA_F_PEAKS / _PT_A_LINEAR
_PT_DELTA_N_REFS = _PT_DELTA_N


def _delta_n_min_from_result(result, n_ppd_samples: int) -> float:
    """
    Compute the minimum detectable baryon density jump Δn_min [fm⁻³] for a
    single post:hierarchical result.

    Evaluates the PPD at the fixed anchor Λ̃ = lam_comp, takes the 95th
    percentile of f_peak there, subtracts the MPA1 baseline at the same point,
    and converts the frequency threshold to Δn via the DD2F-SF linear fit.

    Parameters
    ----------
    result        : bilby Result object from post:hierarchical
    n_ppd_samples : number of posterior draws to use

    Returns
    -------
    delta_n_min : float  [fm⁻³]
    """
    post = result.posterior
    idx  = np.random.default_rng(0).integers(0, len(post), size=n_ppd_samples)

    fp_at_lam_comp = np.array([
        - 10.0 ** post["alpha"].iloc[i] * _PT_LAM_COMP
        + post["beta"].iloc[i]
        for i in idx
    ])
    fp_hi_at_lam_comp = np.percentile(fp_at_lam_comp, 95)

    M_grid        = np.linspace(MMIN, MMAX, 1000)
    eos_true      = lalsim.SimNeutronStarEOSByName(EOS)
    fam_true      = lalsim.CreateSimNeutronStarFamily(eos_true)
    lam_grid      = np.array([lambda_from_mass_and_family(m / 2, fam_true) for m in M_grid])
    M_at_lam_comp = M_grid[np.argmin(np.abs(lam_grid - _PT_LAM_COMP))]
    fp_mpa1       = pm.fpeak_ts(np.array([M_at_lam_comp]))[0]

    delta_f_thresh = fp_hi_at_lam_comp - fp_mpa1
    return delta_f_thresh / _PT_A_LINEAR


# ===========================================================================
# PPD computation helper for post-merger (shared by post:ppd and multi:post_ppd)
# ===========================================================================
def _compute_post_ppd_arrays(
    result,
    n_ppd_samples: int,
    n_lam_points:  int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Draw (α, β) posterior samples and evaluate f_peak(Λ̃) on a Λ̃ grid.

    The Λ̃ grid is fixed to the range spanned by the MPA1 truth curve so that
    all run directories share the same x-axis, enabling direct visual comparison
    in the overlay figure.

    Returns
    -------
    lam_plot : (n_lam_points,)              Λ̃ grid
    fp_ppd   : (n_ppd_samples, n_lam_points) f_peak draws [kHz]
    lam_true : (n_M_grid,)                  MPA1 Λ̃ values
    fp_true  : (n_M_grid,)                  MPA1 f_peak values [kHz]
    """
    M_grid = np.linspace(MMIN, MMAX, 1000)

    eos_true = lalsim.SimNeutronStarEOSByName(EOS)
    fam_true = lalsim.CreateSimNeutronStarFamily(eos_true)
    lam_true = np.array([lambda_from_mass_and_family(m / 2, fam_true) for m in M_grid])
    fp_true  = pm.fpeak_ts(M_grid)

    lam_plot = np.linspace(lam_true.min(), lam_true.max(), n_lam_points)

    post = result.posterior
    idx  = np.random.default_rng(0).integers(0, len(post), size=n_ppd_samples)

    fp_ppd = np.array([
        - 10.0 ** post["alpha"].iloc[i] * lam_plot
        + post["beta"].iloc[i]
        for i in idx
    ])

    return lam_plot, fp_ppd, lam_true, fp_true


# ===========================================================================
# Stage post:3 – PPD plot (f_peak vs Λ̃)
# ===========================================================================
def stage_post_ppd(run_dir: str, result, n_ppd_samples: int = 500, n_lam_points: int = 300):
    """Draw (α, β) posterior samples, plot f_peak(Λ̃) PPD with 90% CI, save ppd_post.png."""
    plot_path = _path(run_dir, "ppd_post.png")
    print(f"\n[post:ppd] Drawing {n_ppd_samples} posterior samples ...")

    lam_plot, fp_ppd, lam_true, fp_true = _compute_post_ppd_arrays(
        result, n_ppd_samples, n_lam_points
    )

    fp_median = np.median(fp_ppd, axis=0)
    fp_lo     = np.percentile(fp_ppd,  5, axis=0)
    fp_hi     = np.percentile(fp_ppd, 95, axis=0)

    # ── PT detectability annotation ───────────────────────────────────────────
    post = result.posterior
    idx  = np.random.default_rng(0).integers(0, len(post), size=n_ppd_samples)

    fp_at_lam_comp = np.array([
        - 10.0 ** post["alpha"].iloc[i] * _PT_LAM_COMP
        + post["beta"].iloc[i]
        for i in idx
    ])

    fp_hi_at_lam_comp = np.percentile(fp_at_lam_comp, 95)
    delta_n_min       = _delta_n_min_from_result(result, n_ppd_samples)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.fill_between(lam_plot, fp_lo, fp_hi, alpha=0.3, color="steelblue", label="PPD 90% CI")
    ax.plot(lam_plot, fp_median, color="steelblue", lw=2, label="PPD median")
    ax.plot(lam_true, fp_true,   color="crimson",   lw=2, ls="--",
            label=f"True ({EOS} / Soultanis)")
    ax.scatter(_PT_LAM_COMP, fp_hi_at_lam_comp, color="darkorange", zorder=5,
               label=rf"Min detectable $\Delta n = {delta_n_min:.3f}\ \mathrm{{fm}}^{{-3}}$")

    ax.set_xlabel(r"$\tilde{\Lambda}$", fontsize=13)
    ax.set_ylabel(r"$f_{\rm peak}$ [kHz]", fontsize=13)
    ax.legend(fontsize=11)
    ax.set_title(f"Post-merger PPD: {n_ppd_samples} posterior draws", fontsize=13)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"[post:ppd] Saved → {plot_path}")


# ===========================================================================
# Stage multi:post_ppd – overlay post-merger PPDs from multiple run directories
# ===========================================================================
def stage_multi_post_ppd(
    run_dirs:      list[str],
    labels:        list[str],
    out_dir:       str,
    n_ppd_samples: int = 500,
    n_lam_points:  int = 300,
    force:         bool = False,
):
    """
    Load the post:hierarchical result from each run directory, compute the
    90% CI PPD envelope for each, and produce two figures:

      1. multi_post_ppd.png   – overlaid f_peak(Λ̃) envelopes for all N
      2. delta_n_vs_N.png     – minimum detectable Δn [fm⁻³] vs N, with
                                 horizontal reference lines for each DD2F-SF
                                 model (models 1, 3, 4, 5, 6, 7; model 2
                                 excluded due to a different hadronic baseline)

    N is read from M_tot_draws_post.npy in each run directory (the authoritative
    count for the post-merger arm).  Run directories are processed in the order
    supplied; the overlay preserves that order.  The Δn vs N figure is sorted
    by ascending N regardless of input order.

    Parameters
    ----------
    run_dirs      : pipeline run directories; each must contain
                    outdir_post/unif_hierarchical_result.json and
                    M_tot_draws_post.npy
    labels        : legend label for each run directory in the PPD overlay
    out_dir       : directory where both figures are written
    n_ppd_samples : posterior draws per run directory
    n_lam_points  : Λ̃ grid resolution
    force         : overwrite existing outputs

    Saves
    -----
    <out_dir>/multi_post_ppd.png
    <out_dir>/delta_n_vs_N.png
    """
    ppd_path   = os.path.join(out_dir, "multi_post_ppd.png")
    delta_path = os.path.join(out_dir, "delta_n_vs_N.png")

    need_ppd   = force or not os.path.exists(ppd_path)
    need_delta = force or not os.path.exists(delta_path)

    if not need_ppd and not need_delta:
        print(
            f"\n[multi:post_ppd] Both outputs exist (use --force to regenerate):\n"
            f"  {ppd_path}\n  {delta_path}"
        )
        return

    if len(run_dirs) != len(labels):
        sys.exit(
            f"ERROR: [multi:post_ppd] --run-dirs ({len(run_dirs)} entries) and "
            f"--run-labels ({len(labels)} entries) must have the same length."
        )

    os.makedirs(out_dir, exist_ok=True)

    # ── MPA1 truth (computed once; EOS-determined, identical across run dirs) ─
    first_json = _path(run_dirs[0], "outdir_post", "unif_hierarchical_result.json")
    if not os.path.exists(first_json):
        sys.exit(
            f"ERROR: [multi:post_ppd] Result not found: {first_json}. "
            f"Run post:hierarchical for '{run_dirs[0]}' first."
        )
    _result_tmp             = bilby.core.result.read_in_result(filename=first_json)
    _, _, lam_true, fp_true = _compute_post_ppd_arrays(_result_tmp, 1, n_lam_points)

    # ── per-run-dir loop ──────────────────────────────────────────────────────
    # Collect (N, delta_n_min, color) for the Δn vs N figure
    delta_records: list[tuple[int, float, str]] = []

    fig_ppd, ax_ppd = plt.subplots(figsize=(9, 6))

    for i, (run_dir, label) in enumerate(zip(run_dirs, labels)):
        json_path      = _path(run_dir, "outdir_post", "unif_hierarchical_result.json")
        # mtot_post_path = _path(run_dir, "M_tot_draws_post.npy")
        mtot_post_path = _path(run_dir, "M_tot_draws.npy")

        for fpath, stage in [
            (json_path,      "post:hierarchical"),
            (mtot_post_path, "post:event_level"),
        ]:
            if not os.path.exists(fpath):
                sys.exit(
                    f"ERROR: [multi:post_ppd] Required file not found: {fpath}. "
                    f"Run {stage} for '{run_dir}' first."
                )

        N = len(np.load(mtot_post_path))
        print(f"\n[multi:post_ppd] '{label}' | N = {N} | loading {json_path} ...")
        result = bilby.core.result.read_in_result(filename=json_path)

        print(f"[multi:post_ppd] '{label}' | computing PPD ({n_ppd_samples} draws) ...")
        lam_plot, fp_ppd, _, _ = _compute_post_ppd_arrays(result, n_ppd_samples, n_lam_points)

        fp_lo  = np.percentile(fp_ppd,  5, axis=0)
        fp_med = np.median(fp_ppd,         axis=0)
        fp_hi  = np.percentile(fp_ppd, 95, axis=0)

        color = _MULTI_COLORS[i % len(_MULTI_COLORS)]
        ax_ppd.fill_between(lam_plot, fp_lo, fp_hi, alpha=0.25, color=color, label=label)
        # ax_ppd.plot(lam_plot, fp_med, color=color, lw=2, label=label)

        # ── Δn_min for this run dir ───────────────────────────────────────────
        delta_n_min = _delta_n_min_from_result(result, n_ppd_samples)
        delta_records.append((N, delta_n_min, color))
        print(f"[multi:post_ppd] '{label}' | Δn_min = {delta_n_min:.4f} fm⁻³")

    # ── finalise PPD overlay ──────────────────────────────────────────────────
    ax_ppd.plot(lam_true, fp_true, color="crimson", lw=2.5, ls="--",
                label=f"Injection truth ({EOS} / Soultanis)")
    ax_ppd.set_xlabel(r"$\tilde{\Lambda}$", fontsize=13)
    ax_ppd.set_ylabel(r"$f_{\rm peak}$ [kHz]", fontsize=13)
    ax_ppd.legend(fontsize=11)
    # ax_ppd.set_title("Post-merger PPD: 90% CL envelopes", fontsize=13)
    fig_ppd.tight_layout()

    if need_ppd:
        fig_ppd.savefig(ppd_path, dpi=150)
        print(f"\n[multi:post_ppd] Saved → {ppd_path}")
    else:
        print(f"\n[multi:post_ppd] Skipping {ppd_path} (exists; use --force to overwrite)")
    plt.close(fig_ppd)

    # ── Δn_min vs N figure ────────────────────────────────────────────────────
    if not delta_records:
        print("[multi:post_ppd] WARNING: no Δn records; skipping delta_n_vs_N.png")
        return

    if need_delta:
        delta_records.sort(key=lambda r: r[0])
        ns        = [r[0] for r in delta_records]
        delta_ns  = [r[1] for r in delta_records]
        colors    = [r[2] for r in delta_records]

        fig_d, ax_d = plt.subplots(figsize=(10, 6))

        # ── data curve ───────────────────────────────────────────────────────
        ax_d.plot(ns, delta_ns, color="dimgray", lw=3.0, zorder=1)
        for n_val, dn_val, c in zip(ns, delta_ns, colors):
            ax_d.scatter(n_val, dn_val, color=c, s=120, zorder=2)

        # ── DD2F-SF horizontal reference lines ────────────────────────────────
        x_max = max(ns) if ns else 1
        ax_d.set_xlim(left=0, right=x_max * 1.05)

        line_styles = ["--", "-.", ":", "--", "-.", ":"]
        for model_num, dn_ref, ls in zip(
            _PT_MODEL_NUMBERS_PLOT, _PT_DELTA_N_PLOT, line_styles
        ):
            ax_d.axhline(
                y=dn_ref, color="dimgray", lw=3.0, ls=ls, alpha=0.7, zorder=0,
                label=f"DD2F-SF {model_num}",
            )
            # ax_d.axhline(
            #     y=dn_ref, color="dimgray", lw=3.0, ls=ls, alpha=0.7, zorder=0,
            # )
            # ax_d.annotate(
            #     f"DD2F-SF {model_num}",
            #     xy=(1.0, dn_ref),
            #     xycoords=("axes fraction", "data"),
            #     xytext=(4, 0),
            #     textcoords="offset points",
            #     va="center", ha="left",
            #     fontsize=20, color="dimgray",
            # )

        ax_d.set_xlabel(r"$N$", fontsize=25)
        ax_d.set_ylabel(r"$\Delta n_{\rm min}\ [\mathrm{fm}^{-3}]$", fontsize=25)
        ax_d.tick_params(axis="both", labelsize=25)
        ax_d.set_ylim(bottom=0)
        # ax_d.set_title(
        #     r"Post-merger PT sensitivity: min detectable $\Delta n$ vs $N$",
        #     fontsize=13,
        # )
        ax_d.legend(fontsize=20, loc="upper right")

        fig_d.tight_layout()
        fig_d.subplots_adjust(right=0.82)
        fig_d.savefig(delta_path, dpi=150)
        plt.close(fig_d)
        print(f"[multi:post_ppd] Saved → {delta_path}")
    else:
        print(f"[multi:post_ppd] Skipping {delta_path} (exists; use --force to overwrite)")


# ===========================================================================
# CLI
# ===========================================================================
ALL_SINGLE_STAGES = [
    "pre:prior_kde",
    "pre:event_level",
    "pre:hierarchical",
    "pre:ppd",
    "post:event_level",
    "post:hierarchical",
    "post:ppd",
]
ALL_STAGES = ALL_SINGLE_STAGES + ["multi:ppd", "multi:post_ppd"]


def parse_args():
    p = argparse.ArgumentParser(description="Grand unified BNS inference pipeline")
    p.add_argument("--run-dir",        default="pipeline_run",
                   help="Root directory for all outputs (default: pipeline_run)")
    p.add_argument("--stages",         nargs="+", default=ALL_SINGLE_STAGES,
                   choices=ALL_STAGES, metavar="STAGE",
                   help=f"Stages to run. Choices: {ALL_STAGES}")
    p.add_argument("--n-events",       type=int, default=50,
                   help="Number of BNS events (default: 50)")
    p.add_argument("--n-kde-samples",  type=int, default=10000,
                   help="Viable EOS samples for prior KDE (default: 10000)")
    p.add_argument("--nwalkers-pre",   type=int, default=64,
                   help="emcee walkers for pre:hierarchical (default: 64)")
    p.add_argument("--nsteps-pre",     type=int, default=5000,
                   help="emcee steps for pre:hierarchical (default: 5000)")
    p.add_argument("--nwalkers-post",  type=int, default=32,
                   help="emcee walkers for post:hierarchical (default: 32)")
    p.add_argument("--nsteps-post",    type=int, default=5000,
                   help="emcee steps for post:hierarchical (default: 5000)")
    p.add_argument("--npool",          type=int, default=10,
                   help="Parallel core size for GWBench, emcee and prior KDE (default: 10)")
    p.add_argument("--mc-samples",     type=int, default=100,
                   help="Monte Carlo samples per event in post converse-likelihood (default: 10)")
    p.add_argument("--n-ppd-samples",  type=int, default=200,
                   help="Posterior draws for PPD plots (default: 200)")
    p.add_argument("--seed",           type=int, default=29378,
                   help="RNG seed for injection draws (default: 29378)") 
    p.add_argument("--force",          action="store_true",
                   help="Re-run stages even if their output already exists")
    # ── multi:ppd-specific arguments ─────────────────────────────────────────
    p.add_argument("--run-dirs",       nargs="+", default=[],
                   metavar="PATH",
                   help="Run directories to overlay in multi:ppd")
    p.add_argument("--run-labels",     nargs="+", default=[],
                   metavar="LABEL",
                   help="Legend labels for each entry in --run-dirs (e.g. 'N=50')")
    p.add_argument("--multi-out-dir",  default=".",
                   metavar="PATH",
                   help="Output directory for multi_ppd.png and multi_post_ppd.png (default: current dir)")
    return p.parse_args()


def _need(run_dir, *parts, stage_name, force):
    """Return True if the stage should run (output absent or --force)."""
    out_path = _path(run_dir, *parts)
    if force or not os.path.exists(out_path):
        return True
    print(f"\n[{stage_name}] Output exists, loading from {out_path}")
    return False


def _require(run_dir, *parts, stage_name, prereq_stage):
    """Abort with a clear message if a prerequisite output is missing."""
    out_path = _path(run_dir, *parts)
    if not os.path.exists(out_path):
        sys.exit(
            f"ERROR: [{stage_name}] requires {out_path}. "
            f"Run {prereq_stage} first."
        )


def main():
    args    = parse_args()
    run_dir = args.run_dir
    os.makedirs(run_dir, exist_ok=True)
    print(f"Run directory: {os.path.abspath(run_dir)}")

    stages = args.stages

    # ── pre:prior_kde ────────────────────────────────────────────────────────
    samples_path = _path(run_dir, "viable_gamma_samples.npy")
    if "pre:prior_kde" in stages:
        if _need(run_dir, "viable_gamma_samples.npy",
                 stage_name="pre:prior_kde", force=args.force):
            kde, bounds = stage_pre_prior_kde(
                run_dir, args.n_kde_samples, n_pool=args.npool)
        else:
            kde, bounds = _load_pre_prior_kde(run_dir)
    else:
        if os.path.exists(samples_path):
            kde, bounds = _load_pre_prior_kde(run_dir)
        else:
            kde, bounds = None, None

    # ── pre:event_level ──────────────────────────────────────────────────────
    pre_event_path = _path(run_dir, "event_data_pre.npz")
    if "pre:event_level" in stages:
        if _need(run_dir, "event_data_pre.npz",
                 stage_name="pre:event_level", force=args.force):
            pre_covs, pre_means = stage_pre_event_level(
                run_dir, args.n_events, args.seed, args.npool)
        else:
            pre_covs, pre_means = _load_pre_event_data(run_dir)
    else:
        if os.path.exists(pre_event_path):
            pre_covs, pre_means = _load_pre_event_data(run_dir)
        else:
            pre_covs = pre_means = None

    # ── pre:hierarchical ─────────────────────────────────────────────────────
    pre_result_pkl = _path(run_dir, "outdir_pre", "eos_hierarchical_result.pkl")
    pre_result = None
    if "pre:hierarchical" in stages:
        if kde is None:
            _require(run_dir, "viable_gamma_samples.npy",
                     stage_name="pre:hierarchical", prereq_stage="pre:prior_kde")
            kde, bounds = _load_pre_prior_kde(run_dir)
        if pre_covs is None:
            _require(run_dir, "event_data_pre.npz",
                     stage_name="pre:hierarchical", prereq_stage="pre:event_level")
            pre_covs, pre_means = _load_pre_event_data(run_dir)
        if _need(run_dir, "outdir_pre", "eos_hierarchical_result.pkl",
                 stage_name="pre:hierarchical", force=args.force):
            pre_result = stage_pre_hierarchical(
                run_dir, pre_covs, pre_means, kde, bounds,
                n_walkers=args.nwalkers_pre,
                n_steps=args.nsteps_pre,
                n_pool=args.npool,
            )
        else:
            pre_result = _load_pre_hierarchical(run_dir)
    else:
        if os.path.exists(pre_result_pkl):
            pre_result = _load_pre_hierarchical(run_dir)

    # ── pre:ppd ──────────────────────────────────────────────────────────────
    if "pre:ppd" in stages:
        if pre_result is None:
            _require(run_dir, "outdir_pre", "eos_hierarchical_result.pkl",
                     stage_name="pre:ppd", prereq_stage="pre:hierarchical")
            pre_result = _load_pre_hierarchical(run_dir)
        ppd_pre_path = _path(run_dir, "ppd_pre.png")
        if _need(run_dir, "ppd_pre.png", stage_name="pre:ppd", force=args.force):
            stage_pre_ppd(run_dir, pre_result, n_ppd_samples=args.n_ppd_samples)
        else:
            print(f"\n[pre:ppd] Output exists at {ppd_pre_path} (use --force to regenerate)")

    # ── post:event_level ─────────────────────────────────────────────────────
    # Returns four arrays: (pre_post_covs, pre_post_means, post_covs, post_means)
    pre_post_covs = pre_post_means = post_covs = post_means = None
    post_event_path = _path(run_dir, "event_data_post.npz")
    if "post:event_level" in stages:
        if _need(run_dir, "event_data_post.npz",
                 stage_name="post:event_level", force=args.force):
            pre_post_covs, pre_post_means, post_covs, post_means = \
                stage_post_event_level(run_dir, args.n_events, args.seed, args.npool)
        else:
            pre_post_covs, pre_post_means, post_covs, post_means = \
                _load_post_event_data(run_dir)
    else:
        if os.path.exists(post_event_path):
            pre_post_covs, pre_post_means, post_covs, post_means = \
                _load_post_event_data(run_dir)

    # ── post:hierarchical ────────────────────────────────────────────────────
    post_result_pkl = _path(run_dir, "outdir_post", "unif_hierarchical_result.json")
    post_result = None
    if "post:hierarchical" in stages:
        if pre_post_covs is None:
            _require(run_dir, "event_data_pre_post.npz",
                     stage_name="post:hierarchical", prereq_stage="post:event_level")
            pre_post_covs, pre_post_means, post_covs, post_means = \
                _load_post_event_data(run_dir)
        if _need(run_dir, "outdir_post", "unif_hierarchical_result.json",
                 stage_name="post:hierarchical", force=args.force):
            post_result = stage_post_hierarchical(
                run_dir, pre_post_covs, pre_post_means, post_covs, post_means,
                n_walkers=args.nwalkers_post,
                n_steps=args.nsteps_post,
                n_pool=args.npool,
                mc_samples=args.mc_samples,
            )
        else:
            post_result = _load_post_hierarchical(run_dir)
    else:
        if os.path.exists(post_result_pkl):
            post_result = _load_post_hierarchical(run_dir)

    # ── post:ppd ─────────────────────────────────────────────────────────────
    if "post:ppd" in stages:
        if post_result is None:
            _require(run_dir, "outdir_post", "unif_hierarchical_result.json",
                     stage_name="post:ppd", prereq_stage="post:hierarchical")
            post_result = _load_post_hierarchical(run_dir)
        ppd_post_path = _path(run_dir, "ppd_post.png")
        if _need(run_dir, "ppd_post.png", stage_name="post:ppd", force=args.force):
            stage_post_ppd(run_dir, post_result, n_ppd_samples=args.n_ppd_samples)
        else:
            print(f"\n[post:ppd] Output exists at {ppd_post_path} (use --force to regenerate)")

    # ── multi:ppd ────────────────────────────────────────────────────────────
    if "multi:ppd" in stages:
        if not args.run_dirs:
            sys.exit(
                "ERROR: [multi:ppd] requires --run-dirs. "
                "Provide one path per N value, in the desired legend order."
            )
        labels = args.run_labels if args.run_labels else [
            os.path.basename(os.path.abspath(d)) for d in args.run_dirs
        ]
        stage_multi_ppd(
            run_dirs=args.run_dirs,
            labels=labels,
            out_dir=args.multi_out_dir,
            n_ppd_samples=args.n_ppd_samples,
            n_mass_points=200,
            force=args.force,
        )

    # ── multi:post_ppd ───────────────────────────────────────────────────────
    if "multi:post_ppd" in stages:
        if not args.run_dirs:
            sys.exit(
                "ERROR: [multi:post_ppd] requires --run-dirs. "
                "Provide one path per N value, in the desired legend order."
            )
        labels = args.run_labels if args.run_labels else [
            os.path.basename(os.path.abspath(d)) for d in args.run_dirs
        ]
        stage_multi_post_ppd(
            run_dirs=args.run_dirs,
            labels=labels,
            out_dir=args.multi_out_dir,
            n_ppd_samples=args.n_ppd_samples,
            n_lam_points=300,
            force=args.force,
        )

    print(f"\nPipeline complete. All outputs in: {os.path.abspath(run_dir)}/")


if __name__ == "__main__":
    main()