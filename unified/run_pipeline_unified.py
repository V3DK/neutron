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
  pre:ppd           – posterior predictive λ(m) plot
                      → ppd_pre.png

Post-merger arm:
  post:event_level  – Fisher-matrix runs (post-merger waveform)
                      → event_data_post.npz
                      [requires M_tot_draws.npy from pre:event_level]
  post:hierarchical – emcee on EMRelationLikelihood (converse likelihood)
                      → outdir_post/unif_hierarchical_result.json
  post:ppd          – posterior predictive f_peak(Λ̃) plot
                      → ppd_post.png

Dependency graph
----------------
  pre:prior_kde → pre:event_level → pre:hierarchical → pre:ppd
                       │
                (M_tot_draws.npy)
                       │
               post:event_level → post:hierarchical → post:ppd

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
  M_tot_draws.npy                   – (n_events,)      shared mass draws
  event_data_pre.npz                – keys: covs (N,2,2), means (N,2)
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

  --stages  subset of:
              pre:prior_kde  pre:event_level  pre:hierarchical  pre:ppd
              post:event_level  post:hierarchical  post:ppd
            (default: all seven, in dependency order)
  --force   re-run every requested stage even if output exists
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
COSMO_DICT = {"zmin": 0, "zmax": 0.2, "sampler": "uniform_comoving_volume_inversion"}
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
def _make_injection_draws(n_events: int, seed: int):
    """
    Return (injections_data, M_tot_draws).

    injections_data is the gwbench tuple; indices 8-13 carry extrinsic params.
    M_tot_draws is shape (n_events,), drawn from U[MMIN, MMAX].
    Both pipelines use this function with the same seed to guarantee identical draws.
    """
    mass_dict = {"dist": "uniform", "mmin": MMIN, "mmax": MMAX}
    injections_data = injections_CBC_params_redshift(
        COSMO_DICT, mass_dict, SPIN_DICT,
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
    Fisher-matrix runs for the inspiral waveform.

    Saves
    -----
    event_data_pre.npz   keys: covs (N,2,2), means (N,2)  [Mc, log_lam_t]
    M_tot_draws.npy      shape (N,)                        [shared mass draws]
    """
    event_path  = _path(run_dir, "event_data_pre.npz")
    mtot_path   = _path(run_dir, "M_tot_draws.npy")

    print(f"\n[pre:event_level] Running {n_events} Fisher-matrix events ...")

    injections_data, M_tot_draws = _make_injection_draws(n_events, seed)

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
        print(f"[pre:event_level | inj {inj_id:02d}] M_tot={M_tot:.4f}  f_hi={f_hi:.2f} Hz")

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
# Stage pre:4 – PPD plot (λ vs m)
# ===========================================================================
def stage_pre_ppd(run_dir: str, result, n_ppd_samples: int = 200, n_mass_points: int = 200):
    """Draw EOS posterior samples, evaluate λ(m), save ppd_pre.png."""
    plot_path = _path(run_dir, "ppd_pre.png")
    print(f"\n[pre:ppd] Drawing {n_ppd_samples} posterior samples ...")

    m_plot    = np.linspace(MMIN / 2, MMAX / 2, n_mass_points)
    posterior = result.posterior
    draws     = posterior.sample(n_ppd_samples, replace=True)[
        ["param1", "param2", "param3", "log10_p1", "log10_p2"]
    ]

    fig, ax = plt.subplots(figsize=(8, 5))

    for _, row in draws.iterrows():
        family = _build_family(row.param1, row.log10_p1, row.param2, row.log10_p2, row.param3)
        if family is None:
            continue
        m_min   = lalsim_SimNeutronStarFamMinimumMass(family) / solar_mass
        m_max   = lalsim_SimNeutronStarMaximumMass(family) / solar_mass
        m_valid = m_plot[(m_plot >= m_min) & (m_plot <= m_max)]
        if len(m_valid) == 0:
            continue
        lam = np.array([lambda_from_mass_and_family(m, family) for m in m_valid])
        ax.plot(m_valid, lam, color="steelblue", alpha=0.05, lw=0.8)

    eos_true    = lalsim.SimNeutronStarEOSByName(EOS)
    family_true = lalsim.CreateSimNeutronStarFamily(eos_true)
    if family_true is not None:
        m_min_t = lalsim_SimNeutronStarFamMinimumMass(family_true) / solar_mass
        m_max_t = lalsim_SimNeutronStarMaximumMass(family_true) / solar_mass
        m_t     = m_plot[(m_plot >= m_min_t) & (m_plot <= m_max_t)]
        lam_t   = np.array([lambda_from_mass_and_family(m, family_true) for m in m_t])
        ax.plot(m_t, lam_t, color="crimson", lw=2, label=f"Injection truth ({EOS})")

    ax.set_xlabel(r"$m\ [M_\odot]$")
    ax.set_ylabel(r"$\Lambda$")
    ax.set_xlim(MMIN / 2, MMAX / 2)
    ax.set_ylim(bottom=0)
    ax.legend()
    ax.set_title(f"Pre-merger PPD: {n_ppd_samples} posterior draws")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"[pre:ppd] Saved → {plot_path}")


# ===========================================================================
# Stage post:1 – event-level (post-merger Fisher)
# ===========================================================================
def stage_post_event_level(run_dir: str, n_events: int, seed: int, n_cores: int) -> tuple:
    """
    Fisher-matrix runs for the post-merger waveform, using shared M_tot draws.

    Requires
    --------
    M_tot_draws.npy (written by pre:event_level)

    Saves
    -----
    event_data_post.npz  keys: covs (N,), means (N,)  [fpeak_ts_ variance, mean]
    """
    mtot_path  = _path(run_dir, "M_tot_draws.npy")
    event_path = _path(run_dir, "event_data_post.npz")

    if not os.path.exists(mtot_path):
        sys.exit(
            f"ERROR: {mtot_path} not found. "
            "Run pre:event_level first to generate shared mass draws."
        )

    M_tot_draws     = np.load(mtot_path)
    injections_data, _ = _make_injection_draws(n_events, seed)  # same seed → same extrinsics

    print(f"\n[post:event_level] Running {n_events} post-merger Fisher-matrix events ...")

    post_f_lo = 1.0
    post_f_hi = 8000.0
    post_df   = 2.0 ** -4
    post_f    = np.arange(post_f_lo, post_f_hi + post_df, post_df)

    post_wf_model_name    = "pm_waveform"
    post_wf_other_var_dic = None
    post_user_waveform    = PM_WAVEFORM_PATH

    deriv_symbs_string = (
        "fpeak_ts_ zeta_drift_ t_star_ "
        "f_spiral_ f_2m0_ f_2p0_ "
        "A_peak_ A_spiral_ A_2m0_ A_2p0_ "
        "tau_peak_ tau_spiral_ tau_2m0_ tau_2p0_ "
        "phi_peak_ phi_spiral_ phi_2m0_ phi_2p0_ "
        "N_ "
        "DL tc phic iota"
    )
    ana_deriv_symbs_string = "DL tc phic"
    conv_cos = ("dec", "iota")
    conv_log = ("DL",)

    post_covs  = []
    post_means = []

    for inj_id in range(n_events):
        M_tot = M_tot_draws[inj_id]
        p     = pm.get_params(M_tot)

        inj_params = {
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

        print(f"[post:event_level | inj {inj_id:02d}] M_tot={M_tot:.4f}")

        net = Network("E", logger_name="POST", logger_level="WARNING")
        net.set_net_vars(
            wf_model_name=post_wf_model_name,
            wf_other_var_dic=post_wf_other_var_dic,
            user_waveform=post_user_waveform,
            f=post_f,
            inj_params=inj_params,
            deriv_symbs_string=deriv_symbs_string,
            ana_deriv_symbs_string=ana_deriv_symbs_string,
            conv_cos=conv_cos,
            conv_log=conv_log,
            use_rot=0,
        )
        net.calc_errors(
            only_net=1, derivs="num",
            step=1e-6, method="central", order=2,
            gen_derivs=None, num_cores=n_cores,
        )

        keep    = "fpeak_ts_"
        idx     = net.deriv_variables.index(keep)
        cov_fp  = net.cov[idx, idx]
        mean_fp = net.inj_params[keep]

        post_covs.append(cov_fp)
        post_means.append(mean_fp)

    post_covs  = np.array(post_covs)
    post_means = np.array(post_means)
    np.savez(event_path, covs=post_covs, means=post_means)
    print(f"\n[post:event_level] Saved → {event_path}  ({len(post_means)} events)")
    return post_covs, post_means


def _load_post_event_data(run_dir: str) -> tuple:
    data = np.load(_path(run_dir, "event_data_post.npz"))
    return data["covs"].astype(np.float64), data["means"]


# ===========================================================================
# Stage post:2 – hierarchical (empirical relation)
# ===========================================================================
def stage_post_hierarchical(
    run_dir:       str,
    pre_covs:      np.ndarray,
    pre_means:     np.ndarray,
    post_covs:     np.ndarray,
    post_means:    np.ndarray,
    n_walkers:     int = 32,
    n_steps:       int = 5000,
    n_pool:        int = 10,
    mc_samples:    int = 10,
):
    """
    emcee hierarchical inference on empirical-relation parameters (α, β, γ).
    bilby writes result to RUN_DIR/outdir_post/unif_hierarchical_result.json.
    """
    outdir = _path(run_dir, "outdir_post")
    bilby.utils.check_directory_exists_and_if_not_mkdir(outdir)

    print(f"\n[post:hierarchical] Building likelihood and priors ...")

    # Inspiral posteriors supply log_lam_t constraints
    means_log_lam = pre_means[:, 1]
    vars_log_lam  = pre_covs[:, 1, 1]

    likelihood = EMRelationLikelihood(
        means_log_lam=means_log_lam,
        vars_log_lam=vars_log_lam,
        means_fp=post_means,
        vars_fp=post_covs,
        M=mc_samples,
    )

    priors = {
        "alpha": bilby.core.prior.Uniform(
            minimum=-20, maximum=-2, name="alpha", latex_label=r"$\alpha$"),
        "beta":  bilby.core.prior.Uniform(
            minimum=-5,  maximum=-1, name="beta",  latex_label=r"$\beta$"),
        "gamma": bilby.core.prior.Uniform(
            minimum=2.0, maximum=4.0, name="gamma", latex_label=r"$\gamma$"),
    }

    BOUNDS = np.array([
        [priors[p].minimum, priors[p].maximum]
        for p in ["alpha", "beta", "gamma"]
    ])
    p0_centre = BOUNDS.mean(axis=1)
    p0_scale  = (BOUNDS[:, 1] - BOUNDS[:, 0]) * 0.05
    rng       = np.random.default_rng(seed=42)
    init_pos  = np.clip(
        p0_centre + rng.standard_normal((n_walkers, 3)) * p0_scale,
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
# Stage post:3 – PPD plot (f_peak vs Λ̃)
# ===========================================================================
def stage_post_ppd(run_dir: str, result, n_ppd_samples: int = 500, n_lam_points: int = 300):
    """Draw (α, β, γ) posterior samples, plot f_peak(Λ̃) PPD, save ppd_post.png."""
    plot_path = _path(run_dir, "ppd_post.png")
    print(f"\n[post:ppd] Drawing {n_ppd_samples} posterior samples ...")

    M_grid = np.linspace(MMIN, MMAX, 1000)

    eos_true = lalsim.SimNeutronStarEOSByName(EOS)
    fam_true = lalsim.CreateSimNeutronStarFamily(eos_true)
    lam_true = np.array([lambda_from_mass_and_family(m / 2, fam_true) for m in M_grid])
    fp_true  = pm.fpeak_ts(M_grid)

    post     = result.posterior
    idx      = np.random.default_rng(0).integers(0, len(post), size=n_ppd_samples)
    lam_plot = np.linspace(lam_true.min(), lam_true.max(), n_lam_points)

    fp_ppd = np.array([
        10.0 ** post["alpha"].iloc[i] * lam_plot ** 2
        - 10.0 ** post["beta"].iloc[i]  * lam_plot
        + post["gamma"].iloc[i]
        for i in idx
    ])

    fp_median = np.median(fp_ppd, axis=0)
    fp_lo     = np.percentile(fp_ppd,  5, axis=0)
    fp_hi     = np.percentile(fp_ppd, 95, axis=0)

    # PT stuff
    lam_comp = 531.14
    f_peak_DD2F = 3.098 # kHz

    delta_n_pt = np.array([0.106, 0.094, 0.082, 0.108, 0.121, 0.030])
    f_peaks_pt = np.array([3.54, 3.58, 3.36, 3.59, 3.67, 3.33])

    delta_f_peak_pt = f_peaks_pt - f_peak_DD2F

    # --- Evaluate PPD at lam_comp ---
    fp_ppd_at_lam_comp = np.array([
        10.0 ** post["alpha"].iloc[i] * lam_comp ** 2
        - 10.0 ** post["beta"].iloc[i]  * lam_comp
        + post["gamma"].iloc[i]
        for i in idx
    ])  # (N_PPD,)

    fp_hi_at_lam_comp = np.percentile(fp_ppd_at_lam_comp, 95)

    # --- MPA1 baseline at lam_comp ---
    # lam_comp corresponds to a specific total binary mass; find it
    M_at_lam_comp = M_grid[np.argmin(np.abs(lam_true - lam_comp))]
    fp_mpa1_at_lam_comp = pm.fpeak_ts(np.array([M_at_lam_comp]))[0]

    # --- Threshold and delta_n_min ---
    delta_f_threshold = fp_hi_at_lam_comp - fp_mpa1_at_lam_comp

    # Linear fit through origin: y = a_linear * x
    x = delta_n_pt
    y = delta_f_peak_pt

    a_linear = (x @ y) / (x @ x)  # OLS through origin

    # --- Threshold and delta_n_min ---
    delta_n_min = delta_f_threshold / a_linear

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.fill_between(lam_plot, fp_lo, fp_hi, alpha=0.3, color="steelblue", label="PPD 90% CI")
    ax.plot(lam_plot, fp_median, color="steelblue", lw=2, label="PPD median")
    ax.plot(lam_true, fp_true,   color="crimson",   lw=2, ls="--",
            label=f"True ({EOS} / Soultanis)")
    ax.scatter(lam_comp, fp_hi_at_lam_comp, color="darkorange", zorder=5,
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
# CLI
# ===========================================================================
ALL_STAGES = [
    "pre:prior_kde",
    "pre:event_level",
    "pre:hierarchical",
    "pre:ppd",
    "post:event_level",
    "post:hierarchical",
    "post:ppd",
]


def parse_args():
    p = argparse.ArgumentParser(description="Grand unified BNS inference pipeline")
    p.add_argument("--run-dir",        default="pipeline_run",
                   help="Root directory for all outputs (default: pipeline_run)")
    p.add_argument("--stages",         nargs="+", default=ALL_STAGES,
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
    post_event_path = _path(run_dir, "event_data_post.npz")
    post_covs = post_means = None
    if "post:event_level" in stages:
        _require(run_dir, "M_tot_draws.npy",
                 stage_name="post:event_level", prereq_stage="pre:event_level")
        if _need(run_dir, "event_data_post.npz",
                 stage_name="post:event_level", force=args.force):
            post_covs, post_means = stage_post_event_level(
                run_dir, args.n_events, args.seed, args.npool)
        else:
            post_covs, post_means = _load_post_event_data(run_dir)
    else:
        if os.path.exists(post_event_path):
            post_covs, post_means = _load_post_event_data(run_dir)

    # ── post:hierarchical ────────────────────────────────────────────────────
    post_result_pkl = _path(run_dir, "outdir_post", "unif_hierarchical_result.json")
    post_result = None
    if "post:hierarchical" in stages:
        if pre_covs is None:
            _require(run_dir, "event_data_pre.npz",
                     stage_name="post:hierarchical", prereq_stage="pre:event_level")
            pre_covs, pre_means = _load_pre_event_data(run_dir)
        if post_covs is None:
            _require(run_dir, "event_data_post.npz",
                     stage_name="post:hierarchical", prereq_stage="post:event_level")
            post_covs, post_means = _load_post_event_data(run_dir)
        if _need(run_dir, "outdir_post", "unif_hierarchical_result.json",
                 stage_name="post:hierarchical", force=args.force):
            post_result = stage_post_hierarchical(
                run_dir, pre_covs, pre_means, post_covs, post_means,
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

    print(f"\nPipeline complete. All outputs in: {os.path.abspath(run_dir)}/")


if __name__ == "__main__":
    main()
