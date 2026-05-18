"""
End-to-end EOS inference pipeline
==================================
Stages (each is skipped if its output already exists in RUN_DIR):

  1. prior_kde   – rejection-sample viable (γ₀, γ₁, γ₂) triples, fit KDE,
                   save viable_gamma_samples.npy; bounds derived from sample
                   min/max and saved to bounds.npy
  2. event_level – Fisher-matrix event-level runs, save event_data.npz
  3. hierarchical – emcee on EOSHyperparameterLikelihood, bilby result saved
                    to outdir/ inside RUN_DIR
  4. ppd         – posterior predictive λ(m) plot, save ppd.png

Storage conventions (matching original notebooks):
  viable_gamma_samples.npy  – np.save / np.load          (stage 1)
  bounds.npy                – np.save / np.load, shape (5,2)  (stage 1)
  event_data.npz            – np.savez / np.load          (stage 2)
  outdir/eos_hierarchical_result.pkl  – bilby             (stage 3)
  ppd.png                             – matplotlib         (stage 4)

Usage
-----
  python run_pipeline.py [--run-dir PATH] [--stages STAGE [STAGE ...]]
                         [--n-events N] [--n-kde-samples N]
                         [--nwalkers N] [--nsteps N] [--npool N]
                         [--force]

  --stages   subset of {prior_kde, event_level, hierarchical, ppd}
             (default: all four, in order)
  --force    re-run every requested stage even if output exists
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

from converse_likelihood import EOSHyperparameterLikelihood, _build_family
from KDEPrior import KDEJointDist

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Fixed EOS (injection truth)
# ---------------------------------------------------------------------------
PARAM1_TRUE  = 4.183994058757201
PARAM2_TRUE  = 3.168076721819637
PARAM3_TRUE  = 1.300271383168368
LOG10_P1_CGS = 35.293288373856534
LOG10_P2_CGS = 35.648689969687915
CAUSAL       = False

MMIN         = 1.0   # M☉  component-mass prior bounds
MMAX         = 2.0
MIN_MAX_MASS = 2.0   # M☉  viability floor on EOS maximum mass


# ---------------------------------------------------------------------------
# Shared viability check – thin wrapper over _build_family
# ---------------------------------------------------------------------------
def is_viable(g0: float, g1: float, g2: float, lp1: float, lp2: float) -> bool:
    """True iff the EOS is physical and supports M_max >= MIN_MAX_MASS."""
    family = _build_family(g0, lp1, g1, lp2, g2, int(CAUSAL))
    if family is None:
        return False
    return lalsim_SimNeutronStarMaximumMass(family) / solar_mass >= MIN_MAX_MASS


# ===========================================================================
# Stage 1 – prior KDE
# ===========================================================================

def _check_candidate(args):
    """Worker function for parallel rejection sampling (must be top-level for pickle)."""
    g0, g1, g2, lp1, lp2 = args
    return (g0, g1, g2, lp1, lp2) if is_viable(g0, g1, g2, lp1, lp2) else None


def stage_prior_kde(run_dir: str, n_samples: int, n_pool: int = 1) -> tuple:
    """
    Rejection-sample n_samples viable EOS 5-tuples, fit a KDE, derive
    per-parameter bounds from the sample min/max.

    Uses a multiprocessing pool for parallel viability checks, with staged
    time estimates printed after the first 100 and 1000 accepted samples.

    Saves
    -----
    viable_gamma_samples.npy  shape (n_samples, 5)
    bounds.npy                shape (5, 2)  – [[lo0,hi0],...,[lo4,hi4]]
    """
    import time
    from multiprocessing import Pool

    samples_path = os.path.join(run_dir, "viable_gamma_samples.npy")
    bounds_path  = os.path.join(run_dir, "bounds.npy")

    print(f"\n[prior_kde] Rejection-sampling {n_samples} viable EOS 5-tuples "
          f"using {n_pool} process(es) ...")

    viable        = []
    n_total       = 0
    BATCH         = n_pool * 200   # candidates per parallel batch
    estimate_done = {100: False, 1000: False}
    t_start       = time.perf_counter()

    def _print_estimate(n_viable, n_tried, label):
        elapsed   = time.perf_counter() - t_start
        rate      = n_viable / elapsed           # viable/s
        remaining = (n_samples - n_viable) / rate if rate > 0 else float("inf")
        accept    = n_viable / n_tried
        print(
            f"[prior_kde]  {label}: {n_viable} accepted / {n_tried} tried  "
            f"(acceptance {accept:.2%})  |  "
            f"elapsed {elapsed:.1f}s  |  "
            f"estimated remaining {remaining/60:.1f} min"
        )

    with Pool(processes=n_pool) as pool:
        while len(viable) < n_samples:
            # Draw a batch of raw candidates
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

            results = pool.map(_check_candidate, candidates)
            for r in results:
                if r is not None and len(viable) < n_samples:
                    viable.append(r)

            n_acc = len(viable)
            for milestone, done in list(estimate_done.items()):
                if not done and n_acc >= milestone:
                    _print_estimate(n_acc, n_total, f"After {milestone} accepted")
                    estimate_done[milestone] = True

            print(
                f"[prior_kde]  {n_acc}/{n_samples} accepted "
                f"({n_acc/n_samples:.1%})  |  tried {n_total}",
                end="\r", flush=True,
            )

    print()  # newline after \r progress

    viable = np.array(viable[:n_samples])          # (n_samples, 5)
    elapsed_total = time.perf_counter() - t_start
    print(f"[prior_kde] Done in {elapsed_total/60:.1f} min  |  "
          f"acceptance rate: {n_samples / n_total:.3%}")

    np.save(samples_path, viable)
    print(f"[prior_kde] Saved viable samples -> {samples_path}")

    bounds = np.column_stack([viable.min(axis=0), viable.max(axis=0)])  # (5, 2)
    np.save(bounds_path, bounds)
    print(f"[prior_kde] Derived bounds (g0, g1, g2, lp1, lp2):")
    for i, name in enumerate(["g0", "g1", "g2", "lp1", "lp2"]):
        print(f"             {name}: [{bounds[i, 0]:.4f}, {bounds[i, 1]:.4f}]")
    print(f"[prior_kde] Saved bounds -> {bounds_path}")

    kde = gaussian_kde(viable.T, bw_method="scott")
    return kde, bounds


def load_prior_kde(run_dir: str) -> tuple:
    """Reconstruct KDE from saved viable samples; load bounds array."""
    viable = np.load(os.path.join(run_dir, "viable_gamma_samples.npy"))
    bounds = np.load(os.path.join(run_dir, "bounds.npy"))
    kde    = gaussian_kde(viable.T, bw_method="scott")
    return kde, bounds


# ===========================================================================
# Stage 2 – event-level Fisher runs
# ===========================================================================
def stage_event_level(run_dir: str, n_events: int, seed: int = 29378) -> tuple:
    """
    Run Fisher-matrix analysis for n_events BNS events.

    Saves
    -----
    event_data.npz  – keys: covs (N,2,2), means (N,2)
    """
    event_path = os.path.join(run_dir, "event_data.npz")

    print(f"\n[event_level] Running {n_events} Fisher-matrix events ...")
    random.seed(123)

    cosmo_dict = {"zmin": 0, "zmax": 0.2, "sampler": "uniform_comoving_volume_inversion"}
    mass_dict  = {"dist": "uniform", "mmin": MMIN, "mmax": MMAX}
    spin_dict  = {"dim": 1, "geom": "cartesian", "chi_lo": 0, "chi_hi": 0}

    injections_data = injections_CBC_params_redshift(
        cosmo_dict, mass_dict, spin_dict,
        redshifted=1, num_injs=n_events, seed=seed, file_path=None,
    )

    wf_model_name          = "lal_bns"
    wf_other_var_dic       = {"approximant": "IMRPhenomD_NRTidalv2"}
    deriv_symbs_string     = "Mc DL tc phic iota lam_t ra dec psi"
    ana_deriv_symbs_string = "DL tc phic ra dec psi"
    conv_cos               = ("dec", "iota")
    conv_log               = ("DL", "lam_t")

    covs  = []
    means = []

    for inj_id in range(n_events):
        m = np.random.uniform(MMIN, MMAX)

        lam_1, lam_2, ok = polytrope_or_causal_params_to_lambda_1_lambda_2(
            param1=PARAM1_TRUE, param2=PARAM2_TRUE, param3=PARAM3_TRUE,
            log10_pressure1_cgs=LOG10_P1_CGS, log10_pressure2_cgs=LOG10_P2_CGS,
            mass_1_source=m, mass_2_source=m, causal=CAUSAL,
        )
        if not ok:
            print(f"[event_level] mass {m:.3f} Msun failed EOS check - skipping")
            continue

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

        net = Network("E", logger_name="CSU", logger_level="WARNING")
        net.set_net_vars(
            wf_model_name=wf_model_name, wf_other_var_dic=wf_other_var_dic,
            f=f, inj_params=inj_params,
            deriv_symbs_string=deriv_symbs_string,
            conv_cos=conv_cos, conv_log=conv_log, use_rot=1,
            ana_deriv_symbs_string=ana_deriv_symbs_string,
        )
        net.calc_errors(
            only_net=1, derivs="num",
            step=1e-6, method="central", order=2,
            gen_derivs=None, num_cores=None,
        )

        keep = ["Mc", "log_lam_t"]
        idx  = [list(net.deriv_variables).index(p) for p in keep]
        covs.append(net.cov[np.ix_(idx, idx)])
        means.append(np.array([
            net.inj_params["Mc"],
            np.log(net.inj_params["lam_t"]),
        ]))

        print(f"[event_level] {inj_id + 1}/{n_events} done", end="\r", flush=True)

    print()
    covs  = np.array(covs)
    means = np.array(means)
    np.savez(event_path, covs=covs, means=means)
    print(f"[event_level] Saved event data -> {event_path}  ({len(means)} events)")
    return covs, means


def load_event_data(run_dir: str) -> tuple:
    data = np.load(os.path.join(run_dir, "event_data.npz"))
    return data["covs"].astype(np.float64), data["means"]


# ===========================================================================
# Stage 3 – hierarchical inference
# ===========================================================================
def stage_hierarchical(
    run_dir:   str,
    covs:      np.ndarray,
    means:     np.ndarray,
    kde:       gaussian_kde,
    bounds:    np.ndarray,           # shape (5, 2)
    n_walkers: int = 64,
    n_steps:   int = 5000,
    n_pool:    int = 10,
):
    """
    Run emcee hierarchical inference.  bilby writes its result to
    RUN_DIR/outdir/eos_hierarchical_result.pkl (bilby's own format).
    """
    outdir = os.path.join(run_dir, "outdir")
    bilby.utils.check_directory_exists_and_if_not_mkdir(outdir)

    print(f"\n[hierarchical] Building likelihood and priors ...")


    # bounds rows: [g0, g1, g2, lp1, lp2]; columns: [lo, hi]
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
        mass_min=MMIN, mass_max=MMAX,
        param_bounds=bounds_dict,
        causal=CAUSAL,
    )

    # Seed walkers from KDE, requiring viability for all 5 parameters
    lo = bounds[:, 0]
    hi = bounds[:, 1]
    init_pos = []
    while len(init_pos) < n_walkers:
        batch     = kde.resample(n_walkers * 10).T          # (10*n_walkers, 5)
        in_bounds = np.all((batch >= lo) & (batch <= hi), axis=1)
        for candidate in batch[in_bounds]:
            if len(init_pos) >= n_walkers:
                break
            if is_viable(candidate[0], candidate[1], candidate[2],
                         candidate[3], candidate[4]):
                init_pos.append(candidate)
    init_pos = np.array(init_pos)

    print(f"[hierarchical] Running emcee: {n_walkers} walkers x {n_steps} steps ...")
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
    print(f"[hierarchical] Result saved to {outdir}/")
    return result


def load_hierarchical(run_dir: str):
    pkl = os.path.join(run_dir, "outdir", "eos_hierarchical_result.pkl")
    return bilby.core.result.read_in_result(filename=pkl)


# ===========================================================================
# Stage 4 – PPD plot
# ===========================================================================
def stage_ppd(run_dir: str, result, n_ppd_samples: int = 200, n_mass_points: int = 200):
    """Draw posterior samples, evaluate lambda(m), save ppd.png."""
    plot_path = os.path.join(run_dir, "ppd.png")
    print(f"\n[ppd] Drawing {n_ppd_samples} posterior samples for PPD ...")

    m_plot    = np.linspace(MMIN, MMAX, n_mass_points)
    posterior = result.posterior
    draws = posterior.sample(n_ppd_samples, replace=True)[
        ["param1", "param2", "param3", "log10_p1", "log10_p2"]
    ]

    fig, ax = plt.subplots(figsize=(8, 5))

    for _, row in draws.iterrows():
        family = _build_family(
            row.param1, row.log10_p1,
            row.param2, row.log10_p2,
            row.param3, int(CAUSAL),
        )
        if family is None:
            continue
        m_min   = lalsim_SimNeutronStarFamMinimumMass(family) / solar_mass
        m_max   = lalsim_SimNeutronStarMaximumMass(family) / solar_mass
        m_valid = m_plot[(m_plot >= m_min) & (m_plot <= m_max)]
        if len(m_valid) == 0:
            continue
        lam = np.array([lambda_from_mass_and_family(m, family) for m in m_valid])
        ax.plot(m_valid, lam, color="steelblue", alpha=0.05, lw=0.8)

    # Injection truth
    family_true = _build_family(
        PARAM1_TRUE, LOG10_P1_CGS,
        PARAM2_TRUE, LOG10_P2_CGS,
        PARAM3_TRUE, int(CAUSAL),
    )
    if family_true is not None:
        m_min_t = lalsim_SimNeutronStarFamMinimumMass(family_true) / solar_mass
        m_max_t = lalsim_SimNeutronStarMaximumMass(family_true) / solar_mass
        m_t     = m_plot[(m_plot >= m_min_t) & (m_plot <= m_max_t)]
        lam_t   = np.array([lambda_from_mass_and_family(m, family_true) for m in m_t])
        ax.plot(m_t, lam_t, color="crimson", lw=2, label="Injection truth")

    ax.set_xlabel(r"$m\ [M_\odot]$")
    ax.set_ylabel(r"$\Lambda$")
    ax.set_xlim(MMIN, MMAX)
    ax.set_ylim(bottom=0)
    ax.legend()
    ax.set_title(f"PPD: {n_ppd_samples} posterior draws")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"[ppd] Saved PPD plot -> {plot_path}")


# ===========================================================================
# CLI
# ===========================================================================
ALL_STAGES = ["prior_kde", "event_level", "hierarchical", "ppd"]


def parse_args():
    p = argparse.ArgumentParser(description="End-to-end EOS inference pipeline")
    p.add_argument("--run-dir",       default="pipeline_run",
                   help="Directory for all outputs (default: pipeline_run)")
    p.add_argument("--stages",        nargs="+", default=ALL_STAGES,
                   choices=ALL_STAGES, metavar="STAGE",
                   help=f"Stages to run (default: all). Choices: {ALL_STAGES}")
    p.add_argument("--n-events",      type=int, default=50,
                   help="Number of GW events (default: 50)")
    p.add_argument("--n-kde-samples", type=int, default=10000,
                   help="Viable EOS samples for KDE (default: 10000)")
    p.add_argument("--nwalkers",      type=int, default=64,
                   help="emcee walkers (default: 64)")
    p.add_argument("--nsteps",        type=int, default=5000,
                   help="emcee steps per walker (default: 5000)")
    p.add_argument("--npool",         type=int, default=10,
                   help="emcee parallel pool size (default: 10)")
    p.add_argument("--n-ppd-samples", type=int, default=200,
                   help="Posterior draws for PPD plot (default: 200)")
    p.add_argument("--force",         action="store_true",
                   help="Re-run stages even if outputs already exist")
    return p.parse_args()


def main():
    args    = parse_args()
    run_dir = args.run_dir
    os.makedirs(run_dir, exist_ok=True)
    print(f"Run directory: {os.path.abspath(run_dir)}")

    stages = args.stages

    # ── Stage 1: prior KDE ──────────────────────────────────────────────────
    samples_path = os.path.join(run_dir, "viable_gamma_samples.npy")
    if "prior_kde" in stages:
        if args.force or not os.path.exists(samples_path):
            kde, bounds = stage_prior_kde(run_dir, args.n_kde_samples, n_pool=args.npool)
        else:
            print(f"\n[prior_kde] Output exists, loading from {samples_path}")
            kde, bounds = load_prior_kde(run_dir)
    else:
        if os.path.exists(samples_path):
            kde, bounds = load_prior_kde(run_dir)
        else:
            sys.exit(f"ERROR: {samples_path} not found. Run prior_kde stage first.")

    # ── Stage 2: event-level ────────────────────────────────────────────────
    event_path = os.path.join(run_dir, "event_data.npz")
    if "event_level" in stages:
        if args.force or not os.path.exists(event_path):
            covs, means = stage_event_level(run_dir, args.n_events)
        else:
            print(f"\n[event_level] Output exists, loading from {event_path}")
            covs, means = load_event_data(run_dir)
    else:
        if os.path.exists(event_path):
            covs, means = load_event_data(run_dir)
        else:
            sys.exit(f"ERROR: {event_path} not found. Run event_level stage first.")

    # ── Stage 3: hierarchical ───────────────────────────────────────────────
    result_pkl = os.path.join(run_dir, "outdir", "eos_hierarchical_result.pkl")
    if "hierarchical" in stages:
        if args.force or not os.path.exists(result_pkl):
            result = stage_hierarchical(
                run_dir, covs, means, kde, bounds,
                n_walkers=args.nwalkers,
                n_steps=args.nsteps,
                n_pool=args.npool,
            )
        else:
            print(f"\n[hierarchical] Output exists, loading from {result_pkl}")
            result = load_hierarchical(run_dir)
    else:
        if os.path.exists(result_pkl):
            result = load_hierarchical(run_dir)
        else:
            result = None

    # ── Stage 4: PPD ────────────────────────────────────────────────────────
    if "ppd" in stages:
        if result is None:
            sys.exit("ERROR: hierarchical result not available. Run hierarchical stage first.")
        ppd_path = os.path.join(run_dir, "ppd.png")
        if args.force or not os.path.exists(ppd_path):
            stage_ppd(run_dir, result, n_ppd_samples=args.n_ppd_samples)
        else:
            print(f"\n[ppd] Output exists at {ppd_path} (use --force to regenerate)")

    print(f"\nPipeline complete. All outputs in: {os.path.abspath(run_dir)}/")


if __name__ == "__main__":
    main()