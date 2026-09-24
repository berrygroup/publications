"""
Minimal re-implementation of the Pol II pools model (N -> P -> E) kinetic model
used in Gillis et al., refactored to expose the following parameters as explicit
arguments

    w_P          relative weight of the genomic (TSS ChIP) data points
    w_armc5      flat multiplier on the ARMC5 imaging points (1 = no reweighting,
                 as used for the reported fit; varied by the sensitivity analysis)
    pt_fraction  fraction of promoter-proximal Pol II that terminates prematurely
                 at T = 1 (the "80%" external constraint)

Model (k_EN = 1 fixes the timescale):
    dN/dt = k_PN P + k_EN E - v_NP(N)
    dP/dt = v_NP(N) - k_PN P - v_PE(P)
    dE/dt = v_PE(P) - k_EN E
    N + P + E = T
with v_NP(N) = Vmax_NP N /(Km_NP + N),  v_PE(P) = Vmax_PE P /(Km_PE + P).
"""

import os

import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize

DATA_DIR = os.environ.get("POLII_DATA", "../data")

PARAM_NAMES = ["k_pn", "Vmax_pe", "Km_pe", "Vmax_np", "Km_np"]
BOUNDS = [(0.01, 10), (0.01, 20), (0.01, 10), (0.01, 20), (0.01, 10)]
K_EN = 1.0


# ----------------------------------------------------------------------------
# Steady state
# ----------------------------------------------------------------------------
def steady_state(T, params):
    """Steady-state (N, P, E) for total Pol II = T.

    Reduced to a scalar root-find in P: eq. for E gives E = v_PE(P)/k_EN, the
    conservation law gives N = T - P - E, and the remaining balance
        f(P) = v_NP(N) - k_PN P - v_PE(P)
    is strictly decreasing in P on (0, P_max), with f(0+) > 0, so the root is
    unique.  (Same solution as the multi-start `scipy.optimize.root` solver in
    the original notebook, see `check_solver_agreement`.)
    """
    k_pn, Vmax_pe, Km_pe, Vmax_np, Km_np = params

    def NPE(P):
        E = Vmax_pe * P / (Km_pe + P) / K_EN
        return T - P - E, P, E

    def f(P):
        N, P, E = NPE(P)
        return Vmax_np * N / (Km_np + N) - k_pn * P - Vmax_pe * P / (Km_pe + P)

    # P_max: the (unique) P at which N = 0, since P + E(P) increases with P
    if NPE(T)[0] >= 0:
        P_max = T
    else:
        P_max = brentq(lambda P: NPE(P)[0], 1e-15, T, xtol=1e-15, rtol=1e-13)
    lo = 1e-12
    if f(lo) <= 0:  # degenerate: everything nucleoplasmic
        return np.array([T, 0.0, 0.0])
    P = brentq(f, lo, P_max, xtol=1e-14, rtol=1e-12)
    N, P, E = NPE(P)
    return np.array([N, P, E])


def saturation(T, params):
    """(s_NP, s_PE) = (N/(Km_NP+N), P/(Km_PE+P)) at steady state."""
    k_pn, Vmax_pe, Km_pe, Vmax_np, Km_np = params
    N, P, E = steady_state(T, params)
    return N / (Km_np + N), P / (Km_pe + P)


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
def load_data(data_dir=None):
    """Return a dict of the three data blocks used in the global fit.

    Identical selection / normalisation to `pol_pools_mm.ipynb`.
    """
    d = data_dir or DATA_DIR
    p = lambda f: os.path.join(d, f)

    # --- E (pSer2 immunofluorescence) -------------------------------------
    e1 = pd.read_csv(p("mACPOLR2A_pSer2PolII_8hours_Values_ByExperiment_3reps.csv"))
    c1 = e1[(e1["AUXIN_CONCENTRATION_nM"] == 0) & (e1["TIME_IN_AUXIN_HRS"] == 0)]
    T1 = (e1["Total_PolII_F12_Normed"] / c1["Total_PolII_F12_Normed"].mean()).values
    E1 = (e1["pSer2_PolII_3E10_Normed"] / c1["pSer2_PolII_3E10_Normed"].mean()).values

    e2 = pd.read_csv(p("mACmChPOLR2A_ActivePolII_Values_ByExperiment.csv"))
    c2 = e2[e2["TIME_IN_AUXIN_H"] == 0]
    T2 = (e2["Total_PolII_F12_Normed"] / c2["Total_PolII_F12_Normed"].mean()).values
    E2 = (e2["pSer2_PolII_3E10_Normed"] / c2["pSer2_PolII_3E10_Normed"].mean()).values

    a_T = pd.read_csv(p("ARMC5_mAC_Clone5A_TotalPolII_Values_ByExperiment_3reps.csv"))
    a_E = pd.read_csv(p("ARMC5_mAC_Clone5A1_Phospho_Values_ByExperiment_3reps.csv"))
    aT = a_T.groupby("TIME_IN_AUXIN_H")["Total_PolII_F12_Normed"].agg(["mean", "std"])
    aE = a_E.groupby("TIME_IN_AUXIN_H")["pSer2_PolII_3E10_Normed"].agg(["mean", "std"])
    a = aT.merge(aE, on="TIME_IN_AUXIN_H", suffixes=("_T", "_E")).reset_index()
    T3 = (a["mean_T"] / a.loc[a.TIME_IN_AUXIN_H == 0, "mean_T"].mean()).values
    E3 = (a["mean_E"] / a.loc[a.TIME_IN_AUXIN_H == 0, "mean_E"].mean()).values
    E3_sd = (a["std_E"] / a.loc[a.TIME_IN_AUXIN_H == 0, "mean_E"].mean()).values

    # --- bound fraction (FRAP) --------------------------------------------
    b1 = pd.read_csv(p("mACmChPOLR2A_FRAP_Amounts_perday.csv"))
    b1 = b1[b1["CONDITION"] != "mACmChPOLR2A VEH"]
    b2 = pd.read_csv(p("mACmChPOLR2A_mClover3_FRAP_Amounts_perday.csv"))
    b2 = b2[b2["CONDITION"] == "mACPOLR2A VEH"]
    b3 = pd.read_csv(p("mACPOLR2A_biallelic_FRAP_Bound_Free_RelativeTotal.csv"))
    b4 = pd.read_csv(p("mChPOLR2A_FRAP_Scrambled_ARMC5_amounts_per_day.csv"))
    b4 = b4[b4["CONDITION"].isin(["ARMC5 VEHICLE", "SCRAMBLED VEHICLE"])]

    Tb = [
        b1["RELATIVE_TOTAL"].values / 100,
        b2["RELATIVE_TOTAL"].values / 100,
        b3["RELATIVE_TOTAL"].values / 100,
        b4["RELATIVE_TOTAL"].values / 100,
    ]
    Bb = [
        b1["BOUND_FRACTION"].values,
        b2["BOUND"].values / 100,
        b3["BOUND_FRACTION"].values,
        b4["BOUND_FRACTION"].values,
    ]

    T_E = np.concatenate([T1, T2, T3])
    E = np.concatenate([E1, E2, E3])
    E_block = np.concatenate(
        [np.full(len(T1), 0), np.full(len(T2), 1), np.full(len(T3), 2)]
    )
    E_sd = np.concatenate([np.full(len(T1), np.nan), np.full(len(T2), np.nan), E3_sd])

    T_P = np.array([0.738, 1.0, 1.51])
    P = np.array([0.909, 1.0, 1.45])

    # Flag the self-normalisation points.  Where a series is normalised to its
    # own vehicle / t = 0 control, that control is exactly 1 at T = 1 by
    # construction, and the model predicts E*(1)/E*(1) = 1 identically, so the
    # residual is zero for every parameter set.  Such points cannot constrain
    # the fit, but counting them would inflate the number of observations used
    # to estimate the error scale.  They are EXCLUDED FROM FITTING and from all
    # quantitative analysis via these masks, and RETAINED IN THE RETURNED DATA
    # so that plots still show them.
    E_fit = ~((T_E == 1.0) & (E == 1.0))
    P_fit = ~((T_P == 1.0) & (P == 1.0))

    return dict(
        T_E=T_E,
        E=E,
        E_block=E_block,
        E3_sd=E3_sd,
        # per-row SD for the E block (NaN where not applicable); follows rows
        # through bootstrap resampling
        E_sd=E_sd,
        # False for self-normalisation controls: shown on plots, not fitted
        E_fit=E_fit,
        T_B=np.concatenate(Tb),
        B=np.concatenate(Bb),
        B_block=np.concatenate([np.full(len(t), i) for i, t in enumerate(Tb)]),
        # TSS Pol II by dxChIP-seq (half-degron 90 min; ARMC5 KO), each the
        # ratio of mean spike-in-normalised TSS coverage to its own control
        T_P=T_P,
        P=P,
        P_fit=P_fit,
    )


def e_weights(data, w_armc5=1.0):
    """Per-point weights for the E block.

    Every imaging point carries weight 1, which is the scheme used for the
    reported fit.  `w_armc5` applies a flat multiplier to the five ARMC5 points
    and exists only so that the weighting-sensitivity analysis can vary it.
    """
    w = np.ones(len(data["T_E"]))
    if w_armc5 != 1.0:
        w[~np.isnan(data["E_sd"])] = w_armc5
    return w


# ----------------------------------------------------------------------------
# Cost
# ----------------------------------------------------------------------------
def cost(
    params,
    data,
    w_P=3.0,
    w_armc5=1.0,
    w_E=1.0,
    pt_fraction=0.8,
    constraint_weight=100.0,
    full=False,
):
    """Weighted least-squares cost (same form as the original notebook).

    `pt_fraction` f enforces flux(P->N) = r * flux(P->E) at T = 1 with
    r = f/(1-f); f = 0.8 -> r = 4 reproduces the published fit.
    """
    params = np.asarray(params, float)
    if np.any(params <= 0) or np.any(params > np.array([b[1] for b in BOUNDS])):
        return 1e6
    k_pn, Vmax_pe, Km_pe, Vmax_np, Km_np = params

    base = steady_state(1.0, params)
    if not np.all(np.isfinite(base)) or base[2] <= 0 or base[1] <= 0:
        return 1e6
    N0, P0, E0 = base

    mE = data.get("E_fit", np.ones(len(data["T_E"]), bool))
    mP = data.get("P_fit", np.ones(len(data["T_P"]), bool))
    wE = (w_E * e_weights(data, w_armc5))[mE]
    ssE = np.array([steady_state(T, params)[2] for T in data["T_E"][mE]]) / E0
    ssB = np.array([np.sum(steady_state(T, params)[1:]) / T for T in data["T_B"]])
    ssP = np.array([steady_state(T, params)[1] for T in data["T_P"][mP]]) / P0
    if not (
        np.all(np.isfinite(ssE))
        and np.all(np.isfinite(ssB))
        and np.all(np.isfinite(ssP))
    ):
        return 1e6

    rE = np.sum(wE * (ssE - data["E"][mE]) ** 2)
    rB = np.sum((ssB - data["B"]) ** 2)
    rP = w_P * np.sum((ssP - data["P"][mP]) ** 2)
    sse = rE + rB + rP
    n = wE.sum() + len(ssB) + w_P * len(ssP)

    r = pt_fraction / (1.0 - pt_fraction)
    flux_pn = k_pn * P0
    flux_pe = Vmax_pe * P0 / (Km_pe + P0)
    pen = constraint_weight * (flux_pn - r * flux_pe) ** 2

    if full:
        return dict(
            cost=(sse + pen) / n,
            sse=sse,
            n_eff=n,
            penalty=pen,
            resid_E=ssE - data["E"][mE],
            resid_B=ssB - data["B"],
            resid_P=ssP - data["P"][mP],
            w_E=wE,
            pt_achieved=flux_pn / (flux_pn + flux_pe),
        )
    return (sse + pen) / n


# ----------------------------------------------------------------------------
# Fitting
# ----------------------------------------------------------------------------
def fit(data, n_starts=100, seed=42, x0=None, fixed=None, **kw):
    """Multi-start L-BFGS-B fit.  `fixed` = {index: value} freezes parameters."""
    fixed = fixed or {}
    free = [i for i in range(5) if i not in fixed]

    def expand(xf):
        x = np.empty(5)
        for i, v in fixed.items():
            x[i] = v
        x[free] = xf
        return x

    obj = lambda xf: cost(expand(xf), data, **kw)

    rng = np.random.default_rng(seed)
    starts = []
    if x0 is not None:
        starts.append(np.asarray(x0, float)[free])
    for _ in range(n_starts):
        starts.append(
            np.array(
                [
                    10 ** rng.uniform(np.log10(BOUNDS[i][0]), np.log10(BOUNDS[i][1]))
                    for i in free
                ]
            )
        )
    best, allres = None, []
    for s in starts:
        try:
            r = minimize(
                obj,
                s,
                method="L-BFGS-B",
                bounds=[BOUNDS[i] for i in free],
                options={"maxiter": 2000},
            )
        except Exception:
            continue
        if np.isfinite(r.fun):
            allres.append((r.fun, expand(r.x)))
            if best is None or r.fun < best[0]:
                best = (r.fun, expand(r.x))
    allres.sort(key=lambda t: t[0])
    return dict(cost=best[0], params=best[1], all=allres)


def summarise(params, data, **kw):
    """Best-fit summary: parameters, pool sizes, saturations, buffering range."""
    N0, P0, E0 = steady_state(1.0, params)
    s = {
        f"s_{n}_T{t}": v
        for t in (0.5, 0.7, 1.0)
        for n, v in zip(("np", "pe"), saturation(t, params))
    }
    out = dict(zip(PARAM_NAMES, params))
    out.update(N0=N0, P0=P0, E0=E0, **s)
    out["T_low"], out["T_high"] = buffering_range(params)
    out["S_T1"] = log_sensitivity(params)
    out["cost"] = cost(params, data, **kw)
    return out


def buffering_range(params, tol=0.05, lo=0.15, hi=3.0):
    """Interval of T over which E(T)/E(1) stays within +/- tol of 1.

    Returns NaN for an endpoint if E never crosses that bound inside
    [lo, hi] (typically the upper bound: E asymptotes to Vmax_PE/k_EN, which
    for the best fit lies below +5%).
    """
    E0 = steady_state(1.0, params)[2]
    g = lambda T, target: steady_state(T, params)[2] / E0 - target

    def solve(a, b, target):
        try:
            if g(a, target) * g(b, target) > 0:
                return np.nan
            return brentq(g, a, b, args=(target,), xtol=1e-6)
        except Exception:
            return np.nan

    return solve(lo, 1.0, 1 - tol), solve(1.0, hi, 1 + tol)


def log_sensitivity(params, T=1.0, h=1e-3):
    """S = dln E* / dln T at T.  S ~ 0 = perfectly buffered, S = 1 = no buffering."""
    a = steady_state(T * (1 - h), params)[2]
    b = steady_state(T * (1 + h), params)[2]
    return (np.log(b) - np.log(a)) / (np.log(1 + h) - np.log(1 - h))
