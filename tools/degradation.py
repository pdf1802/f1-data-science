"""
Tyre degradation as a two-level (hierarchical) model — V3.2.

Why. Module 1 V3.1 had the circuit as a feature, but shallow regularised trees
COMPRESS interactions: its implied degradation ranged 0.034–0.069 s/lap across
circuits while the measured one ranges 0.025–0.127 (Bahrain 1.8× too gentle).
That is the root of the one-stop bias in every strategy tool. And degradation is
not a property of the circuit alone: races at the same circuit differ almost as
much (sd 0.024 s/lap) as circuits do (sd 0.026).

Level 1 — between races (this module). Each race has a degradation rate γ:

    LapTime = α[driver] − φ·LapNumber + β[compound] + γ·TyreLife + ε       (per race)

    φ  fuel burn + track evolution per lap. Identified because tyre age resets at
       a stop while LapNumber keeps going.
    γ  degradation, s per lap of tyre age (all compounds pooled).

and the γ of a race is drawn around its circuit, the circuits around the field:

    γ[race] ~ N(μ[circuit], race_sd²)        μ[circuit] ~ N(global, circuit_sd²)

Fitted by maximum likelihood (random effects). A circuit's estimate is SHRUNK to
the global mean in proportion to how little data it has: two noisy races move it
less than four consistent ones (empirical Bayes). The model gives, for any race,
a prediction AND its uncertainty — before the race (circuit prior) and during it
(`live_estimate`: the prior updated with the laps run so far).

Level 2 — within a race (tools/training.py). Module 1 learns
LapDelta = f(compound, tyre age, circuit, γ, γ·age) with γ known, so at prediction
time it is handed the best current estimate of γ instead of having to guess it
from a circuit label.

Measured on races never seen (leave-one-race-out, 32 well-measured races):
    calibration of the implied degradation   V3.1 slope 1.35 → V3.2 0.90   (1 = perfect)
    Bahrain implied degradation              0.069 → 0.095 s/lap           (real 0.127)
    if the day's γ were known (the live estimate's ceiling): error 0.023 → 0.014 s/lap
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tools.shared import DEG_RANGE, circuit_name

DRY = ("SOFT", "MEDIUM", "HARD")
CLEAN_AIR_GAP = 2.0
# live_estimate: how much less than "independent laps" the race's laps are worth, as (before any
# car has stopped, once LIVE_STINT2_SHARE of the clean laps are on a second set), and how far a
# race's fuel + track-evolution slope φ may sit from the usual one. Calibrated leave-one-race-out.
LIVE_DESIGN_EFFECT = (12.0, 2.0)
LIVE_STINT2_SHARE = 0.4
LIVE_FUEL_SD = 0.012            # s/lap
MIN_LIVE_LAPS = 40              # clean laps before the in-race estimate moves off the prior


# ── Level 1a: one race ───────────────────────────────────────────────────────
def stint_laps(laps: pd.DataFrame, clean_air_gap: float | None = CLEAN_AIR_GAP,
               max_tyre_age: int = 35) -> pd.DataFrame:
    """
    The laps that show tyre behaviour: green flag, dry compound, no in/out laps, not lap 1,
    in clean air (traffic hides pace), within 3 s of the driver's median (no mistakes).
    Adds LT (s), race ("2023 Bahrain Grand Prix"), key (race + driver) and circuit.
    """
    from tools.training import _seconds, add_gap_ahead
    d = add_gap_ahead(laps) if clean_air_gap is not None else laps.copy()
    d["LT"] = _seconds(d["LapTime"])
    keep = (d["Compound"].isin(DRY) & (d["TrackStatus"].astype(str) == "1")
            & d["PitInTime"].isna() & d["PitOutTime"].isna() & (d["LapNumber"] > 1)
            & d["TyreLife"].between(2, max_tyre_age) & d["LT"].between(60, 150))
    if clean_air_gap is not None:
        keep &= ((d["PosStart"] == 1) | (d["GapAheadStart"] > clean_air_gap)).fillna(False).astype(bool)
    d = d[keep].copy()
    d["race"] = d["Season"].astype(str) + " " + d["Race"].astype(str)
    d["key"] = d["race"] + " " + d["Driver"].astype(str)
    d = d[(d["LT"] - d.groupby("key")["LT"].transform("median")).abs() < 3]
    d["circuit"] = d["Race"].map(circuit_name)
    return d


def _design(g: pd.DataFrame):
    """Driver-demeaned design matrix [lap, MEDIUM, HARD, age] and target for one race."""
    X = pd.DataFrame({"lap": g["LapNumber"], "M": g["Compound"] == "MEDIUM", "H": g["Compound"] == "HARD",
                      "age": g["TyreLife"]}).astype(float)
    y = g["LT"].astype(float)
    Xd = (X - X.groupby(g["key"]).transform("mean")).to_numpy()
    yd = (y - y.groupby(g["key"]).transform("mean")).to_numpy()
    return Xd, yd


def _fit_race(g: pd.DataFrame) -> tuple[float, float]:
    """(degradation γ, fuel+evolution φ) of one race by least squares with driver fixed effects."""
    Xd, yd = _design(g)
    beta, *_ = np.linalg.lstsq(Xd, yd, rcond=None)
    return float(beta[3]), float(-beta[0])


def race_physics(laps: pd.DataFrame, n_boot: int = 200, seed: int = 0) -> pd.DataFrame:
    """
    One row per race: deg (γ), fuel (φ) and their standard errors from a bootstrap over
    DRIVERS (laps of one driver are not independent; drivers are the unit to resample).
    A race where everyone stopped at the same time has age ≈ lap − constant, so γ and φ
    cannot be told apart: the bootstrap shows it as a large standard error.
    """
    d = stint_laps(laps)
    rng = np.random.default_rng(seed)
    rows = []
    for race, g in d.groupby("race"):
        if g["Driver"].nunique() < 4 or len(g) < 80:
            continue
        deg, fuel = _fit_race(g)
        drivers = g["Driver"].unique()
        parts = {p: g[g["Driver"] == p] for p in drivers}
        boots = []
        for _ in range(n_boot):
            pick = rng.choice(drivers, len(drivers))
            boots.append(_fit_race(pd.concat([parts[p].assign(key=str(i)) for i, p in enumerate(pick)])))
        boots = np.array(boots)
        rows.append({"race": race, "circuit": g["circuit"].iloc[0], "n_laps": len(g), "deg": deg,
                     "deg_se": float(boots[:, 0].std()), "fuel": fuel, "fuel_se": float(boots[:, 1].std())})
    return pd.DataFrame(rows).set_index("race")


# ── Level 1b: races within circuits (random effects) ─────────────────────────
def _fit_random_effects(P: pd.DataFrame) -> tuple[float, float, float]:
    """ML estimates of (global mean, between-circuit variance, race-to-race variance)."""
    from scipy.optimize import minimize
    groups = [(g["deg"].to_numpy(), g["deg_se"].to_numpy() ** 2) for _, g in P.groupby("circuit")]

    def nll(theta):
        mu, tau2, sig2 = theta[0], np.exp(theta[1]), np.exp(theta[2])
        out = 0.0
        for d, s2 in groups:                       # races of a circuit share its effect: cov = tau2
            S = np.diag(sig2 + s2) + tau2
            r = d - mu
            out += 0.5 * (np.linalg.slogdet(S)[1] + r @ np.linalg.solve(S, r))
        return out

    v = max(float(P["deg"].var()), 1e-6)
    res = minimize(nll, [float(P["deg"].mean()), np.log(v / 2), np.log(v / 2)], method="Nelder-Mead",
                   options={"xatol": 1e-7, "fatol": 1e-9, "maxiter": 5000})
    return float(res.x[0]), float(np.exp(res.x[1])), float(np.exp(res.x[2]))


def fit_prior(P: pd.DataFrame) -> dict:
    """
    race_physics table → the degradation model as a plain dict (saved as degradation.json):

        global, circuit_sd, race_sd      the random-effects fit
        circuits {name: γ}               each circuit's estimate, shrunk to `global`
        races {race: γ}                  each race's own γ, shrunk to its circuit by its precision
                                         (the value Module 1 is trained with)
        fuel {global, races {race: φ}}   fuel + track evolution per race, shrunk the same way
    """
    if len(P) < 3 or P["circuit"].nunique() < 2:                 # not enough races to split the variance
        mu = float(np.average(P["deg"], weights=1 / P["deg_se"].clip(lower=1e-3) ** 2)) if len(P) else 0.058
        tau2 = sig2 = 0.025 ** 2
    else:
        mu, tau2, sig2 = _fit_random_effects(P)
    circuits = {}
    for c, g in P.groupby("circuit"):
        w = 1.0 / (sig2 + g["deg_se"].to_numpy() ** 2)
        m, V = float((w * g["deg"]).sum() / w.sum()), float(1.0 / w.sum())
        circuits[c] = float(np.clip((m / V + mu / tau2) / (1 / V + 1 / tau2), *DEG_RANGE))
    s2 = P["deg_se"].clip(lower=1e-4) ** 2
    prior = P["circuit"].map(circuits)
    races = ((P["deg"] / s2 + prior / sig2) / (1 / s2 + 1 / sig2)).clip(*DEG_RANGE)

    f2 = P["fuel_se"].clip(lower=1e-4) ** 2
    f_mu = float(np.average(P["fuel"], weights=1 / f2)) if len(P) else 0.055
    f_tau2 = max(float(P["fuel"].var() - f2.mean()) if len(P) > 1 else 0.0, 1e-5)
    fuel = (P["fuel"] / f2 + f_mu / f_tau2) / (1 / f2 + 1 / f_tau2)
    return {"global": float(np.clip(mu, *DEG_RANGE)), "circuit_sd": float(np.sqrt(tau2)),
            "race_sd": float(np.sqrt(sig2)), "circuits": circuits,
            "races": {r: float(v) for r, v in races.items()},
            "fuel": {"global": f_mu, "sd": float(np.sqrt(f_tau2)), "races": {r: float(v) for r, v in fuel.items()}},
            "n_races": int(len(P))}


def fuel_map(prior: dict) -> dict:
    """{race: φ, "_default": global φ} for training.build_lap_dataset(fuel_s_per_lap=…)."""
    return {**prior["fuel"]["races"], "_default": prior["fuel"]["global"]}


# ── During a race: update the prior with the laps run so far ─────────────────
def live_estimate(laps: pd.DataFrame, prior_mean: float, prior_sd: float,
                  fuel_mean: float = 0.055, fuel_sd: float = LIVE_FUEL_SD,
                  clean_air_gap: float | None = CLEAN_AIR_GAP, design_effect: float | None = None) -> dict:
    """
    Bayesian update of this race's degradation γ with the laps completed so far.

    Same regression as _fit_race, with Gaussian priors on γ (the circuit prior) and φ
    (fuel + evolution). Early in a race every car is on its first set: tyre age and lap
    number move together, the data alone cannot separate γ from φ, and the posterior stays
    near the prior. As cars stop, the data takes over. Closed form (ridge regression with
    a prior mean); the noise variance is the residual variance of the unpenalised fit, inflated
    by a design effect (below).

    Returns {mean, sd, n_laps, prior_mean, prior_sd}.
    """
    out = {"mean": float(prior_mean), "sd": float(prior_sd), "n_laps": 0,
           "prior_mean": float(prior_mean), "prior_sd": float(prior_sd)}
    try:
        g = stint_laps(laps, clean_air_gap=clean_air_gap)
    except Exception:                                  # incomplete live data: keep the prior
        return out
    out["n_laps"] = int(len(g))
    if len(g) < MIN_LIVE_LAPS or g["key"].nunique() < 3:
        return out
    Xd, yd = _design(g)
    keep = Xd.std(axis=0) > 1e-9                       # a compound nobody has used yet
    keep[[0, 3]] = True
    Xd = Xd[:, keep]
    k = Xd.shape[1]
    beta_ols, *_ = np.linalg.lstsq(Xd, yd, rcond=None)
    dof = max(len(yd) - g["key"].nunique() - k, 1)
    s2 = max(float(((yd - Xd @ beta_ols) ** 2).sum() / dof), 0.05 ** 2)
    m0 = np.zeros(k)                                   # prior means: [−φ, compounds…, γ]
    prec = np.full(k, 1 / 2.0 ** 2)                    # compounds: almost flat (sd 2 s)
    m0[0], prec[0] = -fuel_mean, 1 / fuel_sd ** 2
    m0[-1], prec[-1] = prior_mean, 1 / prior_sd ** 2
    # How much to trust the laps. Two things make N laps worth less than N independent ones:
    # laps of one driver are correlated, and — the big one — until cars have stopped, every tyre
    # ages one lap per lap: wear (γ) and fuel burn (φ) move together and only their difference
    # shows. So the noise variance is inflated by a design effect that starts high and comes down
    # as second stints appear (tyre age resets, the lap number does not). The two ends were set so
    # that the 80 % interval holds the race's final γ ~80 % of the time at every stage of the
    # race, leave one race out (see the degradation_by_circuit notebook).
    if design_effect is None:
        s_ = g.sort_values(["key", "LapNumber"])
        new_set = s_.groupby("key")["TyreLife"].diff() < 0                # the tyre got younger: a stop
        on_second = float((new_set.groupby(s_["key"]).cumsum() > 0).mean())
        hi, lo = LIVE_DESIGN_EFFECT
        deff = hi - (hi - lo) * min(on_second / LIVE_STINT2_SHARE, 1.0)
    else:
        deff = design_effect
    s2 = s2 * deff
    A = Xd.T @ Xd / s2 + np.diag(prec)
    cov = np.linalg.inv(A)
    post = cov @ (Xd.T @ yd / s2 + prec * m0)
    sd = float(np.sqrt(min(cov[-1, -1], prior_sd ** 2)))
    out.update(mean=float(np.clip(post[-1], *DEG_RANGE)), sd=sd)
    return out
