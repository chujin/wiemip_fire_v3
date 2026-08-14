## Fire explicit v5 + time-varying FRI
##
## Same ML candidate burns as fire_explicit_v5.py (1pctCO2 monthly pred_prob,
## ≤1 burn/pixel/year), then a time-varying fire-return interval so annual
## burned pixels rise linearly:
##
##   ~200 pixels/year in 1850  →  ~1000 pixels/year in 2000
##
## Why v5's constant FRI=20 does not do this
## -----------------------------------------
## v5 burn-in draws ONE random stable month per year, so preindustrial fires
## are too rare, delta is too large, and 1850 keeps ~1000 candidates (a spike).
## After that, constant FRI=20 settles near ~400–500 burns/year, not 1000.
##
## What this script changes
## ------------------------
## 1. Burn-in uses a full 12-month stable year (any-month fire, then FRI),
##    so 1850 starts near the preindustrial rate (~200/year at FRI≈40).
## 2. Transient FRI is time-varying. Default ADAPTIVE_FRI=True: each year
##    choose integer L in [FRI_MIN, FRI_MAX] so that
##        count(candidates with delta >= L) ≈ linear target N(year).
##    Burns are still kept iff delta >= L(year) — a recurrence limit, not a
##    random quota. Cleared candidates get sign-flipped exp_pred_prob.
##
## Reuses {mod}_burn_v5_events_candidates.csv when present (same locations
## and DOYs as v5). Outputs do not overwrite v5 files:
##   explicit-fire_{mod}_v5_FRI.nc
##   explicit-fire_{mod}_v5_FRI_new.nc
##
## Edit CONFIG, then:
##   python fire_explicit_v5_FRI.py

import os
import warnings

import numpy as np
import pandas as pd
import xarray as xr
from cftime import DatetimeNoLeap

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIG
# =============================================================================
indir = "/home/ejafarov/wiemip_fire_v3/input/1pctCO2FireMLRawInput"
# mod = "GFDL-ESM4"
# mod = "IPSL-CM6A-LR"
mod = "UKESM1-0-LL"

outtemdir = "/home/ejafarov/wiemip_fire_v3/output"
climatedir = "/home/ejafarov/wiemip_fire_v3/input"
figdir = "/home/ejafarov/wiemip_fire_v3/figures"

RANDOM_SEED = 42
REUSE_CANDIDATE_CSV = True  # use v5 candidate events if the CSV exists

# Linear burned-pixel target
TARGET_BURNS_1850 = 200
TARGET_BURNS_2000 = 1000

# Time-varying FRI
# Burn-in FRI≈40 yields ~200 fires/year on the 12-month stable climatology.
FRI_BURNIN = 40
FRI_MIN = 5
FRI_MAX = 50
ADAPTIVE_FRI = True  # False → linear FRI_BURNIN → FRI_END every year
FRI_END = 8  # used when ADAPTIVE_FRI is False
BurnInYears = 400

stable_prob_file = os.path.join(
    indir, "stable_monthly_prob_Native_combined_rescaled_v2.nc"
)
# =============================================================================

if mod == "stable":
    raise SystemExit(
        "fire_explicit_v5_FRI.py is for GCM cases only (UKESM/IPSL/GFDL).\n"
        "Use fire_explicit_v3.py or fire_explicit_v4.py for stable."
    )

if RANDOM_SEED is not None:
    np.random.seed(RANDOM_SEED)

os.makedirs(outtemdir, exist_ok=True)
os.makedirs(figdir, exist_ok=True)

infile = os.path.join(
    indir, f"{mod}_monthly_prob_Native_combined_rescaled_v2.nc"
)
climate_file = os.path.join(climatedir, f"climate_{mod}.nc")
csv_path = os.path.join(indir, mod + "_burn_v5_events_candidates.csv")
out_native = os.path.join(outtemdir, f"explicit-fire_{mod}_v5_FRI.nc")
out_new = os.path.join(outtemdir, f"explicit-fire_{mod}_v5_FRI_new.nc")

if not os.path.isfile(infile):
    raise FileNotFoundError(f"Missing ML input: {infile}")
if not os.path.isfile(climate_file):
    raise FileNotFoundError(f"Missing climate template: {climate_file}")
if not os.path.isfile(stable_prob_file):
    raise FileNotFoundError(
        f"Missing stable prob file for burn-in: {stable_prob_file}"
    )


MONTH_LENGTHS_NOLEAP = (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)
MONTH_START_DOY_NOLEAP = (
    0,
    1,
    32,
    60,
    91,
    121,
    152,
    182,
    213,
    244,
    274,
    305,
    335,
)
MONTH_LENGTHS_ARR = np.asarray(MONTH_LENGTHS_NOLEAP, dtype=np.int16)

INT_FIRE_VARS = (
    "exp_burn_mask",
    "exp_jday_of_burn",
    "exp_fire_severity",
    "exp_area_of_burn",
)


def noleap_doy_vec(months: np.ndarray, days: np.ndarray) -> np.ndarray:
    months = np.asarray(months, dtype=np.int16)
    days = np.asarray(days, dtype=np.int16)
    starts = np.asarray(MONTH_START_DOY_NOLEAP[1:], dtype=np.int16)
    return starts[months - 1] + days - 1


def drop_fillvalue_attrs(ds: xr.Dataset) -> xr.Dataset:
    ds = ds.copy()
    for name in ds.data_vars:
        if "_FillValue" in ds[name].attrs:
            ds[name].attrs = {
                k: v for k, v in ds[name].attrs.items() if k != "_FillValue"
            }
    return ds


def cast_tem_fire_dtypes(ds: xr.Dataset) -> xr.Dataset:
    ds = ds.copy(deep=True)
    for name in INT_FIRE_VARS:
        if name not in ds:
            continue
        vals = np.asarray(ds[name].values, dtype=np.float64)
        vals = np.nan_to_num(vals, nan=0.0)
        vals[vals < -900] = 0.0
        ds[name].values = vals.astype(np.int32)
    if "exp_pred_prob" in ds:
        vals = np.asarray(ds["exp_pred_prob"].values, dtype=np.float64)
        vals = np.nan_to_num(vals, nan=0.0)
        vals[vals < -900] = 0.0
        ds["exp_pred_prob"].values = vals.astype(np.float64)
    for name in ("lat", "lon"):
        if name in ds:
            ds[name].values = np.asarray(ds[name].values, dtype=np.float64)
    return ds


def pad_missing_axis(ds: xr.Dataset, dim: str, missing_coords: list) -> xr.Dataset:
    if not missing_coords:
        return ds
    n = len(missing_coords)
    empty_vars = {}
    for name, da in ds.data_vars.items():
        shape = tuple(n if d == dim else ds.sizes[d] for d in da.dims)
        if name in INT_FIRE_VARS:
            arr = np.zeros(shape, dtype=np.int32)
        elif name in ("exp_pred_prob", "pred_prob"):
            arr = np.zeros(shape, dtype=np.float64)
        else:
            arr = np.full(shape, np.nan, dtype=np.float64)
        empty_vars[name] = (da.dims, arr)

    coords = {dim: missing_coords}
    for d in ds.dims:
        if d != dim:
            coords[d] = ds.coords[d]
    empty = xr.Dataset(empty_vars, coords=coords)
    out = xr.concat([ds, empty], dim=dim).sortby(dim)
    return cast_tem_fire_dtypes(out)


def assert_jday_in_range(ds: xr.Dataset, label: str) -> None:
    if "exp_jday_of_burn" not in ds:
        return
    j = np.asarray(ds["exp_jday_of_burn"].values, dtype=np.float64)
    j = j[np.isfinite(j)]
    if j.size == 0:
        return
    jmin, jmax = int(j.min()), int(j.max())
    if jmin < 0 or jmax > 365:
        raise RuntimeError(
            f"{label}: exp_jday_of_burn out of range [{jmin}, {jmax}] "
            f"(expected 0..365)"
        )
    print(f"  {label}: exp_jday_of_burn range [{jmin}, {jmax}]")


def _time_year_month(time_vals) -> tuple[np.ndarray, np.ndarray]:
    years = np.empty(len(time_vals), dtype=np.int32)
    months = np.empty(len(time_vals), dtype=np.int16)
    for i, t in enumerate(time_vals):
        years[i] = int(t.year)
        months[i] = int(t.month)
    return years, months


def linear_target(years: np.ndarray, n0: float, n1: float) -> np.ndarray:
    years = np.asarray(years, dtype=float)
    y0, y1 = float(years[0]), float(years[-1])
    return n0 + (n1 - n0) * (years - y0) / max(y1 - y0, 1.0)


def choose_fri_for_year(delta_cand: np.ndarray, target_n: float, lmin: int, lmax: int) -> int:
    """Integer L in [lmin, lmax] whose keep-count is closest to target_n."""
    best_l = int(lmax)
    best_err = 1e18
    d = np.asarray(delta_cand)
    for lim in range(int(lmin), int(lmax) + 1):
        k = int((d >= lim).sum())
        err = abs(k - target_n)
        if err < best_err:
            best_err = err
            best_l = lim
    return best_l


def apply_recurrence_time_varying_fri(
    cand_mask: np.ndarray,
    cand_jday: np.ndarray,
    cand_prob: np.ndarray,
    active: np.ndarray,
    stable_pp: np.ndarray,
    year_list: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """12-month stable burn-in, then time-varying FRI on annual candidates.

    A burn is kept iff years-since-last-fire >= L(year).
    """
    n_years, n_lat, n_lon = cand_mask.shape
    n_stable_t = stable_pp.shape[0]
    if stable_pp.shape[1:] != (n_lat, n_lon):
        raise SystemExit(
            f"Stable prob grid {stable_pp.shape[1:]} != GCM grid {(n_lat, n_lon)}"
        )
    n_sy = n_stable_t // 12
    if n_sy < 1:
        raise SystemExit("Stable pred_prob needs at least 12 months for burn-in")

    flat_stable = np.ascontiguousarray(
        np.clip(np.nan_to_num(stable_pp, nan=0.0), 0.0, 1.0).reshape(n_stable_t, -1)
    )
    n_pix = flat_stable.shape[1]
    active_flat = active.reshape(-1)
    active_idx = np.where(active_flat)[0]
    n_act = int(active_idx.size)
    if n_act == 0:
        raise SystemExit("No active pixels for FRI burn-in")

    target = linear_target(year_list, TARGET_BURNS_1850, TARGET_BURNS_2000)
    tfrac = (year_list.astype(float) - float(year_list[0])) / max(
        float(year_list[-1] - year_list[0]), 1.0
    )
    L_linear = np.clip(
        np.rint(FRI_BURNIN + (FRI_END - FRI_BURNIN) * tfrac),
        FRI_MIN,
        FRI_MAX,
    ).astype(np.int32)

    print(
        f"  burn-in: {BurnInYears} years, 12-month stable years "
        f"(n_stable_years={n_sy}, FRI_BURNIN={FRI_BURNIN}, n_active={n_act:,}) ..."
    )
    delta_act = np.full(n_act, FRI_BURNIN, dtype=np.int32)
    n_burnin_kept = 0
    for t in range(BurnInYears):
        yi = int(rng.integers(0, n_sy))
        block = flat_stable[yi * 12 : (yi + 1) * 12][:, active_idx]
        fire_try = (rng.random(block.shape) < block).any(axis=0)
        keep = fire_try & (delta_act >= FRI_BURNIN)
        n_burnin_kept += int(keep.sum())
        delta_act = np.where(keep, 0, delta_act) + 1
        if (t + 1) % 100 == 0 or t + 1 == BurnInYears:
            print(f"    burn-in year {t + 1}/{BurnInYears}")

    print(
        f"  burn-in done: kept fires={n_burnin_kept:,} "
        f"({n_burnin_kept / BurnInYears:.1f}/year); "
        f"delta median={int(np.median(delta_act))} mean={float(delta_act.mean()):.1f}"
    )

    delta = np.full(n_pix, FRI_BURNIN, dtype=np.int32)
    delta[active_idx] = delta_act
    delta_start = delta.copy()

    out_mask = cand_mask.copy()
    out_jday = cand_jday.copy()
    out_prob = cand_prob.copy()
    L_used = np.empty(n_years, dtype=np.int32)
    annual = np.empty(n_years, dtype=np.int32)
    n_cand = int((cand_mask == 1).sum())
    n_cleared = 0
    n_kept = 0

    mode = "adaptive" if ADAPTIVE_FRI else "linear"
    print(f"  applying time-varying FRI ({mode}) to {n_years} scenario years ...")
    for y in range(n_years):
        cand = (out_mask[y].reshape(-1) == 1) & active_flat
        if ADAPTIVE_FRI:
            L = choose_fri_for_year(
                delta[cand], float(target[y]), FRI_MIN, FRI_MAX
            )
        else:
            L = int(L_linear[y])
        L_used[y] = L
        keep = cand & (delta >= L)
        clear = cand & ~keep
        n_keep_y = int(keep.sum())
        annual[y] = n_keep_y
        n_kept += n_keep_y
        n_cleared += int(clear.sum())

        if clear.any():
            clear2 = clear.reshape(n_lat, n_lon)
            out_mask[y][clear2] = 0
            out_jday[y][clear2] = 0
            out_prob[y][clear2] = -1.0 * out_prob[y][clear2]

        delta = np.where(keep, 0, delta) + 1

    rmse = float(np.sqrt(np.mean((annual.astype(float) - target) ** 2)))
    stats = {
        "burn_in_years": BurnInYears,
        "fri_burnin": FRI_BURNIN,
        "fri_min": FRI_MIN,
        "fri_max": FRI_MAX,
        "adaptive": bool(ADAPTIVE_FRI),
        "burnin_kept_fires": n_burnin_kept,
        "burnin_mean_annual": float(n_burnin_kept / BurnInYears),
        "candidate_burns": n_cand,
        "kept_burns": n_kept,
        "cleared_burns": n_cleared,
        "rmse_vs_linear_target": rmse,
        "annual_1850": int(annual[0]),
        "annual_2000": int(annual[-1]),
        "annual_mean": float(annual.mean()),
        "fri_1850": int(L_used[0]),
        "fri_2000": int(L_used[-1]),
        "delta_start_median_active": int(np.median(delta_start[active_flat])),
        "delta_start_mean_active": float(delta_start[active_flat].mean()),
    }
    print(
        f"  time-varying FRI: candidates={n_cand:,} kept={n_kept:,} "
        f"cleared={n_cleared:,} ({100.0 * n_cleared / max(n_cand, 1):.1f}%)"
    )
    print(
        f"  annual burns 1850={annual[0]}  2000={annual[-1]}  "
        f"mean={annual.mean():.1f}  RMSE vs target={rmse:.1f}"
    )
    print(f"  FRI(year) {L_used[0]} → {L_used[-1]}  (median {int(np.median(L_used))})")
    return out_mask, out_jday, out_prob, L_used, stats


def draw_candidates(lat, lon, time_vals, pred_prob, rng) -> pd.DataFrame:
    """Bernoulli monthly draws → ≤1 burn per pixel-year (same as v5)."""
    n_time, n_lat, n_lon = pred_prob.shape
    years_t, months_t = _time_year_month(time_vals)
    active = np.sum(pred_prob, axis=0) > 0.0
    print(f"  active pixels (sum pred_prob > 0): {int(active.sum())} / {n_lat * n_lon}")
    print("Drawing monthly burns (vectorized) ...")
    chunk = 120
    final_choice = np.zeros((n_time, n_lat, n_lon), dtype=np.bool_)
    for t0 in range(0, n_time, chunk):
        t1 = min(t0 + chunk, n_time)
        block = pred_prob[t0:t1]
        draws = rng.random(block.shape, dtype=np.float32) < block
        draws &= active[np.newaxis, :, :]
        final_choice[t0:t1] = draws
        print(f"  months {t0}:{t1} done")
    n_month_burns = int(np.count_nonzero(final_choice))
    print(f"  monthly burn flags: {n_month_burns:,}")
    if n_month_burns == 0:
        raise RuntimeError("No burns drawn; check pred_prob input")

    ti, yi, xi = np.nonzero(final_choice)
    probs_at_burn = pred_prob[ti, yi, xi].astype(np.float64, copy=False)
    del final_choice
    burned = pd.DataFrame(
        {
            "t_idx": ti.astype(np.int32),
            "y_idx": yi.astype(np.int32),
            "x_idx": xi.astype(np.int32),
            "year": years_t[ti],
            "month": months_t[ti],
            "pred_prob": probs_at_burn,
            "lat": lat[yi],
            "lon": lon[xi],
        }
    )
    del ti, yi, xi, probs_at_burn
    burn = (
        burned.groupby(["y_idx", "x_idx", "year"], sort=False)
        .sample(n=1, random_state=RANDOM_SEED)
        .reset_index(drop=True)
    )
    del burned
    mlen = MONTH_LENGTHS_ARR[burn["month"].to_numpy(dtype=np.int16) - 1]
    dom = rng.integers(1, mlen + 1, size=len(burn), dtype=np.int16)
    burn["day_of_year"] = noleap_doy_vec(burn["month"].to_numpy(), dom)
    return burn[["y_idx", "x_idx", "year", "month", "pred_prob", "lat", "lon", "day_of_year"]]


def plot_annual_vs_target(year_list, annual, L_used, target, stats, out_png):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    years = np.asarray(year_list)
    sm = (
        pd.Series(annual.astype(float))
        .rolling(11, center=True, min_periods=6)
        .mean()
        .to_numpy()
    )

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, constrained_layout=True)
    ax = axes[0]
    ax.plot(years, annual, color="#c23b22", lw=1.2, alpha=0.85, label="burned pixels / year")
    ax.plot(years, sm, color="#1a1a1a", lw=1.8, label="11-yr mean")
    ax.plot(years, target, color="#2c7bb6", lw=1.8, ls="--", label="linear target 200→1000")
    ax.set_ylabel("Burned pixels / year")
    ax.set_title(
        f"{mod} v5 time-varying FRI   "
        f"1850={int(annual[0])}  2000={int(annual[-1])}  "
        f"mean={annual.mean():.0f}  RMSE={stats['rmse_vs_linear_target']:.1f}"
    )
    ax.legend(frameon=False, ncol=3, fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(int(years[0]), int(years[-1]))

    ax = axes[1]
    ax.plot(years, L_used, color="#4d4d4d", lw=1.6, label="FRI L(year)  (keep if delta ≥ L)")
    ax.set_xlabel("Year")
    ax.set_ylabel("Recurrence limit (years)")
    ax.set_title("Time-varying fire-return interval")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, max(int(L_used.max()) + 2, FRI_BURNIN + 2))

    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_png}")


# -----------------------------------------------------------------------------
# 1) Candidates (reuse v5 CSV when possible)
# -----------------------------------------------------------------------------
print(f"mod={mod}")
print(f"Reading grid from {infile} ...")
with xr.open_dataset(infile, decode_times=True) as ds_in:
    if "pred_prob" not in ds_in:
        raise SystemExit(f"pred_prob missing in {infile}")
    lat = np.asarray(ds_in["lat"].values, dtype=np.float64)
    lon = np.asarray(ds_in["lon"].values, dtype=np.float64)
    time_vals = ds_in["time"].values
    need_draws = not (REUSE_CANDIDATE_CSV and os.path.isfile(csv_path))
    pred_prob = None
    if need_draws:
        print(f"Reading pred_prob from {infile} (float32) ...")
        pp = ds_in["pred_prob"]
        if set(pp.dims) >= {"time", "lat", "lon"}:
            pp = pp.transpose("time", "lat", "lon")
        pred_prob = np.nan_to_num(pp.values, nan=0.0).astype(np.float32, copy=False)

n_lat, n_lon = lat.size, lon.size
years_t, months_t = _time_year_month(time_vals)
jan_mask = months_t == 1
if not np.any(jan_mask):
    raise RuntimeError("No January timesteps found — cannot build annual fire file")
year_list = years_t[jan_mask]
n_years = len(year_list)
year_to_idx = {int(y): i for i, y in enumerate(year_list)}
print(f"  annual years: {int(year_list[0])}..{int(year_list[-1])} (n={n_years})")
print(f"  native grid lat={n_lat} lon={n_lon}")

rng = np.random.default_rng(RANDOM_SEED)
if need_draws:
    burn = draw_candidates(lat, lon, time_vals, pred_prob, rng)
    del pred_prob
    burn.to_csv(csv_path, index=False)
    print(f"Wrote sparse candidate CSV {csv_path}")
else:
    print(f"Reusing candidate CSV {csv_path}")
    burn = pd.read_csv(csv_path)
    need = {"y_idx", "x_idx", "year", "pred_prob", "day_of_year"}
    missing = need - set(burn.columns)
    if missing:
        raise SystemExit(f"Candidate CSV missing columns {missing}")

print(f"  candidate annual burn events: {len(burn):,}")

# -----------------------------------------------------------------------------
# Assemble annual native arrays
# -----------------------------------------------------------------------------
print("Assembling annual native fire arrays ...")
exp_burn_mask = np.zeros((n_years, n_lat, n_lon), dtype=np.int32)
exp_jday = np.zeros((n_years, n_lat, n_lon), dtype=np.int32)
exp_sev = np.zeros((n_years, n_lat, n_lon), dtype=np.int32)
exp_area = np.zeros((n_years, n_lat, n_lon), dtype=np.int32)
exp_prob = np.zeros((n_years, n_lat, n_lon), dtype=np.float64)

yi = burn["y_idx"].to_numpy(dtype=np.int32)
xi = burn["x_idx"].to_numpy(dtype=np.int32)
yy = burn["year"].to_numpy(dtype=np.int32)
t_ann = np.array([year_to_idx[int(y)] for y in yy], dtype=np.int32)
exp_burn_mask[t_ann, yi, xi] = 1
exp_jday[t_ann, yi, xi] = burn["day_of_year"].to_numpy(dtype=np.int32)
exp_prob[t_ann, yi, xi] = burn["pred_prob"].to_numpy(dtype=np.float64)
active = exp_burn_mask.any(axis=0)
del burn

n_before = int((exp_burn_mask == 1).sum())
print(f"  candidate annual burns (pre-FRI): {n_before:,}")
print(f"  active pixels (ever a candidate): {int(active.sum()):,}")

print(f"Loading stable probs {stable_prob_file} ...")
with xr.open_dataset(stable_prob_file, decode_times=False) as ds_st:
    spp = ds_st["pred_prob"]
    if set(spp.dims) >= {"time", "lat", "lon"}:
        spp = spp.transpose("time", "lat", "lon")
    stable_pp = np.asarray(spp.values, dtype=np.float32)
    st_lat = np.asarray(ds_st["lat"].values, dtype=np.float64)
    st_lon = np.asarray(ds_st["lon"].values, dtype=np.float64)
if st_lat.shape != lat.shape or st_lon.shape != lon.shape:
    raise SystemExit(
        f"Stable lat/lon shape {st_lat.shape}/{st_lon.shape} != "
        f"GCM {lat.shape}/{lon.shape}"
    )

rng_fri = np.random.default_rng(None if RANDOM_SEED is None else RANDOM_SEED + 101)
exp_burn_mask, exp_jday, exp_prob, L_used, fri_stats = apply_recurrence_time_varying_fri(
    exp_burn_mask,
    exp_jday,
    exp_prob,
    active,
    stable_pp,
    year_list,
    rng_fri,
)
del stable_pp
print(f"  FRI stats: {fri_stats}")

n_after = int((exp_burn_mask == 1).sum())
n_override = int((exp_prob < 0).sum())
print(f"  final annual burns: {n_after:,}")
print(f"  FRI overrides (exp_pred_prob < 0): {n_override:,}")

annual_kept = exp_burn_mask.sum(axis=(1, 2)).astype(np.int32)
target = linear_target(year_list, TARGET_BURNS_1850, TARGET_BURNS_2000)
sched_csv = os.path.join(figdir, f"explicit-fire_{mod}_v5_FRI_schedule.csv")
pd.DataFrame(
    {
        "year": year_list,
        "target_burns": np.rint(target).astype(np.int32),
        "kept_burns": annual_kept,
        "FRI_years": L_used,
    }
).to_csv(sched_csv, index=False)
print(f"Wrote {sched_csv}")

plot_annual_vs_target(
    year_list,
    annual_kept,
    L_used,
    target,
    fri_stats,
    os.path.join(figdir, f"explicit-fire_{mod}_v5_FRI.png"),
)

time_ann = [DatetimeNoLeap(int(y), 1, 1) for y in year_list]
lat2d = np.broadcast_to(lat[:, None], (n_lat, n_lon)).astype(np.float64)
lon2d = np.broadcast_to(lon[None, :], (n_lat, n_lon)).astype(np.float64)
lat_t = np.broadcast_to(lat2d[None, :, :], (n_years, n_lat, n_lon)).copy()
lon_t = np.broadcast_to(lon2d[None, :, :], (n_years, n_lat, n_lon)).copy()

nc = xr.Dataset(
    data_vars={
        "lat": (("time", "Y", "X"), lat_t),
        "lon": (("time", "Y", "X"), lon_t),
        "exp_burn_mask": (("time", "Y", "X"), exp_burn_mask),
        "exp_jday_of_burn": (("time", "Y", "X"), exp_jday),
        "exp_fire_severity": (("time", "Y", "X"), exp_sev),
        "exp_area_of_burn": (("time", "Y", "X"), exp_area),
        "exp_pred_prob": (("time", "Y", "X"), exp_prob),
        "exp_fri_limit": (("time",), L_used.astype(np.int32)),
    },
    coords={
        "time": time_ann,
        "Y": lat.astype(np.float64),
        "X": lon.astype(np.float64),
    },
)
nc["lat"].attrs = {"standard_name": "latitude", "units": "degree_north"}
nc["lon"].attrs = {"standard_name": "longitude", "units": "degree_east"}
nc["Y"].attrs = {
    "standard_name": "projection_y_coordinate",
    "long_name": "y coordinate of projection",
    "units": "m",
}
nc["X"].attrs = {
    "standard_name": "projection_x_coordinate",
    "long_name": "x coordinate of projection",
    "units": "m",
}
nc["exp_burn_mask"].attrs = {
    "standard_name": "exp_burn_mask",
    "units": "",
    "grid_mapping": "albers_conical_equal_area",
}
nc["exp_jday_of_burn"].attrs = {
    "standard_name": "exp_jday_of_burn",
    "units": "doy",
    "grid_mapping": "albers_conical_equal_area",
}
nc["exp_fire_severity"].attrs = {
    "standard_name": "exp_fire_severity",
    "units": "",
    "grid_mapping": "albers_conical_equal_area",
}
nc["exp_area_of_burn"].attrs = {
    "standard_name": "exp_area_of_burn",
    "units": "km2",
    "grid_mapping": "albers_conical_equal_area",
}
nc["exp_pred_prob"].attrs = {
    "standard_name": "exp_pred_prob",
    "long_name": (
        "probability of burn; negative = FRI-overridden candidate "
        "(sign-flipped original probability)"
    ),
    "units": "",
    "comment": (
        ">0 kept burn; <0 cleared by time-varying recurrence "
        "(use abs for original p); 0 no candidate. "
        "TEM should use exp_burn_mask, not this field."
    ),
    "grid_mapping": "albers_conical_equal_area",
}
nc["exp_fri_limit"].attrs = {
    "standard_name": "exp_fri_limit",
    "long_name": "time-varying fire-return interval applied this year (years)",
    "units": "year",
    "comment": (
        "A candidate burn is kept only if years-since-last-fire >= this value. "
        "Chosen each year so kept count tracks the 200 (1850) → 1000 (2000) "
        "linear target."
    ),
}
nc.attrs.update(
    {
        "title": f"explicit fire {mod} v5 with time-varying FRI",
        "source_candidates": csv_path,
        "target_burns_1850": TARGET_BURNS_1850,
        "target_burns_2000": TARGET_BURNS_2000,
        "fri_burnin": FRI_BURNIN,
        "fri_mode": "adaptive" if ADAPTIVE_FRI else "linear",
        "burn_in_years": BurnInYears,
    }
)

encoding = {
    "exp_burn_mask": {"dtype": "int32", "_FillValue": np.int32(-999)},
    "exp_jday_of_burn": {"dtype": "int32", "_FillValue": np.int32(-999)},
    "exp_fire_severity": {"dtype": "int32", "_FillValue": np.int32(-999)},
    "exp_area_of_burn": {"dtype": "int32", "_FillValue": np.int32(-999)},
    "exp_pred_prob": {"dtype": "float64", "_FillValue": np.float64(-999.0)},
    "exp_fri_limit": {"dtype": "int32", "_FillValue": np.int32(-999)},
}
nc = drop_fillvalue_attrs(nc)
assert_jday_in_range(nc, "native")
print(f"Writing {out_native}")
nc.to_netcdf(out_native, unlimited_dims="time", encoding=encoding)
del nc, exp_burn_mask, exp_jday, exp_sev, exp_area, exp_prob, lat_t, lon_t

# -----------------------------------------------------------------------------
# 2) Pad onto climate Y/X
# -----------------------------------------------------------------------------
print(f"Padding onto climate grid from {climate_file}")
df = xr.open_dataset(out_native, decode_coords="all")
clmt = xr.open_dataset(climate_file, decode_coords="all")

missingX = sorted(list(set(clmt.X.values.tolist()) - set(df.X.values.tolist())))
missingY = sorted(list(set(clmt.Y.values.tolist()) - set(df.Y.values.tolist())))
print(f"  missing X={len(missingX)}, missing Y={len(missingY)}")

df2 = df
if missingX:
    df2 = pad_missing_axis(df2, "X", missingX)
if missingY:
    df2 = pad_missing_axis(df2, "Y", missingY)

df2 = cast_tem_fire_dtypes(df2)
df2 = drop_fillvalue_attrs(df2)
assert_jday_in_range(df2, "padded")
print(f"Writing {out_new}")
df2.to_netcdf(out_new, unlimited_dims="time", encoding=encoding)
df.close()
clmt.close()
del df2

print("Done (GCM v5 time-varying FRI).")
print(f"  native: {out_native}")
print(f"  padded: {out_new}")
print(
    f"  target {TARGET_BURNS_1850}→{TARGET_BURNS_2000}  "
    f"FRI burn-in={FRI_BURNIN} adaptive={ADAPTIVE_FRI} "
    f"BurnInYears={BurnInYears}"
)
