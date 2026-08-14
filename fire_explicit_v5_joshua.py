## Fire explicit v5 — Joshua recurrence (spec-faithful)
##
## Same ML candidate burns as fire_explicit_v5.py, then Joshua's FRI:
##
##   deltaYearStart = RecurrenceLimit - 1
##   burn-in X years from stable:
##     randomly pull 1 of the ~21 stable years
##     apply the same monthly Bernoulli as GCM fires (u < p, any month)
##     if fire and deltaYearStart > RecurrenceLimit: deltaYearStart = 0
##     deltaYearStart += 1
##   for each GCM year with a predicted fire:
##     deltaYear = years since last KEPT fire
##     if none in the scenario: deltaYear = deltaYearStart + years elapsed
##     if deltaYear < RecurrenceLimit: clear_burns_where()
##
## Spec notes (implemented as intended, not typos):
##   - "if value > probability" would fire with P=1-p. GCM draws use u < p;
##     burn-in matches that ("matching the approach for determining fires above").
##   - Counting back to year 0 uses deltaYearStart PLUS elapsed years (otherwise
##     a cell with deltaYearStart < L would have every candidate deactivated).
##
## Does not overwrite v5 outputs:
##   explicit-fire_{mod}_v5_joshua.nc
##   explicit-fire_{mod}_v5_joshua_new.nc
##
##   python fire_explicit_v5_joshua.py

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
REUSE_CANDIDATE_CSV = True

UseFireRecurrenceLimit = True
RecurrenceLimit = 20  # years; Joshua typical range 20–40
BurnInYears = RecurrenceLimit * 20  # X in the spec; 400 for L=20
TARGET_BURNS_1850 = 200
TARGET_BURNS_2000 = 1000

stable_prob_file = os.path.join(
    indir, "stable_monthly_prob_Native_combined_rescaled_v2.nc"
)
# =============================================================================

if mod == "stable":
    raise SystemExit(
        "fire_explicit_v5_joshua.py is for GCM cases only (UKESM/IPSL/GFDL)."
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
out_native = os.path.join(outtemdir, f"explicit-fire_{mod}_v5_joshua.nc")
out_new = os.path.join(outtemdir, f"explicit-fire_{mod}_v5_joshua_new.nc")

if not os.path.isfile(infile):
    raise FileNotFoundError(f"Missing ML input: {infile}")
if not os.path.isfile(climate_file):
    raise FileNotFoundError(f"Missing climate template: {climate_file}")
if UseFireRecurrenceLimit and not os.path.isfile(stable_prob_file):
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
    return cast_tem_fire_dtypes(xr.concat([ds, empty], dim=dim).sortby(dim))


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
            f"{label}: exp_jday_of_burn out of range [{jmin}, {jmax}]"
        )
    print(f"  {label}: exp_jday_of_burn range [{jmin}, {jmax}]")


def clear_burns_where(ds: xr.Dataset, clear: np.ndarray) -> xr.Dataset:
    """Deactivate fires where `clear` is True (Joshua spec).

    mask/jday/severity/area → 0; exp_pred_prob → −original.
    """
    ds = ds.copy(deep=True)
    for var, fill in (
        ("exp_burn_mask", 0),
        ("exp_jday_of_burn", 0),
        ("exp_fire_severity", 0),
        ("exp_area_of_burn", 0),
    ):
        if var not in ds:
            continue
        vals = np.array(ds[var].values, copy=True)
        vals[clear] = fill
        ds[var].values = vals
    if "exp_pred_prob" in ds:
        vals = np.array(ds["exp_pred_prob"].values, copy=True, dtype=np.float64)
        vals[clear] = -1.0 * vals[clear]
        ds["exp_pred_prob"].values = vals
    return ds


def _time_year_month(time_vals) -> tuple[np.ndarray, np.ndarray]:
    years = np.empty(len(time_vals), dtype=np.int32)
    months = np.empty(len(time_vals), dtype=np.int16)
    for i, t in enumerate(time_vals):
        years[i] = int(t.year)
        months[i] = int(t.month)
    return years, months


def linear_target(years, n0, n1) -> np.ndarray:
    years = np.asarray(years, dtype=float)
    y0, y1 = float(years[0]), float(years[-1])
    return n0 + (n1 - n0) * (years - y0) / max(y1 - y0, 1.0)


def joshua_clear_mask(
    cand_mask: np.ndarray,
    active: np.ndarray,
    stable_pp: np.ndarray,
    recurrence_limit: int,
    burn_in_years: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict]:
    """Joshua burn-in + recurrence. Returns clear mask aligned with cand_mask.

    Burn-in keep test:  delta > RecurrenceLimit
    Transient clear:    delta < RecurrenceLimit   (keep if delta >= L)
    """
    n_years, n_lat, n_lon = cand_mask.shape
    n_stable_t = stable_pp.shape[0]
    if stable_pp.shape[1:] != (n_lat, n_lon):
        raise SystemExit(
            f"Stable prob grid {stable_pp.shape[1:]} != GCM grid {(n_lat, n_lon)}"
        )
    n_sy = n_stable_t // 12
    if n_sy < 1:
        raise SystemExit("Stable pred_prob needs at least 12 months")

    flat_stable = np.ascontiguousarray(
        np.clip(np.nan_to_num(stable_pp, nan=0.0), 0.0, 1.0).reshape(n_stable_t, -1)
    )
    n_pix = flat_stable.shape[1]
    active_flat = active.reshape(-1)
    active_idx = np.where(active_flat)[0]
    n_act = int(active_idx.size)

    # --- burn-in: deltaYearStart = RecurrenceLimit - 1 ---
    delta_act = np.full(n_act, recurrence_limit - 1, dtype=np.int32)
    print(
        f"  Joshua burn-in: {burn_in_years} years, random stable year "
        f"(n_stable_years={n_sy}), start delta={recurrence_limit - 1}, "
        f"keep if delta > {recurrence_limit}, n_active={n_act:,}"
    )
    n_burnin_kept = 0
    for t in range(burn_in_years):
        yi = int(rng.integers(0, n_sy))
        block = flat_stable[yi * 12 : (yi + 1) * 12][:, active_idx]
        # Same monthly Bernoulli as GCM candidate draws (u < p)
        fire_try = (rng.random(block.shape) < block).any(axis=0)
        keep = fire_try & (delta_act > recurrence_limit)
        n_burnin_kept += int(keep.sum())
        delta_act = np.where(keep, 0, delta_act) + 1
        if (t + 1) % 100 == 0 or t + 1 == burn_in_years:
            print(f"    burn-in year {t + 1}/{burn_in_years}")

    delta = np.full(n_pix, recurrence_limit - 1, dtype=np.int32)
    delta[active_idx] = delta_act
    delta_start = delta.copy()
    print(
        f"  burn-in done: kept={n_burnin_kept:,} "
        f"({n_burnin_kept / burn_in_years:.1f}/year); "
        f"deltaYearStart median={int(np.median(delta_act))} "
        f"mean={float(delta_act.mean()):.1f}"
    )

    # --- transient: count-back ≡ running delta (kept fires only) ---
    # if no kept fire since year 0: delta = deltaYearStart + years elapsed
    clear = np.zeros((n_years, n_lat, n_lon), dtype=bool)
    n_cand = int((cand_mask == 1).sum())
    n_cleared = 0
    n_kept = 0
    print(f"  Joshua recurrence on {n_years} scenario years ...")
    for y in range(n_years):
        cand = (cand_mask[y].reshape(-1) == 1) & active_flat
        # deactivate if deltaYear < RecurrenceLimit
        too_soon = cand & (delta < recurrence_limit)
        keep = cand & ~too_soon
        n_kept += int(keep.sum())
        n_cleared += int(too_soon.sum())
        if too_soon.any():
            clear[y] = too_soon.reshape(n_lat, n_lon)
        delta = np.where(keep, 0, delta) + 1

    stats = {
        "burn_in_years": burn_in_years,
        "recurrence_limit": recurrence_limit,
        "delta_init": recurrence_limit - 1,
        "burnin_keep_test": "delta > RecurrenceLimit",
        "transient_clear_test": "delta < RecurrenceLimit",
        "burnin_kept_fires": n_burnin_kept,
        "burnin_mean_annual": float(n_burnin_kept / max(burn_in_years, 1)),
        "candidate_burns": n_cand,
        "kept_burns": n_kept,
        "cleared_burns": n_cleared,
        "delta_start_median_active": int(np.median(delta_start[active_flat])),
        "delta_start_mean_active": float(delta_start[active_flat].mean()),
    }
    print(
        f"  Joshua filter: candidates={n_cand:,} kept={n_kept:,} "
        f"cleared={n_cleared:,} ({100.0 * n_cleared / max(n_cand, 1):.1f}%)"
    )
    return clear, stats


def draw_candidates(lat, lon, time_vals, pred_prob, rng) -> pd.DataFrame:
    n_time, n_lat, n_lon = pred_prob.shape
    years_t, months_t = _time_year_month(time_vals)
    active = np.sum(pred_prob, axis=0) > 0.0
    print(f"  active pixels: {int(active.sum())} / {n_lat * n_lon}")
    chunk = 120
    final_choice = np.zeros((n_time, n_lat, n_lon), dtype=np.bool_)
    for t0 in range(0, n_time, chunk):
        t1 = min(t0 + chunk, n_time)
        block = pred_prob[t0:t1]
        draws = (rng.random(block.shape, dtype=np.float32) < block) & active[
            np.newaxis, :, :
        ]
        final_choice[t0:t1] = draws
        print(f"  months {t0}:{t1} done")
    ti, yi, xi = np.nonzero(final_choice)
    burned = pd.DataFrame(
        {
            "y_idx": yi.astype(np.int32),
            "x_idx": xi.astype(np.int32),
            "year": years_t[ti],
            "month": months_t[ti],
            "pred_prob": pred_prob[ti, yi, xi].astype(np.float64),
            "lat": lat[yi],
            "lon": lon[xi],
        }
    )
    del final_choice
    burn = (
        burned.groupby(["y_idx", "x_idx", "year"], sort=False)
        .sample(n=1, random_state=RANDOM_SEED)
        .reset_index(drop=True)
    )
    del burned
    mlen = MONTH_LENGTHS_ARR[burn["month"].to_numpy(dtype=np.int16) - 1]
    dom = rng.integers(1, mlen + 1, size=len(burn), dtype=np.int16)
    burn["day_of_year"] = noleap_doy_vec(burn["month"].to_numpy(), dom)
    return burn[
        ["y_idx", "x_idx", "year", "month", "pred_prob", "lat", "lon", "day_of_year"]
    ]


def plot_vs_target(years, annual, target, out_png):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sm = (
        pd.Series(annual.astype(float))
        .rolling(11, center=True, min_periods=6)
        .mean()
        .to_numpy()
    )
    rmse = float(np.sqrt(np.mean((annual.astype(float) - target) ** 2)))
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(years, annual, color="#c23b22", lw=1.2, alpha=0.85, label="Joshua kept / year")
    ax.plot(years, sm, color="#1a1a1a", lw=1.8, label="11-yr mean")
    ax.plot(years, target, color="#2c7bb6", lw=1.8, ls="--", label="linear target 200→1000")
    ax.set_xlabel("Year")
    ax.set_ylabel("Burned pixels / year")
    ax.set_title(
        f"{mod} v5 Joshua FRI   1850={int(annual[0])}  2000={int(annual[-1])}  "
        f"mean={annual.mean():.0f}  RMSE vs target={rmse:.1f}"
    )
    ax.legend(frameon=False, ncol=3, fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(int(years[0]), int(years[-1]))
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_png}")


# -----------------------------------------------------------------------------
# Candidates (same as v5; reuse CSV when present)
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
        pp = ds_in["pred_prob"]
        if set(pp.dims) >= {"time", "lat", "lon"}:
            pp = pp.transpose("time", "lat", "lon")
        pred_prob = np.nan_to_num(pp.values, nan=0.0).astype(np.float32, copy=False)

n_lat, n_lon = lat.size, lon.size
years_t, months_t = _time_year_month(time_vals)
jan_mask = months_t == 1
if not np.any(jan_mask):
    raise RuntimeError("No January timesteps found")
year_list = years_t[jan_mask]
n_years = len(year_list)
year_to_idx = {int(y): i for i, y in enumerate(year_list)}
print(f"  annual years: {int(year_list[0])}..{int(year_list[-1])} (n={n_years})")

rng = np.random.default_rng(RANDOM_SEED)
if need_draws:
    print("Drawing GCM candidate burns ...")
    burn = draw_candidates(lat, lon, time_vals, pred_prob, rng)
    del pred_prob
    burn.to_csv(csv_path, index=False)
    print(f"Wrote {csv_path}")
else:
    print(f"Reusing candidate CSV {csv_path}")
    burn = pd.read_csv(csv_path)

print(f"  candidate annual burn events: {len(burn):,}")

exp_burn_mask = np.zeros((n_years, n_lat, n_lon), dtype=np.int32)
exp_jday = np.zeros((n_years, n_lat, n_lon), dtype=np.int32)
exp_sev = np.zeros((n_years, n_lat, n_lon), dtype=np.int32)
exp_area = np.zeros((n_years, n_lat, n_lon), dtype=np.int32)
exp_prob = np.zeros((n_years, n_lat, n_lon), dtype=np.float64)
yi = burn["y_idx"].to_numpy(dtype=np.int32)
xi = burn["x_idx"].to_numpy(dtype=np.int32)
t_ann = np.array(
    [year_to_idx[int(y)] for y in burn["year"].to_numpy(dtype=np.int32)],
    dtype=np.int32,
)
exp_burn_mask[t_ann, yi, xi] = 1
exp_jday[t_ann, yi, xi] = burn["day_of_year"].to_numpy(dtype=np.int32)
exp_prob[t_ann, yi, xi] = burn["pred_prob"].to_numpy(dtype=np.float64)
active = exp_burn_mask.any(axis=0)
del burn

cand_mask = exp_burn_mask.copy()
print(f"  pre-FRI candidates: {int((cand_mask == 1).sum()):,}")

clear = np.zeros((n_years, n_lat, n_lon), dtype=bool)
fri_stats = {}
if UseFireRecurrenceLimit:
    print(f"Loading stable probs {stable_prob_file} ...")
    with xr.open_dataset(stable_prob_file, decode_times=False) as ds_st:
        spp = ds_st["pred_prob"]
        if set(spp.dims) >= {"time", "lat", "lon"}:
            spp = spp.transpose("time", "lat", "lon")
        stable_pp = np.asarray(spp.values, dtype=np.float32)
    rng_fri = np.random.default_rng(
        None if RANDOM_SEED is None else RANDOM_SEED + 101
    )
    clear, fri_stats = joshua_clear_mask(
        cand_mask,
        active,
        stable_pp,
        RecurrenceLimit,
        BurnInYears,
        rng_fri,
    )
    del stable_pp
    print(f"  FRI stats: {fri_stats}")
else:
    print("UseFireRecurrenceLimit=False — keeping all candidates")

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
        ">0 kept burn; <0 cleared by Joshua recurrence "
        "(use abs for original p); 0 no candidate. "
        "TEM should use exp_burn_mask, not this field."
    ),
    "grid_mapping": "albers_conical_equal_area",
}
nc.attrs.update(
    {
        "title": f"explicit fire {mod} v5 Joshua recurrence",
        "recurrence_limit": RecurrenceLimit,
        "burn_in_years": BurnInYears,
        "joshua_delta_init": RecurrenceLimit - 1,
        "joshua_burnin_keep": "delta > RecurrenceLimit",
        "joshua_transient_clear": "delta < RecurrenceLimit",
    }
)

if UseFireRecurrenceLimit and clear.any():
    print("Deactivating too-soon fires via clear_burns_where() ...")
    nc = clear_burns_where(nc, clear)

encoding = {
    "exp_burn_mask": {"dtype": "int32", "_FillValue": np.int32(-999)},
    "exp_jday_of_burn": {"dtype": "int32", "_FillValue": np.int32(-999)},
    "exp_fire_severity": {"dtype": "int32", "_FillValue": np.int32(-999)},
    "exp_area_of_burn": {"dtype": "int32", "_FillValue": np.int32(-999)},
    "exp_pred_prob": {"dtype": "float64", "_FillValue": np.float64(-999.0)},
}
nc = drop_fillvalue_attrs(nc)
assert_jday_in_range(nc, "native")

annual = np.asarray(nc["exp_burn_mask"].values == 1).sum(axis=(1, 2)).astype(np.int32)
target = linear_target(year_list, TARGET_BURNS_1850, TARGET_BURNS_2000)
rmse = float(np.sqrt(np.mean((annual.astype(float) - target) ** 2)))
slope = float(np.polyfit(year_list.astype(float), annual.astype(float), 1)[0])
print(
    f"  final burns: {int(annual.sum()):,}  "
    f"1850={int(annual[0])}  2000={int(annual[-1])}  "
    f"mean={float(annual.mean()):.1f}  slope={slope:.2f}/yr  "
    f"RMSE vs 200→1000={rmse:.1f}"
)
print(
    f"  linear target 200→1000: "
    f"{'NO — constant Joshua FRI does not impose that ramp' if rmse > 80 else 'close'}"
)
plot_vs_target(
    year_list,
    annual,
    target,
    os.path.join(figdir, f"explicit-fire_{mod}_v5_joshua.png"),
)

print(f"Writing {out_native}")
nc.to_netcdf(out_native, unlimited_dims="time", encoding=encoding)
del nc, exp_burn_mask, exp_jday, exp_sev, exp_area, exp_prob, lat_t, lon_t

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

print("Done (Joshua v5).")
print(f"  native: {out_native}")
print(f"  padded: {out_new}")
print(
    f"  RecurrenceLimit={RecurrenceLimit} BurnInYears={BurnInYears} "
    f"delta_init={RecurrenceLimit - 1}"
)
