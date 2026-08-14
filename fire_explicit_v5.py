## Fire explicit v5 — GCM only: v4 draws + Joshua recurrence with stable burn-in
##
## Based on fire_explicit_v4.py (≤1 burn/pixel/year from Stefano monthly probs),
## then applies a fire recurrence limit using a randomized burn-in from STABLE
## monthly probabilities to initialize years-since-last-fire per cell.
##
## GCM only (UKESM / IPSL / GFDL). For stable fire use v3/v4 instead.
##
## Burn-in (recommended / enforced FRI):
##   delta starts at RecurrenceLimit (eligible)
##   each burn-in year: sample random stable month; if rand < p and
##   delta >= RecurrenceLimit → fire kept, delta = 0; then delta += 1
## Transient:
##   for each candidate annual burn: if delta < RecurrenceLimit → clear;
##   else keep and delta = 0; then delta += 1 every year
## Cleared burns: mask/jday/severity/area → 0; exp_pred_prob → −original
##   (negative marks FRI overrides for analysis plots; TEM should use mask)
##
## Edit CONFIG, then:
##   python fire_explicit_v5.py
##
## Outputs:
##   explicit-fire_{mod}_v5.nc
##   explicit-fire_{mod}_v5_new.nc

import os
import warnings

import numpy as np
import pandas as pd
import xarray as xr
from cftime import DatetimeNoLeap

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIG — change `mod` here (GCM only: not "stable")
# =============================================================================
indir = "/mnt/exacloud/cchang_woodwellclimate_org/wiemip/inputs/1pctCO2FireMLRawInput"
# mod = "GFDL-ESM4"
# mod = "IPSL-CM6A-LR"
mod = "UKESM1-0-LL"

outtemdir = "/mnt/exacloud/cchang_woodwellclimate_org/wiemip/weimip_inputs"
climatedir = "/mnt/exacloud/cchang_woodwellclimate_org/wiemip/inputs/teminputs"

RANDOM_SEED = 42

# Joshua recurrence (GCM only)
UseFireRecurrenceLimit = True
RecurrenceLimit = 20  # years; typical range 20–40
BurnInYears = RecurrenceLimit * 20  # e.g. 400; override to 200–400 if desired
# BurnInYears = 200
stable_prob_file = os.path.join(
    indir, "stable_monthly_prob_Native_combined_rescaled_v2.nc"
)
# =============================================================================

if mod == "stable":
    raise SystemExit(
        "fire_explicit_v5.py is for GCM cases only (UKESM/IPSL/GFDL).\n"
        "Use fire_explicit_v3.py or fire_explicit_v4.py for stable."
    )

if RANDOM_SEED is not None:
    np.random.seed(RANDOM_SEED)

os.makedirs(outtemdir, exist_ok=True)

infile = os.path.join(
    indir, f"{mod}_monthly_prob_Native_combined_rescaled_v2.nc"
)
climate_file = os.path.join(climatedir, f"climate_{mod}.nc")
out_native = os.path.join(outtemdir, f"explicit-fire_{mod}_v5.nc")
out_new = os.path.join(outtemdir, f"explicit-fire_{mod}_v5_new.nc")

if not os.path.isfile(infile):
    raise FileNotFoundError(
        f"Missing ML input: {infile}\n"
        f"Expected pattern: {{mod}}_monthly_prob_Native_combined_rescaled_v2.nc"
    )
if not os.path.isfile(climate_file):
    raise FileNotFoundError(
        f"Missing climate template: {climate_file}\n"
        f"Needed to pad fire onto climate Y/X (set climatedir)."
    )
if UseFireRecurrenceLimit and not os.path.isfile(stable_prob_file):
    raise FileNotFoundError(
        f"Missing stable prob file for burn-in: {stable_prob_file}"
    )


# Non-leap month lengths (TEM / DatetimeNoLeap: 365 days, no Feb 29)
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


def noleap_doy(month: int, day: int) -> int:
    """1-based day-of-year on a 365-day calendar (1..365). Never returns 366."""
    m = int(month)
    d = int(day)
    if m < 1 or m > 12:
        raise ValueError(f"month out of range: {m}")
    if d < 1 or d > MONTH_LENGTHS_NOLEAP[m - 1]:
        raise ValueError(f"day out of range for month {m}: {d}")
    return MONTH_START_DOY_NOLEAP[m] + d - 1


def noleap_doy_vec(months: np.ndarray, days: np.ndarray) -> np.ndarray:
    months = np.asarray(months, dtype=np.int16)
    days = np.asarray(days, dtype=np.int16)
    starts = np.asarray(MONTH_START_DOY_NOLEAP[1:], dtype=np.int16)  # index 0 -> Jan
    return starts[months - 1] + days - 1


def drop_fillvalue_attrs(ds: xr.Dataset) -> xr.Dataset:
    """Avoid xarray conflict when encoding also sets _FillValue."""
    ds = ds.copy()
    for name in ds.data_vars:
        if "_FillValue" in ds[name].attrs:
            ds[name].attrs = {
                k: v for k, v in ds[name].attrs.items() if k != "_FillValue"
            }
    return ds


def cast_tem_fire_dtypes(ds: xr.Dataset) -> xr.Dataset:
    """Force TEM-readable dtypes after ops that may promote ints to float64."""
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
    """Append empty cells along X or Y without promoting int vars to float64."""
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


def clear_burns_where(ds: xr.Dataset, clear: np.ndarray) -> xr.Dataset:
    """Clear burns where `clear` is True. `clear` shape matches time,Y,X.

    Burn mask/day/severity/area → 0. exp_pred_prob → −original (diagnostic
    flag for FRI/override analysis; abs(prob) is the pre-clear probability).
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


def apply_recurrence_with_stable_burnin(
    cand_mask: np.ndarray,
    cand_jday: np.ndarray,
    cand_prob: np.ndarray,
    active: np.ndarray,
    stable_pp: np.ndarray,
    recurrence_limit: int,
    burn_in_years: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Filter annual candidate burns with FRI + stable burn-in.

    cand_* : (n_years, n_lat, n_lon)
    stable_pp : (n_stable_months, n_lat, n_lon), NaN→0, clipped to [0,1]
    active : (n_lat, n_lon) bool

    Burn-in enforces FRI: fire kept only if delta >= recurrence_limit.
    Returns filtered mask/jday/prob and stats dict.
    """
    n_years, n_lat, n_lon = cand_mask.shape
    n_stable_t = stable_pp.shape[0]
    if stable_pp.shape[1:] != (n_lat, n_lon):
        raise SystemExit(
            f"Stable prob grid {stable_pp.shape[1:]} != GCM grid {(n_lat, n_lon)}"
        )

    flat_stable = np.ascontiguousarray(
        np.clip(np.nan_to_num(stable_pp, nan=0.0), 0.0, 1.0).reshape(n_stable_t, -1)
    )
    n_pix = flat_stable.shape[1]
    active_flat = active.reshape(-1)

    # Start eligible (recommended): delta = RecurrenceLimit
    delta = np.full(n_pix, recurrence_limit, dtype=np.int32)

    print(
        f"  burn-in: {burn_in_years} years from stable probs "
        f"(RecurrenceLimit={recurrence_limit}) ..."
    )
    n_burnin_kept = 0
    for t in range(burn_in_years):
        mi = rng.integers(0, n_stable_t, size=n_pix)
        p = flat_stable[mi, np.arange(n_pix)]
        u = rng.random(n_pix)
        fire_try = (u < p) & active_flat
        keep = fire_try & (delta >= recurrence_limit)
        n_burnin_kept += int(keep.sum())
        delta = np.where(keep, 0, delta)
        delta = delta + 1
        if (t + 1) % 100 == 0 or t + 1 == burn_in_years:
            print(f"    burn-in year {t + 1}/{burn_in_years}")

    delta_start = delta.copy()
    print(
        f"  burn-in done: kept fires={n_burnin_kept:,}; "
        f"deltaYearStart median={int(np.median(delta_start[active_flat]))} "
        f"mean={float(delta_start[active_flat].mean()):.1f}"
    )

    # Transient filter on candidates
    out_mask = cand_mask.copy()
    out_jday = cand_jday.copy()
    out_prob = cand_prob.copy()
    n_cand = int((cand_mask == 1).sum())
    n_cleared = 0
    n_kept = 0
    delta = delta_start

    print(f"  applying recurrence to {n_years} scenario years ...")
    for y in range(n_years):
        cand = (out_mask[y].reshape(-1) == 1) & active_flat
        keep = cand & (delta >= recurrence_limit)
        clear = cand & ~keep
        n_kept += int(keep.sum())
        n_cleared += int(clear.sum())

        if clear.any():
            clear2 = clear.reshape(n_lat, n_lon)
            out_mask[y][clear2] = 0
            out_jday[y][clear2] = 0
            # Sign-flip: negative exp_pred_prob marks FRI-overridden candidates
            out_prob[y][clear2] = -1.0 * out_prob[y][clear2]

        delta = np.where(keep, 0, delta)
        delta = delta + 1

    stats = {
        "burn_in_years": burn_in_years,
        "recurrence_limit": recurrence_limit,
        "burnin_kept_fires": n_burnin_kept,
        "candidate_burns": n_cand,
        "kept_burns": n_kept,
        "cleared_burns": n_cleared,
        "delta_start_median_active": int(np.median(delta_start[active_flat])),
        "delta_start_mean_active": float(delta_start[active_flat].mean()),
    }
    print(
        f"  recurrence filter: candidates={n_cand:,} kept={n_kept:,} "
        f"cleared={n_cleared:,} ({100.0 * n_cleared / max(n_cand, 1):.1f}%)"
    )
    return out_mask, out_jday, out_prob, stats


def _time_year_month(time_vals) -> tuple[np.ndarray, np.ndarray]:
    """Return int year, month arrays for an xarray time coordinate."""
    years = np.empty(len(time_vals), dtype=np.int32)
    months = np.empty(len(time_vals), dtype=np.int16)
    for i, t in enumerate(time_vals):
        years[i] = int(t.year)
        months[i] = int(t.month)
    return years, months


# -----------------------------------------------------------------------------
# 1) Read ML probs and draw burns (memory-safe NumPy path)
# -----------------------------------------------------------------------------
print(f"mod={mod}")
print(f"Reading {infile} (pred_prob only, float32) ...")
with xr.open_dataset(infile, decode_times=True) as ds_in:
    if "pred_prob" not in ds_in:
        raise SystemExit(f"pred_prob missing in {infile}")
    # Prefer dimension order (time, lat, lon)
    pp = ds_in["pred_prob"]
    if set(pp.dims) >= {"time", "lat", "lon"}:
        pp = pp.transpose("time", "lat", "lon")
    pred_prob = np.nan_to_num(pp.values, nan=0.0).astype(np.float32, copy=False)
    lat = np.asarray(ds_in["lat"].values, dtype=np.float64)
    lon = np.asarray(ds_in["lon"].values, dtype=np.float64)
    time_vals = ds_in["time"].values

n_time, n_lat, n_lon = pred_prob.shape
bytes_pp = pred_prob.nbytes / 1e9
print(
    f"  shape=(time={n_time}, lat={n_lat}, lon={n_lon}) "
    f"pred_prob={bytes_pp:.2f} GB"
)

years_t, months_t = _time_year_month(time_vals)
jan_mask = months_t == 1
if not np.any(jan_mask):
    raise RuntimeError("No January timesteps found — cannot build annual fire file")
year_list = years_t[jan_mask]
n_years = len(year_list)
year_to_idx = {int(y): i for i, y in enumerate(year_list)}
print(f"  annual years: {int(year_list[0])}..{int(year_list[-1])} (n={n_years})")

# Active pixels: any positive probability over the full record
active = np.sum(pred_prob, axis=0) > 0.0
n_active = int(np.count_nonzero(active))
print(f"  active pixels (sum pred_prob > 0): {n_active} / {n_lat * n_lon}")

# Vectorized Bernoulli draws: P(burn)=pred_prob  (same as choice([1,0], p=[p,1-p]))
print("Drawing monthly burns (vectorized) ...")
rng = np.random.default_rng(RANDOM_SEED)
# Draw in time chunks to limit peak RAM (random array + compare)
chunk = 120  # months
final_choice = np.zeros((n_time, n_lat, n_lon), dtype=np.bool_)
for t0 in range(0, n_time, chunk):
    t1 = min(t0 + chunk, n_time)
    block = pred_prob[t0:t1]
    draws = rng.random(block.shape, dtype=np.float32) < block
    # inactive pixels stay False
    draws &= active[np.newaxis, :, :]
    final_choice[t0:t1] = draws
    print(f"  months {t0}:{t1} done")

n_month_burns = int(np.count_nonzero(final_choice))
print(f"  monthly burn flags: {n_month_burns:,}")
if n_month_burns == 0:
    raise RuntimeError("No burns drawn; check pred_prob input")

# Sparse event table (only burned month-cells) — orders of magnitude smaller
print("Building sparse burn table ...")
ti, yi, xi = np.nonzero(final_choice)
probs_at_burn = pred_prob[ti, yi, xi].astype(np.float64, copy=False)
del final_choice, pred_prob

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
print(f"  sparse rows: {len(burned):,}")

# At most one burn per pixel per year (no multi-year FRI / 30y filter)
print("Sampling one burn month per pixel-year (no 30y filter) ...")
burn = (
    burned.groupby(["y_idx", "x_idx", "year"], sort=False)
    .sample(n=1, random_state=RANDOM_SEED)
    .reset_index(drop=True)
)
del burned
burn = burn[["y_idx", "x_idx", "year", "month", "pred_prob", "lat", "lon"]].copy()

# Random day-of-month on noleap calendar -> DOY in 1..365
mlen = MONTH_LENGTHS_ARR[burn["month"].to_numpy(dtype=np.int16) - 1]
# rng.integers low inclusive high exclusive
dom = rng.integers(1, mlen + 1, size=len(burn), dtype=np.int16)
burn["day_of_year"] = noleap_doy_vec(burn["month"].to_numpy(), dom)
jmin, jmax = int(burn["day_of_year"].min()), int(burn["day_of_year"].max())
if jmin < 1 or jmax > 365:
    raise RuntimeError(f"noleap DOY out of range: [{jmin}, {jmax}]")
print(f"  final annual burn events: {len(burn):,}; DOY range [{jmin}, {jmax}]")

# Optional compact event CSV (pre-recurrence candidates)
csv_path = os.path.join(indir, mod + "_burn_v5_events_candidates.csv")
burn.to_csv(csv_path, index=False)
print(f"Wrote sparse candidate CSV {csv_path}")

# -----------------------------------------------------------------------------
# Build annual native arrays, then apply Joshua recurrence (GCM)
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
# Map calendar year -> annual time index (January skeleton)
t_ann = np.array([year_to_idx[int(y)] for y in yy], dtype=np.int32)
exp_burn_mask[t_ann, yi, xi] = 1
exp_jday[t_ann, yi, xi] = burn["day_of_year"].to_numpy(dtype=np.int32)
exp_prob[t_ann, yi, xi] = burn["pred_prob"].to_numpy(dtype=np.float64)
del burn

n_before = int((exp_burn_mask == 1).sum())
print(f"  candidate annual burns (pre-recurrence): {n_before:,}")

if UseFireRecurrenceLimit:
    print(
        f"Applying fire recurrence limit (RecurrenceLimit={RecurrenceLimit}, "
        f"BurnInYears={BurnInYears}) using {stable_prob_file} ..."
    )
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
    if not (
        np.allclose(st_lat, lat, equal_nan=True)
        and np.allclose(st_lon, lon, equal_nan=True)
    ):
        print(
            "  WARNING: stable lat/lon values differ from GCM; "
            "proceeding on index alignment"
        )
    rng_fri = np.random.default_rng(
        None if RANDOM_SEED is None else RANDOM_SEED + 101
    )
    exp_burn_mask, exp_jday, exp_prob, fri_stats = apply_recurrence_with_stable_burnin(
        exp_burn_mask,
        exp_jday,
        exp_prob,
        active,
        stable_pp,
        RecurrenceLimit,
        BurnInYears,
        rng_fri,
    )
    del stable_pp
    print(f"  FRI stats: {fri_stats}")
else:
    print("UseFireRecurrenceLimit=False — keeping all annual candidates")

n_after = int((exp_burn_mask == 1).sum())
n_override = int((exp_prob < 0).sum())
print(f"  final annual burns: {n_after:,}")
print(
    f"  FRI overrides (exp_pred_prob < 0): {n_override:,} "
    f"(expect ~ cleared={fri_stats.get('cleared_burns', '?')})"
)

time_ann = [DatetimeNoLeap(int(y), 1, 1) for y in year_list]
# Y/X coords are lat/lon values (same convention as prior scripts)
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
        ">0 kept burn; <0 cleared by recurrence (use abs for original p); "
        "0 no candidate. TEM should use exp_burn_mask, not this field."
    ),
    "grid_mapping": "albers_conical_equal_area",
}

encoding = {
    "exp_burn_mask": {"dtype": "int32", "_FillValue": np.int32(-999)},
    "exp_jday_of_burn": {"dtype": "int32", "_FillValue": np.int32(-999)},
    "exp_fire_severity": {"dtype": "int32", "_FillValue": np.int32(-999)},
    "exp_area_of_burn": {"dtype": "int32", "_FillValue": np.int32(-999)},
    "exp_pred_prob": {"dtype": "float64", "_FillValue": np.float64(-999.0)},
}
nc = drop_fillvalue_attrs(nc)
assert_jday_in_range(nc, "native")
print(f"Writing {out_native}")
nc.to_netcdf(out_native, unlimited_dims="time", encoding=encoding)
del nc, exp_burn_mask, exp_jday, exp_sev, exp_area, exp_prob, lat_t, lon_t

# -----------------------------------------------------------------------------
# 2) Pad onto climate Y/X -> explicit-fire_{mod}_v5_new.nc
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
print(
    "  dtypes after cast:",
    {v: str(df2[v].dtype) for v in INT_FIRE_VARS + ("exp_pred_prob",) if v in df2},
)
print(f"Writing {out_new}")
df2.to_netcdf(out_new, unlimited_dims="time", encoding=encoding)
df.close()
clmt.close()
del df2

print("Done (GCM v5 — no stable 151y expansion).")
print(f"  native: {out_native}")
print(f"  padded: {out_new}")
print(
    f"  recurrence: UseFireRecurrenceLimit={UseFireRecurrenceLimit} "
    f"RecurrenceLimit={RecurrenceLimit} BurnInYears={BurnInYears}"
)
