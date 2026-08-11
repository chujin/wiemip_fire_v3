## Fire explicit v3 — memory-safe (UKESM / IPSL / GFDL on ~16 GB nodes)
##
## Same algorithm as colleague v2 + stable 151y thin-on-tile, but:
##   - never calls Dataset.to_dataframe() on the full monthly grid
##   - never writes a multi-GB monthly burn CSV
##   - draws burns with NumPy on float32 arrays (~0.5–1 GB peak for 151y)
##
## Also:
##   - burn DOY uses a fixed 365-day (noleap) calendar (never 366)
##   - TEM burn fields stay int32 through climate-grid padding / full expand
##
## Edit only the CONFIG block (especially `mod`), then:
##   python fire_explicit_v3.py
##
## Outputs under outtemdir:
##   explicit-fire_{mod}_v3.nc       native ML grid
##   explicit-fire_{mod}_v3_new.nc   padded to climate Y/X grid
##   explicit-fire_stable_full_v3.nc only if mod == "stable" (20y cycle -> 151y, thinned)
##
## Note: vectorized RNG uses the same RANDOM_SEED but a different draw order
## than the old per-pixel pandas loop, so bit-identical burns are not expected.

import os
import warnings

import numpy as np
import pandas as pd
import xarray as xr
from cftime import DatetimeNoLeap

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIG — change `mod` here (only one active)
# =============================================================================
indir = "/mnt/exacloud/cchang_woodwellclimate_org/wiemip/inputs/1pctCO2FireMLRawInput"
mod = "GFDL-ESM4"
#mod = "IPSL-CM6A-LR"
# mod = "stable"
# mod = "UKESM1-0-LL"

outtemdir = "/mnt/exacloud/cchang_woodwellclimate_org/wiemip/weimip_inputs"
# Climate grids used to pad fire onto the TEM spatial domain
climatedir = "/mnt/exacloud/cchang_woodwellclimate_org/wiemip/inputs/teminputs"

RANDOM_SEED = 42
# =============================================================================

if RANDOM_SEED is not None:
    np.random.seed(RANDOM_SEED)

os.makedirs(outtemdir, exist_ok=True)

infile = os.path.join(
    indir, f"{mod}_monthly_prob_Native_combined_rescaled_v2.nc"
)
climate_file = os.path.join(climatedir, f"climate_{mod}.nc")
out_native = os.path.join(outtemdir, f"explicit-fire_{mod}_v3.nc")
out_new = os.path.join(outtemdir, f"explicit-fire_{mod}_v3_new.nc")
out_full = os.path.join(outtemdir, "explicit-fire_stable_full_v3.nc")

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
    """Zero burn fields where `clear` is True. `clear` shape matches time,Y,X."""
    ds = ds.copy(deep=True)
    for var, fill in (
        ("exp_burn_mask", 0),
        ("exp_jday_of_burn", 0),
        ("exp_fire_severity", 0),
        ("exp_area_of_burn", 0),
        ("exp_pred_prob", 0.0),
    ):
        if var not in ds:
            continue
        vals = np.array(ds[var].values, copy=True)
        vals[clear] = fill
        ds[var].values = vals
    return ds


def _time_year_month(time_vals) -> tuple[np.ndarray, np.ndarray]:
    """Return int year, month arrays for an xarray time coordinate."""
    # cftime or datetime64
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

# One random burn month per lat/lon/year (same as groupby.sample n=1)
print("Sampling one burn month per pixel-year ...")
ds1burn = burned.groupby(["y_idx", "x_idx", "year"], sort=False).sample(
    n=1, random_state=RANDOM_SEED
)
del burned

ds1burn_sorted = ds1burn.sort_values(
    by=["y_idx", "x_idx", "year"], ascending=[True, True, True]
)
ds1burn_sorted["year_lag1"] = ds1burn_sorted.groupby(["y_idx", "x_idx"])["year"].shift(
    1
)
ds1burn_sorted["t_lag1"] = ds1burn_sorted.groupby(["y_idx", "x_idx"])["t_idx"].shift(1)
ds1burn_sorted["month_lag1"] = ds1burn_sorted.groupby(["y_idx", "x_idx"])["month"].shift(
    1
)
ds1burn_sorted["prob_lag1"] = ds1burn_sorted.groupby(["y_idx", "x_idx"])[
    "pred_prob"
].shift(1)
ds1burn_sorted["delta_year"] = ds1burn_sorted["year"] - ds1burn_sorted["year_lag1"]

# Keep first candidate or gap >= 30y vs previous *candidate* (colleague rule)
ds1burn_select = ds1burn_sorted[
    (ds1burn_sorted["delta_year"] >= 30) | (ds1burn_sorted["year_lag1"].isna())
].copy()

# b1 = selected; b2 = previous candidate of selected (lag1), then unique by year/pixel
b1 = ds1burn_select[
    ["y_idx", "x_idx", "year", "month", "pred_prob", "lat", "lon"]
].copy()
b2 = ds1burn_select[
    ["y_idx", "x_idx", "year_lag1", "month_lag1", "prob_lag1", "lat", "lon"]
].rename(
    columns={
        "year_lag1": "year",
        "month_lag1": "month",
        "prob_lag1": "pred_prob",
    }
)
b2 = b2.dropna(subset=["year"])
b2["year"] = b2["year"].astype(np.int32)
b2["month"] = b2["month"].astype(np.int16)

burn = pd.concat([b1, b2], ignore_index=True)
burn = burn.drop_duplicates(subset=["y_idx", "x_idx", "year", "month"], keep="first")
# If duplicate years from b1+b2 with different months, keep first
burn = burn.drop_duplicates(subset=["y_idx", "x_idx", "year"], keep="first")
del ds1burn, ds1burn_sorted, ds1burn_select, b1, b2

# Random day-of-month on noleap calendar -> DOY in 1..365
mlen = MONTH_LENGTHS_ARR[burn["month"].to_numpy(dtype=np.int16) - 1]
# rng.integers low inclusive high exclusive
dom = rng.integers(1, mlen + 1, size=len(burn), dtype=np.int16)
burn["day_of_year"] = noleap_doy_vec(burn["month"].to_numpy(), dom)
jmin, jmax = int(burn["day_of_year"].min()), int(burn["day_of_year"].max())
if jmin < 1 or jmax > 365:
    raise RuntimeError(f"noleap DOY out of range: [{jmin}, {jmax}]")
print(f"  final annual burn events: {len(burn):,}; DOY range [{jmin}, {jmax}]")

# Optional compact event CSV (small — not the old multi-GB monthly dump)
csv_path = os.path.join(indir, mod + "_burn_v3_events.csv")
burn.to_csv(csv_path, index=False)
print(f"Wrote sparse event CSV {csv_path}")

# -----------------------------------------------------------------------------
# Build annual native NetCDF arrays directly (no giant dataframe)
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

time_ann = [DatetimeNoLeap(int(y), 1, 1) for y in year_list]
# Y/X coords are lat/lon values (same convention as prior v2/v3 scripts)
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
    "long_name": "probability of burn",
    "units": "",
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
# 2) Pad onto climate Y/X -> explicit-fire_{mod}_v3_new.nc
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

# -----------------------------------------------------------------------------
# 3) Stable only: expand first 20 years to 151y, thinning burns across tiles
# -----------------------------------------------------------------------------
if mod == "stable":
    print(
        "Expanding stable fire to 151 years (20-year cycle), "
        "partitioning burns across tile repeats (≈1/8 keep) ..."
    )
    df = xr.open_dataset(out_new, decode_coords="all")
    t_all = [DatetimeNoLeap(1850 + i, 1, 1) for i in range(151)]

    n_native = int(df.sizes["time"])
    n_cycle = min(20, n_native)
    if n_native < 20:
        print(
            f"  WARNING: native stable has only {n_native} years; "
            f"cycling with period={n_cycle}"
        )

    starts = [0, 20, 40, 60, 80, 100, 120, 140]
    lengths = [20, 20, 20, 20, 20, 20, 20, 11]
    n_blocks = len(starts)

    blocks_for_year = [[] for _ in range(n_cycle)]
    for b, length in enumerate(lengths):
        for local_t in range(min(length, n_cycle)):
            blocks_for_year[local_t].append(b)

    rng_tile = np.random.default_rng(RANDOM_SEED)
    template_burn = np.array(df["exp_burn_mask"].isel(time=slice(0, n_cycle)).values)
    ny, nY, nX = template_burn.shape
    assignment = np.full((ny, nY, nX), -1, dtype=np.int16)
    n_template_burns = 0
    for y in range(ny):
        candidates = blocks_for_year[y]
        if not candidates:
            continue
        iy, ix = np.where(template_burn[y] == 1)
        n_template_burns += len(iy)
        if len(iy) == 0:
            continue
        chosen = rng_tile.choice(np.asarray(candidates, dtype=np.int16), size=len(iy))
        assignment[y, iy, ix] = chosen

    print(
        f"  template burns={n_template_burns}; "
        f"partitioning across {n_blocks} tile blocks "
        f"(~1/{n_blocks} keep per year-of-cycle)"
    )

    blocks = []
    n_kept = 0
    for b, (start, length) in enumerate(zip(starts, lengths)):
        src = df.isel(time=slice(0, min(length, n_cycle)))
        if src.sizes["time"] < length:
            idx = np.arange(length) % src.sizes["time"]
            src = src.isel(time=idx)
        else:
            src = src.isel(time=slice(0, length))

        burn_arr = np.array(src["exp_burn_mask"].values)
        clear = np.zeros(burn_arr.shape, dtype=bool)
        for local_t in range(src.sizes["time"]):
            y = local_t % n_cycle
            clear[local_t] = (burn_arr[local_t] == 1) & (assignment[y] != b)
            n_kept += int(np.sum((burn_arr[local_t] == 1) & (assignment[y] == b)))

        src = clear_burns_where(src, clear)
        src = src.assign_coords(time=t_all[start : start + length])
        blocks.append(src)

    print(f"  kept burns over 151y={n_kept} (expect ≈ template burns)")

    dftot = xr.concat(blocks, dim="time").sortby("time")
    dftot = cast_tem_fire_dtypes(dftot)
    dftot = drop_fillvalue_attrs(dftot)
    assert_jday_in_range(dftot, "stable_full")
    print(f"Writing {out_full} (time={dftot.sizes['time']})")
    dftot.to_netcdf(out_full, unlimited_dims="time", encoding=encoding)
    df.close()
else:
    print(
        f"mod={mod}: skipped stable 151y expansion "
        f"(only runs when mod == 'stable')."
    )

print("Done.")
print(f"  native: {out_native}")
print(f"  padded: {out_new}")
if mod == "stable":
    print(f"  full:   {out_full}")
