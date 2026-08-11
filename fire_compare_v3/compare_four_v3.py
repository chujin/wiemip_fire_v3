#!/usr/bin/env python3
"""
Compare four v3 explicit-fire products (1850–2000, climate Y/X grid):

  - explicit-fire_GFDL-ESM4_v3_new.nc
  - explicit-fire_IPSL-CM6A-LR_v3_new.nc
  - explicit-fire_UKESM1-0-LL_v3_new.nc
  - explicit-fire_stable_full_v3.nc

Writes PNG figures + summary.json into ./figures and this directory.

Usage:
  cd .../weimip_inputs/fire_compare_v3
  python compare_four_v3.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from netCDF4 import Dataset, num2date

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
OUTDIR = Path(__file__).resolve().parent
FIGDIR = OUTDIR / "figures"
BASE = Path("/mnt/exacloud/cchang_woodwellclimate_org/wiemip/weimip_inputs")

DATASETS = {
    "stable": BASE / "explicit-fire_stable_full_v3.nc",
    "GFDL-ESM4": BASE / "explicit-fire_GFDL-ESM4_v3_new.nc",
    "IPSL-CM6A-LR": BASE / "explicit-fire_IPSL-CM6A-LR_v3_new.nc",
    "UKESM1-0-LL": BASE / "explicit-fire_UKESM1-0-LL_v3_new.nc",
}

COLORS = {
    "stable": "#4D4D4D",
    "GFDL-ESM4": "#1B9E77",
    "IPSL-CM6A-LR": "#D95F02",
    "UKESM1-0-LL": "#7570B3",
}

# noleap month starts (1-based DOY of day 1)
MONTH_START = np.array([1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335, 366])
MONTH_NAMES = [
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
]


def _years_from_time(ds) -> np.ndarray:
    tvar = ds.variables["time"]
    t = np.asarray(tvar[:])
    units = getattr(tvar, "units", None)
    calendar = getattr(tvar, "calendar", "noleap")
    if units:
        try:
            dates = num2date(t, units=units, calendar=calendar)
            return np.array([int(d.year) for d in dates], dtype=np.int32)
        except Exception:
            pass
    return (t.astype(float) / 365.0).astype(np.int32) + 1850


def doy_to_month(doy: np.ndarray) -> np.ndarray:
    """1-based DOY -> month 1..12 on 365-day calendar."""
    doy = np.asarray(doy, dtype=float)
    m = np.searchsorted(MONTH_START, doy, side="right") 
    # searchsorted on starts: DOY 1 -> index 1 after right? 
    # MONTH_START[0]=1; for doy=1, right insert at 1 -> month 1. Good if we use:
    return np.clip(m, 1, 12).astype(np.int16)


def load_fire(path: Path, label: str) -> dict:
    print(f"Loading {label}: {path.name}")
    with Dataset(path) as ds:
        Y = np.asarray(ds.variables["Y"][:], dtype=np.float64)
        X = np.asarray(ds.variables["X"][:], dtype=np.float64)
        years = _years_from_time(ds)
        mask = np.asarray(ds.variables["exp_burn_mask"][:], dtype=np.float64)
        jday = np.asarray(ds.variables["exp_jday_of_burn"][:], dtype=np.float64)
        if "exp_pred_prob" in ds.variables:
            prob = np.asarray(ds.variables["exp_pred_prob"][:], dtype=np.float64)
        else:
            prob = None
        # lat if present for geographic maps
        lat = None
        lon = None
        if "lat" in ds.variables:
            lat_v = np.asarray(ds.variables["lat"][:], dtype=np.float64)
            if lat_v.ndim == 3:
                lat = lat_v[0]
            elif lat_v.ndim == 2:
                lat = lat_v
        if "lon" in ds.variables:
            lon_v = np.asarray(ds.variables["lon"][:], dtype=np.float64)
            if lon_v.ndim == 3:
                lon = lon_v[0]
            elif lon_v.ndim == 2:
                lon = lon_v

    valid = ~(np.isnan(mask) | (mask < -900))
    burn = (mask == 1) & valid
    # jday only meaningful on burns; clean fills
    jday_c = jday.copy()
    jday_c[~burn] = np.nan

    out = {
        "label": label,
        "path": str(path),
        "Y": Y,
        "X": X,
        "years": years,
        "burn": burn,
        "jday": jday_c,
        "prob": prob,
        "lat": lat,
        "lon": lon,
        "n_time": burn.shape[0],
        "total_burns": int(burn.sum()),
        "annual": burn.sum(axis=(1, 2)).astype(np.int32),
        "freq": burn.sum(axis=0).astype(np.float64),
        "ever": burn.any(axis=0),
    }
    jb = jday_c[burn]
    out["jday_burn"] = jb
    out["jday_mean"] = float(np.nanmean(jb)) if jb.size else float("nan")
    out["jday_med"] = float(np.nanmedian(jb)) if jb.size else float("nan")
    out["pixels_ever"] = int(out["ever"].sum())
    out["mean_annual"] = float(out["annual"].mean())
    print(
        f"  years {years[0]}-{years[-1]}  burns={out['total_burns']:,}  "
        f"pixels_ever={out['pixels_ever']:,}  mean_jday={out['jday_mean']:.1f}"
    )
    return out


def reburn_gaps(burn: np.ndarray, years: np.ndarray) -> np.ndarray:
    """Years between successive burns at each pixel (vectorized per column)."""
    gaps = []
    nt, ny, nx = burn.shape
    # Flatten spatial; iterate pixels that burn at least twice
    flat = burn.reshape(nt, -1)
    ever2 = flat.sum(axis=0) >= 2
    idxs = np.where(ever2)[0]
    for k in idxs:
        ts = np.where(flat[:, k])[0]
        gaps.extend(np.diff(years[ts]).tolist())
    return np.asarray(gaps, dtype=np.int32) if gaps else np.array([], dtype=np.int32)


def savefig(fig: plt.Figure, name: str) -> Path:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    path = FIGDIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path.name}")
    return path


def plot_annual(data: dict[str, dict]) -> None:
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for name, d in data.items():
        ax.plot(
            d["years"],
            d["annual"],
            label=name,
            color=COLORS[name],
            lw=1.4,
            alpha=0.9,
        )
    ax.set_xlabel("Year")
    ax.set_ylabel("Burned pixels per year")
    ax.set_title("Annual burn count (1850–2000)")
    ax.legend(frameon=False, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(int(next(iter(data.values()))["years"][0]), int(next(iter(data.values()))["years"][-1]))
    savefig(fig, "01_annual_burn_counts.png")

    # Smoothed (11y rolling)
    fig, ax = plt.subplots(figsize=(11, 4.5))
    w = 11
    for name, d in data.items():
        y = d["annual"].astype(float)
        ker = np.ones(w) / w
        sm = np.convolve(y, ker, mode="same")
        ax.plot(d["years"], sm, label=name, color=COLORS[name], lw=1.8)
    ax.set_xlabel("Year")
    ax.set_ylabel(f"Burned pixels ({w}-yr moving mean)")
    ax.set_title(f"Annual burn count — {w}-year moving mean")
    ax.legend(frameon=False, ncol=2)
    ax.grid(True, alpha=0.3)
    savefig(fig, "01b_annual_burn_counts_smoothed.png")


def plot_spatial_freq(data: dict[str, dict]) -> None:
    # Shared color scale
    vmax = max(float(d["freq"].max()) for d in data.values())
    vmax = max(vmax, 1.0)
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    for ax, (name, d) in zip(axes.ravel(), data.items()):
        im = ax.imshow(
            d["freq"],
            origin="upper",
            aspect="auto",
            cmap="YlOrRd",
            vmin=0,
            vmax=vmax,
            interpolation="nearest",
        )
        ax.set_title(f"{name}\nΣ burns / pixel (max={int(d['freq'].max())})")
        ax.set_xlabel("X index")
        ax.set_ylabel("Y index")
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.7, label="Burn count over 151 years")
    fig.suptitle("Spatial burn frequency (sum of exp_burn_mask==1)", fontsize=12)
    savefig(fig, "02_spatial_burn_frequency.png")


def plot_freq_diff_vs_stable(data: dict[str, dict]) -> None:
    stab = data["stable"]["freq"]
    gcms = [k for k in data if k != "stable"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), constrained_layout=True)
    lim = 0
    diffs = {}
    for name in gcms:
        diffs[name] = data[name]["freq"] - stab
        lim = max(lim, float(np.nanmax(np.abs(diffs[name]))))
    lim = max(lim, 1.0)
    for ax, name in zip(axes, gcms):
        im = ax.imshow(
            diffs[name],
            origin="upper",
            aspect="auto",
            cmap="RdBu_r",
            vmin=-lim,
            vmax=lim,
            interpolation="nearest",
        )
        ax.set_title(f"{name} − stable")
        ax.set_xlabel("X index")
        ax.set_ylabel("Y index")
    fig.colorbar(im, ax=axes.tolist(), shrink=0.8, label="Δ burn count (151y)")
    fig.suptitle("Spatial frequency difference vs stable_full_v3", fontsize=12)
    savefig(fig, "03_spatial_frequency_vs_stable.png")


def plot_jday(data: dict[str, dict]) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.5))
    bins = np.arange(1, 367, 5)
    for name, d in data.items():
        jb = d["jday_burn"]
        if jb.size == 0:
            continue
        ax.hist(
            jb,
            bins=bins,
            histtype="step",
            density=True,
            lw=1.6,
            label=f"{name} (n={jb.size:,})",
            color=COLORS[name],
        )
    ax.set_xlabel("Day of year (noleap, 1–365)")
    ax.set_ylabel("Density")
    ax.set_title("Burn day-of-year distribution")
    ax.legend(frameon=False, fontsize=8)
    ax.set_xlim(1, 365)
    ax.grid(True, alpha=0.3)
    savefig(fig, "04_jday_histogram.png")

    # Monthly seasonality
    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = np.arange(1, 13)
    width = 0.2
    for i, (name, d) in enumerate(data.items()):
        jb = d["jday_burn"]
        if jb.size == 0:
            continue
        months = doy_to_month(jb)
        counts = np.array([(months == m).sum() for m in range(1, 13)], dtype=float)
        counts = counts / counts.sum()
        ax.bar(
            x + (i - 1.5) * width,
            counts,
            width=width,
            label=name,
            color=COLORS[name],
            edgecolor="none",
        )
    ax.set_xticks(x)
    ax.set_xticklabels(MONTH_NAMES)
    ax.set_ylabel("Fraction of burns")
    ax.set_xlabel("Month (from noleap DOY)")
    ax.set_title("Seasonality of burn day-of-year")
    ax.legend(frameon=False, ncol=2, fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    savefig(fig, "05_jday_monthly_seasonality.png")


def plot_reburn_gaps(data: dict[str, dict]) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.5))
    bins = np.arange(0, 161, 5)
    summary_gaps = {}
    for name, d in data.items():
        print(f"  reburn gaps: {name} ...")
        gaps = reburn_gaps(d["burn"], d["years"])
        summary_gaps[name] = {
            "n_gaps": int(gaps.size),
            "median": float(np.median(gaps)) if gaps.size else None,
            "mean": float(np.mean(gaps)) if gaps.size else None,
            "pct_lt_30": float(np.mean(gaps < 30) * 100) if gaps.size else None,
        }
        if gaps.size == 0:
            continue
        ax.hist(
            gaps,
            bins=bins,
            histtype="step",
            density=True,
            lw=1.6,
            label=f"{name} med={np.median(gaps):.0f}y",
            color=COLORS[name],
        )
    ax.axvline(30, color="k", ls="--", lw=1, label="30y rule")
    ax.set_xlabel("Years between successive burns at same pixel")
    ax.set_ylabel("Density")
    ax.set_title("Reburn interval distribution")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, alpha=0.3)
    savefig(fig, "06_reburn_gap_histogram.png")
    return summary_gaps


def plot_cumulative(data: dict[str, dict]) -> None:
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for name, d in data.items():
        ax.plot(
            d["years"],
            np.cumsum(d["annual"]),
            label=name,
            color=COLORS[name],
            lw=1.8,
        )
    ax.set_xlabel("Year")
    ax.set_ylabel("Cumulative burned-pixel events")
    ax.set_title("Cumulative burns over 1850–2000")
    ax.legend(frameon=False, ncol=2)
    ax.grid(True, alpha=0.3)
    savefig(fig, "07_cumulative_burns.png")


def plot_lat_profile(data: dict[str, dict]) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, d in data.items():
        row_sum = d["freq"].sum(axis=1)
        ycoord = None
        ylab = "Y index"
        if d["lat"] is not None and d["lat"].ndim == 2:
            finite = np.isfinite(d["lat"])
            if finite.any():
                lat_sum = np.nansum(np.where(finite, d["lat"], 0.0), axis=1)
                lat_cnt = finite.sum(axis=1).astype(float)
                lat_row = np.divide(
                    lat_sum,
                    lat_cnt,
                    out=np.full(lat_sum.shape, np.nan),
                    where=lat_cnt > 0,
                )
                if np.isfinite(lat_row).sum() > max(5, lat_row.size // 10):
                    ycoord = lat_row
                    ylab = "Latitude (°N)"
        if ycoord is None:
            ycoord = np.arange(row_sum.size, dtype=float)
        ax.plot(row_sum, ycoord, label=name, color=COLORS[name], lw=1.5)
    ax.set_ylabel(ylab)
    ax.set_xlabel("Total burn events (sum over X and years)")
    ax.set_title("Meridional profile of burn frequency")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, alpha=0.3)
    savefig(fig, "08_latitudinal_profile.png")


def plot_summary_bars(data: dict[str, dict]) -> None:
    names = list(data.keys())
    x = np.arange(len(names))
    metrics = [
        ("total_burns", "Total burn events", "09a_summary_total_burns.png"),
        ("mean_annual", "Mean burns / year", "09b_summary_mean_annual.png"),
        ("pixels_ever", "Pixels burned at least once", "09c_summary_pixels_ever.png"),
        ("jday_mean", "Mean burn DOY", "09d_summary_mean_jday.png"),
    ]
    # Combined 2x2
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    for ax, (key, title, _) in zip(axes.ravel(), metrics):
        vals = [data[n][key] for n in names]
        bars = ax.bar(x, vals, color=[COLORS[n] for n in names], edgecolor="none")
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=15, ha="right")
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.3)
        for b, v in zip(bars, vals):
            ax.text(
                b.get_x() + b.get_width() / 2,
                b.get_height(),
                f"{v:.0f}" if key != "jday_mean" else f"{v:.1f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    fig.suptitle("Summary metrics across fire products", fontsize=12)
    savefig(fig, "09_summary_metrics.png")


def plot_example_years(data: dict[str, dict], years_show=(1850, 1900, 1950, 2000)) -> None:
    # One figure per selected year: 2x2 models
    for yr in years_show:
        fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
        for ax, (name, d) in zip(axes.ravel(), data.items()):
            if yr not in set(d["years"].tolist()):
                ax.set_visible(False)
                continue
            ti = int(np.where(d["years"] == yr)[0][0])
            burn = d["burn"][ti]
            # show jday where burn else 0
            jmap = np.zeros(burn.shape, dtype=float)
            jmap[burn] = d["jday"][ti][burn]
            jmap[~burn] = np.nan
            im = ax.imshow(
                jmap,
                origin="upper",
                aspect="auto",
                cmap="plasma",
                vmin=1,
                vmax=365,
                interpolation="nearest",
            )
            n = int(burn.sum())
            ax.set_title(f"{name}  year={yr}  burns={n}")
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
        fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.7, label="DOY")
        fig.suptitle(f"Burn map by DOY — {yr}", fontsize=12)
        savefig(fig, f"10_example_year_{yr}.png")


def plot_overlap_stable(data: dict[str, dict]) -> None:
    """Of pixels that burn in stable at least once, how often do GCMs also burn?"""
    stab_ever = data["stable"]["ever"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    rows = []
    for ax, name in zip(axes, [k for k in data if k != "stable"]):
        g_ever = data[name]["ever"]
        both = stab_ever & g_ever
        only_s = stab_ever & ~g_ever
        only_g = ~stab_ever & g_ever
        # categorical map: 0 none, 1 only stable, 2 only gcm, 3 both
        cat = np.zeros(stab_ever.shape, dtype=np.int16)
        cat[only_s] = 1
        cat[only_g] = 2
        cat[both] = 3
        cmap = matplotlib.colors.ListedColormap(
            ["#f0f0f0", "#4D4D4D", COLORS[name], "#E7298A"]
        )
        ax.imshow(cat, origin="upper", aspect="auto", cmap=cmap, vmin=0, vmax=3)
        ax.set_title(name)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        n_s = int(stab_ever.sum())
        n_g = int(g_ever.sum())
        n_b = int(both.sum())
        rows.append(
            {
                "gcm": name,
                "stable_ever": n_s,
                "gcm_ever": n_g,
                "both": n_b,
                "overlap_of_stable_pct": 100.0 * n_b / n_s if n_s else None,
                "overlap_of_gcm_pct": 100.0 * n_b / n_g if n_g else None,
            }
        )
        ax.text(
            0.02,
            0.02,
            f"both={n_b:,}\n∩/stable={100*n_b/n_s:.1f}%",
            transform=ax.transAxes,
            fontsize=8,
            va="bottom",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="none"),
        )
    fig.suptitle(
        "Ever-burned footprint overlap vs stable (grey=stable only, color=GCM only, pink=both)",
        fontsize=11,
    )
    savefig(fig, "11_everburn_overlap_vs_stable.png")
    return rows


def main() -> None:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    data = {name: load_fire(path, name) for name, path in DATASETS.items()}

    # Align years check
    year0 = data["stable"]["years"]
    for name, d in data.items():
        if d["years"].shape != year0.shape or not np.array_equal(d["years"], year0):
            print(f"WARNING: year axis differs for {name}")

    print("\nPlotting ...")
    plot_annual(data)
    plot_spatial_freq(data)
    plot_freq_diff_vs_stable(data)
    plot_jday(data)
    gap_summary = plot_reburn_gaps(data)
    plot_cumulative(data)
    plot_lat_profile(data)
    plot_summary_bars(data)
    plot_example_years(data)
    overlap_rows = plot_overlap_stable(data)

    summary = {
        "datasets": {
            name: {
                "path": d["path"],
                "years": [int(d["years"][0]), int(d["years"][-1])],
                "shape": list(d["burn"].shape),
                "total_burns": d["total_burns"],
                "mean_annual_burns": d["mean_annual"],
                "pixels_ever_burned": d["pixels_ever"],
                "jday_mean": d["jday_mean"],
                "jday_median": d["jday_med"],
                "jday_min": float(np.nanmin(d["jday_burn"])) if d["jday_burn"].size else None,
                "jday_max": float(np.nanmax(d["jday_burn"])) if d["jday_burn"].size else None,
                "freq_max": float(d["freq"].max()),
                "reburn_gaps": gap_summary.get(name),
            }
            for name, d in data.items()
        },
        "overlap_vs_stable": overlap_rows,
        "notes": [
            "stable_full_v3 uses 20y→151y tile + ≈1/8 burn thinning; GCM files are native 151y draws.",
            "Expect fewer total burns in stable than GCM products for that reason.",
            "DOY is 1-based noleap (1..365).",
        ],
    }
    out_json = OUTDIR / "summary.json"
    out_json.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\nWrote {out_json}")
    print("Done.")


if __name__ == "__main__":
    main()
