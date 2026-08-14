#!/usr/bin/env python3
"""
Compare UKESM explicit-fire products:
  - v3: Helene's 30-yr reburn
  - v4: Stefano's 1-yr reburn
  - v5: Joshua's algorithm

Uses native 0.5° grids for area/pixel time series; climate-padded *_new.nc
for spatial burn frequency.

Usage:
  python compare_v3_v4_v5.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset, num2date

OUTDIR = Path(__file__).resolve().parent
FIGDIR = OUTDIR / "figures"
BASE = Path("/mnt/exacloud/cchang_woodwellclimate_org/wiemip/weimip_inputs")

V3_NATIVE = BASE / "explicit-fire_UKESM1-0-LL_v3.nc"
V4_NATIVE = BASE / "explicit-fire_UKESM1-0-LL_v4.nc"
V5_NATIVE = BASE / "explicit-fire_UKESM1-0-LL_v5.nc"
V3_NEW = BASE / "explicit-fire_UKESM1-0-LL_v3_new.nc"
V4_NEW = BASE / "explicit-fire_UKESM1-0-LL_v4_new.nc"
V5_NEW = BASE / "explicit-fire_UKESM1-0-LL_v5_new.nc"

R_EARTH_KM = 6371.0

# Legend labels (exact wording requested)
L_V3 = "v3: Helene's 30-yr reburn"
L_V4 = "v4: Stefano's 1-yr reburn"
L_V5 = "v5: Joshua's algorithm"

COLORS = {
    L_V3: "#7570B3",
    L_V4: "#D95F02",
    L_V5: "#1B9E77",
}


def cell_area_km2(lat_1d: np.ndarray, lon_1d: np.ndarray) -> np.ndarray:
    lat = np.asarray(lat_1d, dtype=np.float64)
    lon = np.asarray(lon_1d, dtype=np.float64)
    dlat = float(np.abs(np.diff(lat)).mean()) if lat.size > 1 else 0.5
    dlon = float(np.abs(np.diff(lon)).mean()) if lon.size > 1 else 0.5
    a_row = (
        (R_EARTH_KM**2)
        * np.cos(np.deg2rad(lat))
        * np.deg2rad(dlat)
        * np.deg2rad(dlon)
    )
    return np.broadcast_to(a_row[:, None], (lat.size, lon.size)).copy()


def years_from_nc(ds) -> np.ndarray:
    t = ds["time"]
    dates = num2date(
        t[:], units=t.units, calendar=getattr(t, "calendar", "noleap")
    )
    return np.array([int(d.year) for d in dates], dtype=np.int32)


def load_realized(path: Path) -> dict:
    print(f"Loading {path.name} ...")
    with Dataset(path) as ds:
        years = years_from_nc(ds)
        Y = np.asarray(ds["Y"][:], dtype=np.float64)
        X = np.asarray(ds["X"][:], dtype=np.float64)
        area = cell_area_km2(Y, X)
        burn = np.asarray(ds["exp_burn_mask"][:], dtype=np.float64) == 1
        jday = np.asarray(ds["exp_jday_of_burn"][:], dtype=np.float64)
        jb = jday[burn]
    return {
        "years": years,
        "pixels": burn.sum(axis=(1, 2)).astype(np.float64),
        "area_km2": (burn * area[None, :, :]).sum(axis=(1, 2)),
        "freq": burn.sum(axis=0).astype(np.float64),
        "ever": burn.any(axis=0),
        "total_burns": int(burn.sum()),
        "jday_mean": float(np.nanmean(jb)) if jb.size else float("nan"),
        "jday_max": float(np.nanmax(jb)) if jb.size else None,
        "shape": list(burn.shape),
    }


def load_v5_overrides(path: Path) -> dict:
    """FRI overrides: exp_pred_prob < 0 (sign-flipped original p)."""
    print(f"Loading v5 overrides from {path.name} ...")
    with Dataset(path) as ds:
        years = years_from_nc(ds)
        Y = np.asarray(ds["Y"][:], dtype=np.float64)
        X = np.asarray(ds["X"][:], dtype=np.float64)
        area = cell_area_km2(Y, X)
        prob = np.asarray(ds["exp_pred_prob"][:], dtype=np.float64)
        mask = np.asarray(ds["exp_burn_mask"][:], dtype=np.float64) == 1
    override = prob < 0.0
    kept = mask & (prob > 0.0)
    abs_p = np.where(override, -prob, 0.0)
    n_ov_xy = override.sum(axis=0).astype(np.float64)
    sum_abs = abs_p.sum(axis=0)
    mean_abs = np.full(n_ov_xy.shape, np.nan, dtype=np.float64)
    hit = n_ov_xy > 0
    mean_abs[hit] = sum_abs[hit] / n_ov_xy[hit]
    return {
        "years": years,
        "override_pixels": override.sum(axis=(1, 2)).astype(np.float64),
        "kept_pixels": kept.sum(axis=(1, 2)).astype(np.float64),
        "override_area_km2": (override * area[None, :, :]).sum(axis=(1, 2)),
        "override_freq": n_ov_xy,
        "override_mean_abs_p": mean_abs,
        "override_ever": override.any(axis=0),
        "kept_freq": kept.sum(axis=0).astype(np.float64),
        "n_override": int(override.sum()),
        "n_kept": int(kept.sum()),
        "mean_abs_p_all_overrides": float(abs_p[override].mean())
        if override.any()
        else float("nan"),
    }


def savefig(fig: plt.Figure, name: str) -> None:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    p = FIGDIR / name
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {p.name}")


def main() -> None:
    v3 = load_realized(V3_NATIVE)
    v4 = load_realized(V4_NATIVE)
    v5 = load_realized(V5_NATIVE)
    assert list(v3["years"]) == list(v4["years"]) == list(v5["years"])
    years = v3["years"]
    series = [(L_V3, v3), (L_V4, v4), (L_V5, v5)]

    # ----- 01 time series area + pixels -----
    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True, constrained_layout=True)
    ax = axes[0]
    for lab, d in series:
        ax.plot(years, d["area_km2"], color=COLORS[lab], lw=1.5, label=lab)
    ax.set_ylabel("Burn area (km² / year)")
    ax.set_title("UKESM annual burn area: v3 vs v4 vs v5")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for lab, d in series:
        ax.plot(years, d["pixels"], color=COLORS[lab], lw=1.5, label=lab)
    ax.set_ylabel("Burned pixels / year")
    ax.set_title("UKESM annual burned pixel count")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ker = np.ones(11) / 11
    for lab, d in series:
        ax.plot(
            years,
            np.convolve(d["area_km2"], ker, mode="same"),
            color=COLORS[lab],
            lw=2,
            label=lab,
        )
    ax.set_xlabel("Year")
    ax.set_ylabel("Burn area (km² / year)")
    ax.set_title("Burn area — 11-year moving mean")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)
    savefig(fig, "01_area_pixels_timeseries.png")

    # ----- 02 ratios to v4 (Stefano-style baseline) -----
    fig, ax = plt.subplots(figsize=(11, 4.2), constrained_layout=True)
    r3 = v3["area_km2"] / np.maximum(v4["area_km2"], 1e-9)
    r5 = v5["area_km2"] / np.maximum(v4["area_km2"], 1e-9)
    ax.plot(years, r3, color=COLORS[L_V3], lw=1.5, label="v3 / v4")
    ax.plot(years, r5, color=COLORS[L_V5], lw=1.5, label="v5 / v4")
    ax.axhline(1.0, color="k", ls="--", lw=1, label="v4 (=1)")
    ax.set_xlabel("Year")
    ax.set_ylabel("Area ratio vs v4")
    ax.set_title("Burn area relative to v4 (Stefano's 1-yr reburn)")
    ax.set_ylim(0, 1.6)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(True, alpha=0.3)
    savefig(fig, "02_ratio_to_v4.png")

    # ----- 03 summary bars -----
    labels = [L_V3, L_V4, L_V5]
    data = [v3, v4, v5]
    cols = [COLORS[k] for k in labels]
    metrics = [
        ("total", "Total burn events (sum of annual)", [float(s["pixels"].sum()) for s in data]),
        ("mean_pix", "Mean burned pixels / year", [float(s["pixels"].mean()) for s in data]),
        ("mean_area", "Mean burn area (km² / year)", [float(s["area_km2"].mean()) for s in data]),
        (
            "1990s_area",
            "1990–1999 mean area (km² / year)",
            [
                float(s["area_km2"][(years >= 1990) & (years <= 1999)].mean())
                for s in data
            ],
        ),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5), constrained_layout=True)
    x = np.arange(3)
    short = ["v3\nHelene 30-yr", "v4\nStefano 1-yr", "v5\nJoshua"]
    for ax, (_, title, vals) in zip(axes.ravel(), metrics):
        bars = ax.bar(x, vals, color=cols, edgecolor="none")
        ax.set_xticks(x)
        ax.set_xticklabels(short, fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.grid(True, axis="y", alpha=0.3)
        for b, v in zip(bars, vals):
            ax.text(
                b.get_x() + b.get_width() / 2,
                b.get_height(),
                f"{v:.0f}",
                ha="center",
                va="bottom",
                fontsize=7,
            )
    fig.suptitle("Summary: v3 vs v4 vs v5 (UKESM)", fontsize=12)
    savefig(fig, "03_summary_metrics.png")

    # ----- 04 spatial frequency (padded) -----
    print("Loading padded grids for spatial compare ...")
    v3p = load_realized(V3_NEW)
    v4p = load_realized(V4_NEW)
    v5p = load_realized(V5_NEW)
    vmax = max(
        float(v3p["freq"].max()),
        float(v4p["freq"].max()),
        float(v5p["freq"].max()),
        1.0,
    )
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), constrained_layout=True)
    for ax, (name, d) in zip(
        axes,
        [
            (L_V3, v3p),
            (L_V4, v4p),
            (L_V5, v5p),
        ],
    ):
        im = ax.imshow(
            d["freq"],
            origin="upper",
            aspect="auto",
            cmap="YlOrRd",
            vmin=0,
            vmax=vmax,
            interpolation="nearest",
        )
        ax.set_title(
            f"{name}\nmax/pix={int(d['freq'].max())}  ever={int(d['ever'].sum()):,}",
            fontsize=9,
        )
        fig.colorbar(im, ax=ax, shrink=0.85, label="burns / pixel (151y)")
    fig.suptitle("Spatial burn frequency (climate-padded grids)", fontsize=12)
    savefig(fig, "04_spatial_freq_v3_v4_v5.png")

    # ----- 05 pairwise frequency diffs -----
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), constrained_layout=True)
    pairs = [
        ("v4 − v3", v4p["freq"] - v3p["freq"]),
        ("v5 − v3", v5p["freq"] - v3p["freq"]),
        ("v5 − v4", v5p["freq"] - v4p["freq"]),
    ]
    lim = max(float(np.nanmax(np.abs(d))) for _, d in pairs)
    lim = max(lim, 1.0)
    for ax, (title, diff) in zip(axes, pairs):
        im = ax.imshow(
            diff,
            origin="upper",
            aspect="auto",
            cmap="RdBu_r",
            vmin=-lim,
            vmax=lim,
            interpolation="nearest",
        )
        ax.set_title(f"{title}\nmean Δ={diff.mean():.2f}", fontsize=10)
        fig.colorbar(im, ax=ax, shrink=0.85, label="Δ burns / pixel")
    fig.suptitle("Spatial frequency differences (padded)", fontsize=12)
    savefig(fig, "05_spatial_freq_diffs.png")

    # ----- 06 cumulative -----
    fig, ax = plt.subplots(figsize=(11, 4.5), constrained_layout=True)
    for lab, d in series:
        ax.plot(years, np.cumsum(d["pixels"]), color=COLORS[lab], lw=1.7, label=lab)
    ax.set_xlabel("Year")
    ax.set_ylabel("Cumulative burned-pixel events")
    ax.set_title("Cumulative burns 1850–2000")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(True, alpha=0.3)
    savefig(fig, "06_cumulative_pixels.png")

    # ----- 07–08 v5 FRI overrides (exp_pred_prob < 0) -----
    ov_n = load_v5_overrides(V5_NATIVE)
    ov_p = load_v5_overrides(V5_NEW)
    assert list(ov_n["years"]) == list(years)

    fig, axes = plt.subplots(2, 1, figsize=(11, 7.5), sharex=True, constrained_layout=True)
    ax = axes[0]
    ax.plot(
        years,
        ov_n["kept_pixels"],
        color=COLORS[L_V5],
        lw=1.6,
        label="v5 kept burns (mask=1)",
    )
    ax.plot(
        years,
        ov_n["override_pixels"],
        color="#E7298A",
        lw=1.6,
        label="v5 FRI overrides (prob < 0)",
    )
    ax.set_ylabel("Pixels / year")
    ax.set_title("v5 Joshua: kept burns vs FRI-overridden candidates")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    cand = ov_n["kept_pixels"] + ov_n["override_pixels"]
    frac = ov_n["override_pixels"] / np.maximum(cand, 1e-9)
    ax.plot(years, frac, color="#E7298A", lw=1.6, label="override / (kept + override)")
    ax.axhline(float(frac.mean()), color="k", ls="--", lw=1, label=f"mean={frac.mean():.2f}")
    ax.set_xlabel("Year")
    ax.set_ylabel("Override fraction")
    ax.set_ylim(0, 1.05)
    ax.set_title("Fraction of annual candidates cleared by recurrence")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(True, alpha=0.3)
    savefig(fig, "07_v5_fri_override_timeseries.png")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), constrained_layout=True)
    im0 = axes[0].imshow(
        ov_p["kept_freq"],
        origin="upper",
        aspect="auto",
        cmap="YlGn",
        vmin=0,
        vmax=max(float(ov_p["kept_freq"].max()), 1.0),
        interpolation="nearest",
    )
    axes[0].set_title(
        f"Kept burns / pixel\nmax={int(ov_p['kept_freq'].max())}  "
        f"ever={int((ov_p['kept_freq'] > 0).sum()):,}",
        fontsize=9,
    )
    fig.colorbar(im0, ax=axes[0], shrink=0.85, label="kept burns (151y)")

    im1 = axes[1].imshow(
        ov_p["override_freq"],
        origin="upper",
        aspect="auto",
        cmap="RdPu",
        vmin=0,
        vmax=max(float(ov_p["override_freq"].max()), 1.0),
        interpolation="nearest",
    )
    axes[1].set_title(
        f"FRI overrides / pixel (prob < 0)\nmax={int(ov_p['override_freq'].max())}  "
        f"ever={int(ov_p['override_ever'].sum()):,}",
        fontsize=9,
    )
    fig.colorbar(im1, ax=axes[1], shrink=0.85, label="overrides (151y)")

    im2 = axes[2].imshow(
        ov_p["override_mean_abs_p"],
        origin="upper",
        aspect="auto",
        cmap="viridis",
        vmin=0,
        vmax=max(float(np.nanmax(ov_p["override_mean_abs_p"])), 1e-6),
        interpolation="nearest",
    )
    axes[2].set_title(
        "Mean |exp_pred_prob| of overrides\n"
        f"(global mean={ov_p['mean_abs_p_all_overrides']:.4f})",
        fontsize=9,
    )
    fig.colorbar(im2, ax=axes[2], shrink=0.85, label="mean |p|")
    fig.suptitle(
        "v5 FRI overrides — climate-padded grid (Joshua sign-flip diagnostic)",
        fontsize=12,
    )
    savefig(fig, "08_v5_fri_override_spatial.png")

    # CSV + JSON
    csv_path = OUTDIR / "timeseries_v3_v4_v5.csv"
    with csv_path.open("w") as f:
        f.write(
            "year,v3_pixels,v3_area_km2,v4_pixels,v4_area_km2,"
            "v5_pixels,v5_area_km2,ratio_v3_to_v4_area,ratio_v5_to_v4_area,"
            "v5_override_pixels,v5_override_area_km2,v5_override_fraction\n"
        )
        for i, y in enumerate(years):
            f.write(
                f"{int(y)},{v3['pixels'][i]:.6f},{v3['area_km2'][i]:.6f},"
                f"{v4['pixels'][i]:.6f},{v4['area_km2'][i]:.6f},"
                f"{v5['pixels'][i]:.6f},{v5['area_km2'][i]:.6f},"
                f"{r3[i]:.6f},{r5[i]:.6f},"
                f"{ov_n['override_pixels'][i]:.6f},{ov_n['override_area_km2'][i]:.6f},"
                f"{frac[i]:.6f}\n"
            )
    print(f"wrote {csv_path.name}")

    def decade(arr, y0, y1):
        m = (years >= y0) & (years <= y1)
        return float(arr[m].mean())

    summary = {
        "labels": {"v3": L_V3, "v4": L_V4, "v5": L_V5},
        "filters": {
            "v3": "Helene ≥30-year reburn lockout + ≤1 burn/pixel/year",
            "v4": "Stefano ≤1 burn/pixel/year only (no multi-year FRI)",
            "v5": "Joshua recurrence (stable burn-in + RecurrenceLimit on GCM candidates)",
        },
        "totals": {
            "v3_sum_pixels": float(v3["pixels"].sum()),
            "v4_sum_pixels": float(v4["pixels"].sum()),
            "v5_sum_pixels": float(v5["pixels"].sum()),
            "v3_mean_area_km2": float(v3["area_km2"].mean()),
            "v4_mean_area_km2": float(v4["area_km2"].mean()),
            "v5_mean_area_km2": float(v5["area_km2"].mean()),
            "1850_v3_area": float(v3["area_km2"][0]),
            "1850_v4_area": float(v4["area_km2"][0]),
            "1850_v5_area": float(v5["area_km2"][0]),
            "1990s_v3_area": decade(v3["area_km2"], 1990, 1999),
            "1990s_v4_area": decade(v4["area_km2"], 1990, 1999),
            "1990s_v5_area": decade(v5["area_km2"], 1990, 1999),
            "mean_ratio_v3_to_v4": float(r3.mean()),
            "mean_ratio_v5_to_v4": float(r5.mean()),
        },
        "v5_fri_overrides": {
            "definition": "exp_pred_prob < 0 (sign-flipped original candidate probability)",
            "n_kept": ov_n["n_kept"],
            "n_override": ov_n["n_override"],
            "override_fraction": float(
                ov_n["n_override"] / max(ov_n["n_kept"] + ov_n["n_override"], 1)
            ),
            "mean_abs_pred_prob_overrides": ov_n["mean_abs_p_all_overrides"],
            "pixels_ever_overridden": int(ov_n["override_ever"].sum()),
        },
        "paths": {
            "v3_native": str(V3_NATIVE),
            "v4_native": str(V4_NATIVE),
            "v5_native": str(V5_NATIVE),
            "v3_new": str(V3_NEW),
            "v4_new": str(V4_NEW),
            "v5_new": str(V5_NEW),
        },
    }
    (OUTDIR / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("wrote summary.json")

    t = summary["totals"]
    o = summary["v5_fri_overrides"]
    print("\n=== Key numbers (area km²/yr) ===")
    print(f"1850:   v3 {t['1850_v3_area']:.0f}  v4 {t['1850_v4_area']:.0f}  v5 {t['1850_v5_area']:.0f}")
    print(f"1990s:  v3 {t['1990s_v3_area']:.0f}  v4 {t['1990s_v4_area']:.0f}  v5 {t['1990s_v5_area']:.0f}")
    print(f"Mean:   v3 {t['v3_mean_area_km2']:.0f}  v4 {t['v4_mean_area_km2']:.0f}  v5 {t['v5_mean_area_km2']:.0f}")
    print(f"Sum pix: v3 {t['v3_sum_pixels']:.0f}  v4 {t['v4_sum_pixels']:.0f}  v5 {t['v5_sum_pixels']:.0f}")
    print(f"Mean ratio to v4: v3={t['mean_ratio_v3_to_v4']:.3f}  v5={t['mean_ratio_v5_to_v4']:.3f}")
    print(
        f"v5 FRI overrides: {o['n_override']:,} cleared / "
        f"{o['n_kept'] + o['n_override']:,} candidates "
        f"({100 * o['override_fraction']:.1f}%); "
        f"mean |p|={o['mean_abs_pred_prob_overrides']:.4f}"
    )
    print("Done.")


if __name__ == "__main__":
    main()
