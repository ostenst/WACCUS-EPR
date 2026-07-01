"""Map per-plant heat output and new power capacity from EMA results (KPI1 = 10 scenarios)."""

import os

import geopandas as gpd
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse

from model import safe_plant_key

HUB_LAT = 58.527430
HUB_LON = 15.079128
POLICIES = ["Mitigation", "Recovery", "Replacement"]
MAP_EXTENT = {"x_min": 10.0, "x_max": 20.0, "y_min": 55, "y_max": 61.0}
BUBBLE_SCALE = 1.2
BUBBLE_SCALE_DIVISOR = 10.0
HEAT_COLOR = cm.magma(0.65)
POWER_COLOR = cm.magma(0.12)
HUB_RECYCLE_COLOR = cm.magma(0.0)
OTHER_PLANT_COLOR = "0.55"
OTHER_PLANT_EDGE = "0.25"

# Draw order: first entry = back, last = front (z-order increases with index).
PLOTTING_ORDER = {
    "Mitigation": ["plant_heat", "plant_power"],
    "Recovery": ["plant_power", "plant_heat"],
    "Replacement": ["hub_power", "hub_recycle_power", "plant_heat", "plant_power"],
}


def _normalize_plant_name(name) -> str:
    return str(name).strip().lower()


def _load_other_plants_heat(
    main_names,
    other_path: str = "data/plants_other.csv",
    coords_path: str = "data/plants_other_coordinates.csv",
    debug: bool = False,
) -> pd.DataFrame:
    """Other Swedish CHP sites: coords + heat capacity [MWth] (heat + FGC)."""
    other = pd.read_csv(other_path)
    coords = pd.read_csv(coords_path)
    coords["Latitude"] = pd.to_numeric(coords["Latitude"], errors="coerce")
    coords["Longitude"] = pd.to_numeric(coords["Longitude"], errors="coerce")

    other["_key"] = other["Plant Name"].map(_normalize_plant_name)
    coords["_key"] = coords["Name"].map(_normalize_plant_name)
    merged = other.merge(
        coords[["_key", "Latitude", "Longitude"]],
        on="_key",
        how="inner",
    )
    main_keys = {_normalize_plant_name(n) for n in main_names}
    merged = merged[~merged["_key"].isin(main_keys)].copy()
    merged["heat_mwth"] = (
        pd.to_numeric(merged["Heat output (MWheat)"], errors="coerce")
        + pd.to_numeric(merged["Existing FGC heat output (MWheat)"], errors="coerce")
    )

    if debug:
        missing_coords = other.loc[
            ~other["_key"].isin(coords["_key"]), "Plant Name"
        ].tolist()
        if missing_coords:
            print(f"Other plants without coordinates (skipped): {missing_coords}")
        print(f"Other plants for power map: {len(merged)}")
    return merged


def _plot_other_plants(
    ax,
    other_plants: pd.DataFrame,
    scales: dict,
    zorder: int = 3,
    debug: bool = False,
) -> None:
    """Gray heat bubbles for other CHP sites (reference capacity, not EMA outcomes)."""
    for _, row in other_plants.iterrows():
        mw = float(row["heat_mwth"])
        lon, lat = float(row["Longitude"]), float(row["Latitude"])
        if mw <= 0 or not np.isfinite(lon) or not np.isfinite(lat):
            continue
        area = _scatter_area_from_mw(mw, scales["heat_scale"])
        if area <= 0:
            continue
        if debug:
            print(f"_plot_other_plants: {row['Plant Name']} heat={mw:.1f} MWth")
        ax.scatter(
            [lon],
            [lat],
            s=[area],
            c=[OTHER_PLANT_COLOR],
            alpha=0.75,
            edgecolors=OTHER_PLANT_EDGE,
            linewidths=0.8,
            zorder=zorder,
        )


def plant_metric_columns(plants_df: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    power_cols, heat_cols, carbon_cols = [], [], []
    for name in plants_df["Name"]:
        key = safe_plant_key(name)
        power_cols.append(f"plant_{key}_power_mw")
        heat_cols.append(f"plant_{key}_heat_mwth")
        carbon_cols.append(f"plant_{key}_carbon_ktc")
    return power_cols, heat_cols, carbon_cols


def augment_results_plant_outcomes(
    results_path: str = "results/results.csv",
    experiments_path: str = "results/experiments.csv",
    kpi1_target: float = 10.0,
    debug: bool = False,
) -> pd.DataFrame:
    """Re-run model for KPI1 = kpi1_target rows and append per-plant outcome columns."""
    from controller import ema_WACCUS_EPR, model

    plants_df = pd.read_csv("data/plants_clean.csv")
    power_cols, heat_cols, carbon_cols = plant_metric_columns(plants_df)
    plant_cols = (
        power_cols + heat_cols + carbon_cols
        + ["hub_power_mw", "hub_recycle_power_mw", "hub_carbon_ktc", "q_district_heat_hub"]
    )

    results_df = pd.read_csv(results_path)
    experiments_df = pd.read_csv(experiments_path)
    if len(results_df) != len(experiments_df):
        raise ValueError("results.csv and experiments.csv row counts differ.")

    for col in plant_cols:
        if col not in results_df.columns:
            results_df[col] = np.nan

    mask = np.isclose(results_df["KPI1"], kpi1_target)
    idxs = results_df.index[mask]
    if debug:
        print(f"augment_results_plant_outcomes: re-running {len(idxs)} rows with KPI1={kpi1_target}")

    constants = {c.name: c.value for c in model.constants}
    skip = {"policy", "scenario", "model"}
    for idx in idxs:
        row = experiments_df.loc[idx].drop(labels=[c for c in skip if c in experiments_df.columns])
        out = ema_WACCUS_EPR(**{**constants, **row.to_dict()})
        for col in plant_cols:
            results_df.at[idx, col] = out.get(col, np.nan)

    results_df.to_csv(results_path, index=False)
    if debug:
        print(f"Updated {results_path} with plant outcome columns.")
    return results_df


def load_plant_means(
    results_path: str = "results/results.csv",
    plants_path: str = "data/plants_clean.csv",
    kpi1_target: float = 10.0,
    debug: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Mean per-plant heat [MWth] and power [MW] for KPI1 = kpi1_target, by EPR_design."""
    plants_df = pd.read_csv(plants_path)
    power_cols, heat_cols, _ = plant_metric_columns(plants_df)
    hub_cols = ["hub_power_mw", "hub_recycle_power_mw", "q_district_heat_hub"]
    needed = power_cols + heat_cols + hub_cols + ["EPR_design", "KPI1"]

    if not os.path.isfile(results_path):
        raise FileNotFoundError(f"Missing {results_path}. Run controller.py first.")

    results_df = pd.read_csv(results_path)
    missing = [c for c in needed if c not in results_df.columns]
    if missing:
        raise ValueError(
            "Results file lacks per-plant outcome columns "
            f"({missing[:3]}{'...' if len(missing) > 3 else ''}). "
            "Re-run controller.py, or call augment_results_plant_outcomes()."
        )

    sub = results_df.loc[np.isclose(results_df["KPI1"], kpi1_target)].copy()
    if sub.empty:
        raise ValueError(f"No rows with KPI1 = {kpi1_target} in {results_path}.")

    rows = []
    hub_rows = []
    for policy in POLICIES:
        pol = sub.loc[sub["EPR_design"] == policy]
        if pol.empty:
            if debug:
                print(f"load_plant_means: no KPI1={kpi1_target} rows for {policy}")
            continue

        mean_power = pol[power_cols].mean()
        mean_heat = pol[heat_cols].mean()
        for _, plant in plants_df.iterrows():
            key = safe_plant_key(plant["Name"])
            rows.append(
                {
                    "EPR_design": policy,
                    "Name": plant["Name"],
                    "Latitude": plant["Latitude"],
                    "Longitude": plant["Longitude"],
                    "heat_mwth": float(mean_heat[f"plant_{key}_heat_mwth"]),
                    "power_mw": float(mean_power[f"plant_{key}_power_mw"]),
                }
            )

        hub_rows.append(
            {
                "EPR_design": policy,
                "Latitude": HUB_LAT,
                "Longitude": HUB_LON,
                "heat_mwth": float(pol["q_district_heat_hub"].mean(skipna=True))
                if policy == "Replacement"
                else float("nan"),
                "power_mw": float(pol["hub_power_mw"].mean(skipna=True))
                if policy == "Replacement"
                else float("nan"),
                "recycle_power_mw": float(pol["hub_recycle_power_mw"].mean(skipna=True))
                if policy == "Replacement"
                else float("nan"),
                "n_scenarios": len(pol),
            }
        )

    plant_means = pd.DataFrame(rows)
    hub_means = pd.DataFrame(hub_rows)
    if debug:
        print(
            "load_plant_means:",
            {p: len(sub.loc[sub["EPR_design"] == p]) for p in POLICIES},
        )
    return plant_means, hub_means


def compute_harmonized_scales(
    plant_means: pd.DataFrame,
    hub_means: pd.DataFrame,
    debug: bool = False,
) -> dict:
    """Shared bubble diameter scales for heat and power across all three policy maps."""
    heat_vals = plant_means.loc[plant_means["heat_mwth"] > 0, "heat_mwth"]
    hub_heat = hub_means["heat_mwth"].dropna()
    hub_heat = hub_heat[hub_heat > 0]
    all_heat = pd.concat([heat_vals, hub_heat], ignore_index=True)
    heat_max = float(all_heat.max()) if len(all_heat) else 1.0

    power_vals = plant_means.loc[plant_means["power_mw"] > 0, "power_mw"]
    hub_power = hub_means["power_mw"].dropna()
    hub_power = hub_power[hub_power > 0]
    hub_recycle = hub_means.get("recycle_power_mw", pd.Series(dtype=float)).dropna()
    hub_recycle = hub_recycle[hub_recycle > 0]
    all_power = pd.concat([power_vals, hub_power, hub_recycle], ignore_index=True)
    power_max = float(all_power.max()) if len(all_power) else 1.0

    base = BUBBLE_SCALE / BUBBLE_SCALE_DIVISOR
    scales = {
        "heat_scale": base,
        "power_scale": base,
        "heat_max": heat_max,
        "power_max": power_max,
    }
    if debug:
        print(
            "harmonized scales:",
            f"heat_max={heat_max:.1f} MWth",
            f"power_max={power_max:.1f} MW",
            f"bubble_scale={base:.4f}",
        )
    return scales


def _scatter_area_from_mw(mw: float, bubble_scale: float) -> float:
    """Matplotlib scatter area so marker diameter scales with MW."""
    return float(max(mw, 0.0) * bubble_scale) ** 2


def _marker_radii_data(ax, lon: float, lat: float, area_pts2: float) -> tuple[float, float]:
    """Lon/lat semi-axes [deg] for a visually circular marker."""
    fig = ax.figure
    fig.canvas.draw()
    center = ax.transData.transform((lon, lat))
    radius_pts = np.sqrt(max(area_pts2, 0.0) / np.pi)
    edge_x = ax.transData.inverted().transform(center + np.array([radius_pts, 0.0]))
    edge_y = ax.transData.inverted().transform(center + np.array([0.0, radius_pts]))
    rx = abs(float(edge_x[0] - lon))
    ry = abs(float(edge_y[1] - lat))
    return rx, ry


def _draw_bubble(
    ax,
    lon: float,
    lat: float,
    mw: float,
    kind: str,
    scales: dict,
    *,
    hub: bool = False,
    zorder: int = 4,
    debug: bool = False,
) -> None:
    """Draw one heat or power bubble at (lon, lat). kind: 'heat' | 'power' | 'recycle'."""
    if mw <= 0:
        return

    if kind == "heat":
        color, scale = HEAT_COLOR, scales["heat_scale"]
    elif kind == "recycle":
        color, scale = HUB_RECYCLE_COLOR, scales["power_scale"]
    else:
        color, scale = POWER_COLOR, scales["power_scale"]
    area = _scatter_area_from_mw(mw, scale)
    if area <= 0:
        return

    if debug:
        print(f"_draw_bubble {kind} hub={hub} mw={mw:.1f} z={zorder}")

    edge_kw = {"edgecolor": "black", "linewidth": 1.2 if hub else 0.8}
    hub_ls = (0, (4, 3)) if hub and kind == "power" else "solid"

    if hub:
        rx, ry = _marker_radii_data(ax, lon, lat, area)
        ax.add_patch(
            Ellipse(
                (lon, lat),
                width=2.0 * rx,
                height=2.0 * ry,
                facecolor=color,
                edgecolor="black",
                linewidth=1.2,
                linestyle=hub_ls,
                alpha=0.75,
                zorder=zorder,
            )
        )
    else:
        ax.scatter(
            [lon],
            [lat],
            s=[area],
            c=[color],
            alpha=0.75,
            zorder=zorder,
            **edge_kw,
        )


def _plot_layer(
    ax,
    layer: str,
    plants: pd.DataFrame,
    hub_power: float,
    hub_recycle_power: float,
    scales: dict,
    zorder: int,
    debug: bool = False,
) -> None:
    """Draw one named layer from PLOTTING_ORDER."""
    if layer == "plant_heat":
        for _, row in plants.iterrows():
            _draw_bubble(
                ax, row["Longitude"], row["Latitude"], row["heat_mwth"],
                "heat", scales, zorder=zorder, debug=debug,
            )
    elif layer == "plant_power":
        for _, row in plants.iterrows():
            _draw_bubble(
                ax, row["Longitude"], row["Latitude"], row["power_mw"],
                "power", scales, zorder=zorder, debug=debug,
            )
    elif layer == "hub_power" and np.isfinite(hub_power) and hub_power > 0:
        _draw_bubble(
            ax, HUB_LON, HUB_LAT, hub_power, "power", scales,
            hub=True, zorder=zorder, debug=debug,
        )
    elif layer == "hub_recycle_power" and np.isfinite(hub_recycle_power) and hub_recycle_power > 0:
        _draw_bubble(
            ax, HUB_LON, HUB_LAT, hub_recycle_power, "recycle", scales,
            hub=True, zorder=zorder, debug=debug,
        )


def _find_max_plant_power_site(plants: pd.DataFrame) -> dict | None:
    """Return lon/lat/power of the largest on-site power bubble (excludes hub)."""
    best: dict | None = None
    for _, row in plants.iterrows():
        power = float(row["power_mw"])
        if power <= 0:
            continue
        if best is None or power > best["power_mw"]:
            best = {
                "lon": float(row["Longitude"]),
                "lat": float(row["Latitude"]),
                "power_mw": power,
            }
    return best


def _find_max_power_site(
    plants: pd.DataFrame,
    hub_power: float,
    policy: str,
) -> dict | None:
    """Return lon/lat/power of the largest power bubble on the map (incl. hub on Replacement)."""
    best = _find_max_plant_power_site(plants)

    if policy == "Replacement" and np.isfinite(hub_power) and hub_power > 0:
        if best is None or hub_power > best["power_mw"]:
            best = {
                "lon": HUB_LON,
                "lat": HUB_LAT,
                "power_mw": float(hub_power),
                "hub": True,
            }
    return best


def _annotate_power_capacity(
    ax,
    site: dict,
    scales: dict,
    label: str,
    offset_factor: float = 1.15,
    debug: bool = False,
) -> None:
    """Label a power bubble just above its top edge."""
    power = site["power_mw"]
    lon, lat = site["lon"], site["lat"]
    area = _scatter_area_from_mw(power, scales["power_scale"])
    _, ry = _marker_radii_data(ax, lon, lat, area)
    label_lat = lat + offset_factor * ry

    ax.annotate(
        label,
        xy=(lon, lat),
        xytext=(lon, label_lat),
        ha="center",
        va="bottom",
        fontsize=11,
        color="black",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="none", alpha=0.75),
        zorder=25,
    )
    if debug:
        print(f"_annotate_power_capacity: {label} at ({lon:.3f}, {lat:.3f})")


def plot_power_map(
    policy: str,
    plant_means: pd.DataFrame,
    hub_means: pd.DataFrame,
    scales: dict,
    europe_shp: str = "data/shapefiles/Europe/Europe_merged.shp",
    out_path: str | None = None,
    debug: bool = False,
) -> plt.Figure:
    """One map: heat bubble (MWth) with overlaid power bubble (MW); shared scales."""
    plants = plant_means.loc[plant_means["EPR_design"] == policy].copy()
    if plants.empty:
        raise ValueError(f"No plant means for policy {policy!r}.")

    hub = hub_means.loc[hub_means["EPR_design"] == policy]
    hub_power = float(hub["power_mw"].iloc[0]) if len(hub) else float("nan")
    hub_recycle_power = float(hub["recycle_power_mw"].iloc[0]) if len(hub) else float("nan")
    n_scenarios = int(hub["n_scenarios"].iloc[0]) if len(hub) else 0

    europe = gpd.read_file(europe_shp).to_crs("EPSG:4326")
    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    europe.plot(ax=ax, edgecolor="black", facecolor="whitesmoke", linewidth=0.6)
    ax.set_xlim(MAP_EXTENT["x_min"], MAP_EXTENT["x_max"])
    ax.set_ylim(MAP_EXTENT["y_min"], MAP_EXTENT["y_max"])
    ax.set_aspect(1.90)
    ax.set_xticks([])
    ax.set_yticks([])

    main_names = pd.read_csv("data/plants_clean.csv")["Name"]
    other_plants = _load_other_plants_heat(main_names, debug=debug)
    _plot_other_plants(ax, other_plants, scales, zorder=3, debug=debug)

    draw_order = PLOTTING_ORDER.get(policy)
    if draw_order is None:
        raise KeyError(f"No PLOTTING_ORDER entry for policy {policy!r}")

    for zorder, layer in enumerate(draw_order, start=4):
        _plot_layer(
            ax, layer, plants, hub_power, hub_recycle_power, scales, zorder, debug=debug
        )

    max_power = _find_max_power_site(plants, hub_power, policy)

    if policy == "Replacement" and np.isfinite(hub_recycle_power) and hub_recycle_power > 0:
        # Draw first (lower on map — smaller bubble radius) so max-capacity label can sit above.
        _annotate_power_capacity(
            ax,
            {"lon": HUB_LON, "lat": HUB_LAT, "power_mw": hub_recycle_power},
            scales,
            f"Hub power excl. electrolyzer: {hub_recycle_power:.0f} MW",
            debug=debug,
        )

    if max_power is not None:
        _annotate_power_capacity(
            ax,
            max_power,
            scales,
            f"Highest new capacity: {max_power['power_mw']:.0f} MW",
            debug=debug,
        )

    if policy == "Replacement":
        max_hp = _find_max_plant_power_site(plants)
        if max_hp is not None:
            same_site = (
                max_power is not None
                and max_hp["lon"] == max_power["lon"]
                and max_hp["lat"] == max_power["lat"]
            )
            _annotate_power_capacity(
                ax,
                max_hp,
                scales,
                f"Highest new HP capacity: {max_hp['power_mw']:.0f} MW",
                offset_factor=2.35 if same_site else 1.15,
                debug=debug,
            )

    legend_handles = [
        Line2D(
            [0], [0], marker="o", color="w", markerfacecolor=OTHER_PLANT_COLOR,
            markeredgecolor=OTHER_PLANT_EDGE, markersize=10,
            label="Other CHP heat capacity [MWth]",
        ),
        Line2D(
            [0], [0], marker="o", color="w", markerfacecolor=HEAT_COLOR,
            markeredgecolor="black", markersize=11, label="Heat output [MWth]",
        ),
        Line2D(
            [0], [0], marker="o", color="w", markerfacecolor=POWER_COLOR,
            markeredgecolor="black", markersize=8, label="New power capacity [MW]",
        ),
    ]
    if policy == "Replacement":
        legend_handles.extend([
            Line2D(
                [0], [0], marker="o", color="black", markerfacecolor=POWER_COLOR,
                markeredgecolor="black", markersize=10, linestyle="--",
                label="Gasification hub new power [MW]",
            ),
            Line2D(
                [0], [0], marker="o", color="black", markerfacecolor=HUB_RECYCLE_COLOR,
                markeredgecolor="black", markersize=9, linestyle="--",
                label="Hub power excl. electrolyzer [MW]",
            ),
        ])

    ax.set_title(
        f"{policy} — mean over {n_scenarios} scenarios (KPI1 = 10)\n"
        "Bubble diameter ∝ heat [MWth] or new power [MW]",
        fontsize=13,
    )
    ax.legend(handles=legend_handles, loc="lower left", fontsize=10, framealpha=0.85)

    if out_path is None:
        out_path = f"results/power_map_{policy.lower()}.png"
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=450, bbox_inches="tight")
    if debug:
        print(f"Saved {out_path}")
    return fig


def plot_all_power_maps(
    results_path: str = "results/results.csv",
    out_dir: str = "results",
    show: bool = False,
    debug: bool = False,
) -> list[str]:
    """Build Mitigation, Recovery, and Replacement heat/power maps."""
    plant_means, hub_means = load_plant_means(results_path, debug=debug)
    scales = compute_harmonized_scales(plant_means, hub_means, debug=debug)
    saved = []
    for policy in POLICIES:
        out_path = os.path.join(out_dir, f"power_map_{policy.lower()}.png")
        plot_power_map(
            policy, plant_means, hub_means, scales, out_path=out_path, debug=debug
        )
        saved.append(out_path)
        if show:
            plt.show()
        else:
            plt.close()
    return saved


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Plot EPR heat / power maps from EMA results.")
    parser.add_argument(
        "--augment",
        action="store_true",
        help="Re-run model for KPI1=10 rows to add plant outcome columns to results.csv",
    )
    parser.add_argument("--show", action="store_true", help="Display figures interactively.")
    args = parser.parse_args()

    if args.augment:
        augment_results_plant_outcomes(debug=True)
    paths = plot_all_power_maps(show=args.show, debug=True)
    print("Saved:", ", ".join(paths))
