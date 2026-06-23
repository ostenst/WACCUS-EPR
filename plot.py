"""EMA result figures — each plot function saves one figure; main() generates all."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

EPR_FEE_LEVELS = [100, 200, 300, 400, 500]
EPR_DESIGN_ORDER = ["Mitigation", "Recovery", "Replacement"]
MEOH_TO_CO2EQ = 44.0 / 32.0  # kt MeOH → kt CO₂eq (full oxidation stoichiometry)
FIG2_HIGH_COLOR = "#62A7A6"  # high CRC/ETS or pmethanol
FIG2_LOW_COLOR = "#DE4968"  # low CRC/ETS or pmethanol
FIG4_CCS_COLOR = "#4477AA"
FIG4_CCU_COLOR = "#62A7A6"
FEE_COLORS = {
    fee: plt.cm.magma(t)
    for fee, t in zip(EPR_FEE_LEVELS, np.linspace(0.1, 0.8, len(EPR_FEE_LEVELS)))
}

KPI_LABELS = {
    "KPI1": "Plants financed [n]",
    "KPI2": "Fossil CO₂ stored [kt/a]",
    "KPI3": "Biogenic CO₂ stored [kt/a]",
    "KPI4": "Fossil methanol [kt/a]",
    "KPI5": "Biogenic methanol [kt/a]",
    "KPI6": "Carbon treated [ktC/a]",
    "KPI7": "Residual fossil CO₂ [kt/a]",
    "KPI8": "Hub combustor CO₂f [kt/a]",
    "KPI9": "Hub combustor CO₂b [kt/a]",
    "KPI10": "New power (capacity or hub demand) [MW]",
    "KPI11": "New power [TWh/a]",
    "KPI12": "Plastic supply [Mtpl/a]",
    "KPI13": "EPR fee [EUR/tpl]",
    "KPI14": "Available subsidies [MEUR/a]",
    "KPI15": "Remaining subsidies [MEUR/a]",
    "KPI16": "Granulate price increase [%]",
    "KPI17": "Products price increase [%]",
}


def load_ema_results(
    path: str = "results/results.csv",
    experiments_path: str = "results/experiments.csv",
    outcomes_path: str = "results/outcomes.csv",
) -> pd.DataFrame:
    """Load combined EMA results, or merge experiments + outcomes."""
    if Path(path).is_file():
        return pd.read_csv(path)

    exp_path = Path(experiments_path)
    out_path = Path(outcomes_path)
    if exp_path.is_file() and out_path.is_file():
        experiments = pd.read_csv(exp_path)
        outcomes = pd.read_csv(out_path)
        return pd.concat([experiments.reset_index(drop=True), outcomes], axis=1)

    raise FileNotFoundError(
        f"No results found. Run controller.py first, or provide {path} "
        f"(or both {experiments_path} and {outcomes_path})."
    )


def _scenario_level_df(results_df: pd.DataFrame, debug: bool = False) -> pd.DataFrame:
    """One row per scenario (KPI12/KPI14 are identical across policies within a scenario)."""
    df = results_df.copy()
    if "scenario_id" in df.columns:
        out = df.drop_duplicates(subset=["scenario_id"], keep="first")
    elif {"EPR_fee", "KPI12", "KPI14"}.issubset(df.columns):
        out = df.drop_duplicates(subset=["EPR_fee", "KPI12", "KPI14"], keep="first")
    else:
        out = df
    if debug:
        print(f"_scenario_level_df: {len(results_df)} runs -> {len(out)} scenarios")
    return out


def _median_bands_by_x(
    x: pd.Series,
    y: pd.Series,
    n_bins: int = 15,
    p_lo: float = 0.05,
    p_hi: float = 0.95,
    debug: bool = False,
) -> pd.DataFrame:
    """Bin by x (KPI14), return mean x with median y and percentile band."""
    frame = pd.DataFrame({"x": x, "y": y}).dropna().sort_values("x")
    if frame.empty:
        return pd.DataFrame(columns=["x", "y_med", "y_lo", "y_hi"])

    if len(frame) >= n_bins:
        frame = frame.copy()
        frame["_bin"] = pd.qcut(frame["x"], q=n_bins, duplicates="drop")
        grouped = frame.groupby("_bin", observed=True).agg(
            x=("x", "mean"),
            y_med=("y", "median"),
            y_lo=("y", lambda s: float(s.quantile(p_lo))),
            y_hi=("y", lambda s: float(s.quantile(p_hi))),
        )
    else:
        grouped = frame.rename(columns={"y": "y_med"}).copy()
        grouped["y_lo"] = grouped["y_med"]
        grouped["y_hi"] = grouped["y_med"]

    if debug:
        print(f"_median_bands_by_x: {len(frame)} points -> {len(grouped)} bins")
    return grouped.reset_index(drop=True)


def _plot_median_band(ax, grouped: pd.DataFrame, color, label: str) -> None:
    """Median line with percentile fill on ax."""
    grouped = grouped.sort_values("x")
    ax.plot(grouped["x"], grouped["y_med"], color=color, linewidth=2.2, label=label, zorder=4)
    ax.fill_between(
        grouped["x"],
        grouped["y_lo"],
        grouped["y_hi"],
        color=color,
        alpha=0.22,
        linewidth=0,
        zorder=3,
    )


def _pct_financed_by_x(
    x: pd.Series,
    kpi1: pd.Series,
    target: float = 10.0,
    n_bins: int = 15,
    debug: bool = False,
) -> pd.DataFrame:
    """Bin by x (KPI14), return mean x with % of rows where KPI1 equals target."""
    frame = pd.DataFrame({"x": x, "kpi1": kpi1}).dropna().sort_values("x")
    if frame.empty:
        return pd.DataFrame(columns=["x", "pct_financed"])

    frame = frame.copy()
    frame["_financed"] = np.isclose(frame["kpi1"], target)
    if len(frame) >= n_bins:
        frame["_bin"] = pd.qcut(frame["x"], q=n_bins, duplicates="drop")
        grouped = frame.groupby("_bin", observed=True).agg(
            x=("x", "mean"),
            pct_financed=("_financed", "mean"),
        )
    else:
        grouped = pd.DataFrame(
            {
                "x": [frame["x"].mean()],
                "pct_financed": [frame["_financed"].mean()],
            }
        )

    grouped["pct_financed"] *= 100.0
    if debug:
        print(f"_pct_financed_by_x: {len(frame)} points -> {len(grouped)} bins")
    return grouped.reset_index(drop=True)


def _plot_pct_line(
    ax,
    grouped: pd.DataFrame,
    color: str,
    label: str,
    linestyle: str = "-",
) -> None:
    """Solid/dashed line of % financed vs KPI14 bin centres."""
    grouped = grouped.sort_values("x")
    ax.plot(
        grouped["x"],
        grouped["pct_financed"],
        color=color,
        linewidth=2.2,
        linestyle=linestyle,
        label=label,
        zorder=4,
    )


def _style_boxplot(bp, colors):
    """Apply face colors to boxplot patches."""
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_edgecolor("0.25")
        patch.set_alpha(0.88)
    for element in ("whiskers", "caps"):
        for artist in bp[element]:
            artist.set_color("0.35")
    for artist in bp["medians"]:
        artist.set_color("0.15")
        artist.set_linewidth(1.5)


def fig1_upstream_impacts(
    results_df: pd.DataFrame,
    out_path: str = "results/fig1_upstream_impacts.png",
    show: bool = False,
    debug: bool = False,
) -> plt.Figure | None:
    """
    Two-panel figure:
      (1) KPI14 vs KPI12 by EPR fee — deterministic lines + scenario scatter.
      (2) KPI16 & KPI17 vs EPR fee — grouped boxplots, shared y-axis [%].
    """
    needed = {"EPR_fee", "KPI12", "KPI14", "KPI16", "KPI17"}
    if not needed.issubset(results_df.columns):
        raise KeyError(f"results missing columns: {needed - set(results_df.columns)}")

    df = _scenario_level_df(results_df, debug=debug)
    df = df[list(needed)].copy()
    df["EPR_fee"] = pd.to_numeric(df["EPR_fee"], errors="coerce").astype("Int64")
    for col in ("KPI12", "KPI14", "KPI16", "KPI17"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna()

    fee_levels = [f for f in EPR_FEE_LEVELS if f in df["EPR_fee"].unique()]
    if not fee_levels:
        fee_levels = sorted(df["EPR_fee"].unique())

    fig, (ax_supply, ax_prices) = plt.subplots(1, 2, figsize=(10, 5.5))
    plotted_lines = 0

    for fee in fee_levels:
        sub = df.loc[df["EPR_fee"] == fee].sort_values("KPI12")
        if sub.empty:
            continue

        color = FEE_COLORS.get(fee, plt.cm.magma(0.5))
        ax_supply.scatter(
            sub["KPI12"],
            sub["KPI14"],
            s=18,
            color=color,
            alpha=0.35,
            edgecolors="none",
            zorder=3,
        )
        x_line = np.array([sub["KPI12"].min(), sub["KPI12"].max()])
        ax_supply.plot(
            x_line,
            x_line * fee,
            color=color,
            linewidth=2.2,
            label=f"{fee} EUR/tpl",
            zorder=4,
        )
        plotted_lines += 1

    if plotted_lines == 0:
        plt.close(fig)
        if debug:
            print("fig1_upstream_impacts: no data to plot")
        return None

    ax_supply.set_xlabel(KPI_LABELS["KPI12"], fontsize=13)
    ax_supply.set_ylabel(KPI_LABELS["KPI14"], fontsize=13)
    ax_supply.set_title(
        "Available subsidies vs plastic supply\n"
        r"KPI14 = KPI12 × EPR fee",
        fontsize=13,
    )
    ax_supply.tick_params(labelsize=11)
    ax_supply.legend(title="EPR fee", fontsize=9, title_fontsize=9, loc="best")
    ax_supply.grid(axis="both", linestyle="--", alpha=0.4)

    # Panel 2: KPI16 & KPI17 boxplots by fee — color matches panel 1 fee lines
    box_width = 0.32
    offsets = (-box_width / 2, box_width / 2)
    all_data = []
    positions = []
    colors = []

    for i, fee in enumerate(fee_levels):
        center = float(i)
        fee_color = FEE_COLORS.get(fee, plt.cm.magma(0.5))
        for kpi, offset in zip(("KPI17", "KPI16"), offsets):
            vals = df.loc[df["EPR_fee"] == fee, kpi].dropna().values
            all_data.append(vals if len(vals) else [])
            positions.append(center + offset)
            colors.append(fee_color)

    bp = ax_prices.boxplot(
        all_data,
        positions=positions,
        widths=box_width * 0.85,
        patch_artist=True,
        manage_ticks=False,
    )
    _style_boxplot(bp, colors)

    ax_prices.set_xticks(range(len(fee_levels)))
    ax_prices.set_xticklabels([str(f) for f in fee_levels], fontsize=11)
    ax_prices.set_xlabel("EPR fee [EUR/tpl]", fontsize=13)
    ax_prices.set_ylabel("Price increase [%]", fontsize=13)
    ax_prices.set_title(
        "Pass-through to granulate & product prices\n"
        "by scenario uncertainty",
        fontsize=13,
    )
    ax_prices.tick_params(axis="y", labelsize=11)
    ax_prices.grid(axis="y", linestyle="--", alpha=0.4)
    ax_prices.text(
        0.40,
        0.06,
        "Lower boxes — product price increase [%]",
        transform=ax_prices.transAxes,
        fontsize=10,
        va="bottom",
        ha="left",
    )
    ax_prices.text(
        0.35,
        0.94,
        "Upper boxes — granulate price increase [%]",
        transform=ax_prices.transAxes,
        fontsize=10,
        va="top",
        ha="left",
    )

    fig.tight_layout()

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    if debug:
        print(f"Saved {out_path}")
    if show:
        plt.show()
    return fig


def fig2_carbon_treated(
    results_df: pd.DataFrame,
    out_path: str = "results/fig2_carbon_treated.png",
    n_bins: int = 15,
    band_p_lo: float = 0.25,
    band_p_hi: float = 0.75,
    show: bool = False,
    debug: bool = False,
) -> plt.Figure | None:
    """
    Three policy panels: total carbon treated [ktCO₂eq/a] vs KPI14, two curves per panel.

    Band default: 25th–75th percentile (IQR) of y within each KPI14 bin — middle half
    of scenarios at similar subsidy levels, less dominated by unfunded vs fully funded tails.

    Mitigation — KPI2+KPI3 [ktCO₂/a]:
      (1) CRC > 150 and ETS > 150
      (2) CRC < 150 and ETS < 150

    Recovery & Replacement — (KPI4+KPI5) × 44/32 [ktCO₂eq/a]:
      (1) pmethanol > 700 EUR/t
      (2) pmethanol < 700 EUR/t
    """
    needed = {
        "EPR_design", "KPI2", "KPI3", "KPI4", "KPI5", "KPI14",
        "CRC", "ETS", "pmethanol",
    }
    if not needed.issubset(results_df.columns):
        raise KeyError(f"results missing columns: {needed - set(results_df.columns)}")

    df = results_df[list(needed)].copy()
    df["KPI14"] = pd.to_numeric(df["KPI14"], errors="coerce")
    df["CRC"] = pd.to_numeric(df["CRC"], errors="coerce")
    df["ETS"] = pd.to_numeric(df["ETS"], errors="coerce")
    df["pmethanol"] = pd.to_numeric(df["pmethanol"], errors="coerce")
    for col in ("KPI2", "KPI3", "KPI4", "KPI5"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["EPR_design", "KPI14"])

    subset_colors = (FIG2_HIGH_COLOR, FIG2_LOW_COLOR)  # high split first, low second
    fig, axes = plt.subplots(1, 3, figsize=(10, 5), sharey=True)
    any_panel = False

    panel_config = {
        "Mitigation": {
            "y": lambda d: d["KPI2"] + d["KPI3"],
            "splits": [
                ("CRC & ETS > 150", lambda d: (d["CRC"] > 150) & (d["ETS"] > 150)),
                ("CRC & ETS < 150", lambda d: (d["CRC"] < 150) & (d["ETS"] < 150)),
            ],
        },
        "Recovery": {
            "y": lambda d: (d["KPI4"] + d["KPI5"]) * MEOH_TO_CO2EQ,
            "splits": [
                ("pmethanol > 700 EUR/t", lambda d: d["pmethanol"] > 700),
                ("pmethanol < 700 EUR/t", lambda d: d["pmethanol"] < 700),
            ],
        },
        "Replacement": {
            "y": lambda d: (d["KPI4"] + d["KPI5"]) * MEOH_TO_CO2EQ,
            "splits": [
                ("pmethanol > 700 EUR/t", lambda d: d["pmethanol"] > 700),
                ("pmethanol < 700 EUR/t", lambda d: d["pmethanol"] < 700),
            ],
        },
    }

    for ax, policy in zip(axes, EPR_DESIGN_ORDER):
        base = df.loc[df["EPR_design"] == policy].copy()
        if base.empty:
            ax.set_title(policy, fontsize=13)
            if debug:
                print(f"fig2_carbon_treated: no rows for {policy}")
            continue

        cfg = panel_config[policy]
        base["y"] = cfg["y"](base)
        plotted = 0

        for (label, mask_fn), color in zip(cfg["splits"], subset_colors):
            sub = base.loc[mask_fn(base)].dropna(subset=["KPI14", "y"])
            if len(sub) < 2:
                if debug:
                    print(f"fig2_carbon_treated {policy} {label}: n={len(sub)}, skip")
                continue

            grouped = _median_bands_by_x(
                sub["KPI14"],
                sub["y"],
                n_bins=n_bins,
                p_lo=band_p_lo,
                p_hi=band_p_hi,
                debug=debug,
            )
            if grouped.empty:
                continue
            _plot_median_band(ax, grouped, color, label)
            plotted += 1

        ax.set_title(policy, fontsize=13)
        ax.set_xlabel(KPI_LABELS["KPI14"], fontsize=12)
        ax.tick_params(labelsize=11)
        ax.grid(axis="both", linestyle="--", alpha=0.4)
        if plotted:
            ax.legend(fontsize=9, loc="best")
            any_panel = True

    if not any_panel:
        plt.close(fig)
        if debug:
            print("fig2_carbon_treated: no data to plot")
        return None

    axes[0].set_ylabel("Carbon treated [ktCO₂eq/a]", fontsize=13)
    fig.suptitle(
        "Total carbon treated vs available subsidies\n"
        f"Band: {int(band_p_lo * 100)}th–{int(band_p_hi * 100)}th percentile within KPI14 bins; "
        "MeOH as CO₂eq (× 44/32)",
        fontsize=14,
        y=1.03,
    )
    fig.tight_layout()

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    if debug:
        print(f"Saved {out_path}")
    if show:
        plt.show()
    return fig


def fig3_finance_robustness(
    results_df: pd.DataFrame,
    out_path: str = "results/fig3_finance_robustness.png",
    n_bins: int = 15,
    show: bool = False,
    debug: bool = False,
) -> plt.Figure | None:
    """
    Three policy panels: % of scenarios financing 10 plants (KPI1 = 10) vs KPI14.

    Mitigation — two lines:
      CRC & ETS > 150 (green), CRC & ETS < 150 (red).

    Recovery & Replacement — four celc × pmethanol combinations:
      green = celc > 50, red = celc < 50; solid = pmethanol > 700, dashed = pmethanol < 700.
    """
    needed = {
        "EPR_design", "KPI1", "KPI14", "CRC", "ETS", "celc", "pmethanol",
    }
    if not needed.issubset(results_df.columns):
        raise KeyError(f"results missing columns: {needed - set(results_df.columns)}")

    df = results_df[list(needed)].copy()
    df["KPI14"] = pd.to_numeric(df["KPI14"], errors="coerce")
    df["KPI1"] = pd.to_numeric(df["KPI1"], errors="coerce")
    df["CRC"] = pd.to_numeric(df["CRC"], errors="coerce")
    df["ETS"] = pd.to_numeric(df["ETS"], errors="coerce")
    df["celc"] = pd.to_numeric(df["celc"], errors="coerce")
    df["pmethanol"] = pd.to_numeric(df["pmethanol"], errors="coerce")
    df = df.dropna(subset=["EPR_design", "KPI14"])

    fig, axes = plt.subplots(1, 3, figsize=(10, 5), sharey=True)
    any_panel = False

    panel_config = {
        "Mitigation": [
            ("CRC & ETS > 150", lambda d: (d["CRC"] > 150) & (d["ETS"] > 150), FIG2_HIGH_COLOR, "-"),
            ("CRC & ETS < 150", lambda d: (d["CRC"] < 150) & (d["ETS"] < 150), FIG2_LOW_COLOR, "-"),
        ],
        "Recovery": [
            ("celc > 50, pmethanol > 700", lambda d: (d["celc"] > 50) & (d["pmethanol"] > 700), FIG2_HIGH_COLOR, "-"),
            ("celc > 50, pmethanol < 700", lambda d: (d["celc"] > 50) & (d["pmethanol"] < 700), FIG2_HIGH_COLOR, "--"),
            ("celc < 50, pmethanol > 700", lambda d: (d["celc"] < 50) & (d["pmethanol"] > 700), FIG2_LOW_COLOR, "-"),
            ("celc < 50, pmethanol < 700", lambda d: (d["celc"] < 50) & (d["pmethanol"] < 700), FIG2_LOW_COLOR, "--"),
        ],
        "Replacement": [
            ("celc > 50, pmethanol > 700", lambda d: (d["celc"] > 50) & (d["pmethanol"] > 700), FIG2_HIGH_COLOR, "-"),
            ("celc > 50, pmethanol < 700", lambda d: (d["celc"] > 50) & (d["pmethanol"] < 700), FIG2_HIGH_COLOR, "--"),
            ("celc < 50, pmethanol > 700", lambda d: (d["celc"] < 50) & (d["pmethanol"] > 700), FIG2_LOW_COLOR, "-"),
            ("celc < 50, pmethanol < 700", lambda d: (d["celc"] < 50) & (d["pmethanol"] < 700), FIG2_LOW_COLOR, "--"),
        ],
    }

    for ax, policy in zip(axes, EPR_DESIGN_ORDER):
        base = df.loc[df["EPR_design"] == policy].copy()
        if base.empty:
            ax.set_title(policy, fontsize=13)
            if debug:
                print(f"fig3_finance_robustness: no rows for {policy}")
            continue

        plotted = 0
        for label, mask_fn, color, linestyle in panel_config[policy]:
            sub = base.loc[mask_fn(base)].dropna(subset=["KPI14", "KPI1"])
            if len(sub) < 2:
                if debug:
                    print(f"fig3_finance_robustness {policy} {label}: n={len(sub)}, skip")
                continue

            grouped = _pct_financed_by_x(
                sub["KPI14"],
                sub["KPI1"],
                target=10.0,
                n_bins=n_bins,
                debug=debug,
            )
            if grouped.empty:
                continue
            _plot_pct_line(ax, grouped, color, label, linestyle=linestyle)
            plotted += 1

        ax.set_title(policy, fontsize=13)
        ax.set_xlabel(KPI_LABELS["KPI14"], fontsize=12)
        ax.tick_params(labelsize=11)
        ax.grid(axis="both", linestyle="--", alpha=0.4)
        ax.set_ylim(0.0, 100.0)
        if plotted:
            ax.legend(fontsize=8, loc="best")
            any_panel = True

    if not any_panel:
        plt.close(fig)
        if debug:
            print("fig3_finance_robustness: no data to plot")
        return None

    axes[0].set_ylabel("10 plants financed [%]", fontsize=13)
    fig.suptitle(
        "Finance robustness — share of scenarios with all 10 plants financed\n"
        "within KPI14 bins",
        fontsize=14,
        y=1.03,
    )
    fig.tight_layout()

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    if debug:
        print(f"Saved {out_path}")
    if show:
        plt.show()
    return fig


def _plant_cost_columns(plants_df: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    """CCS/CCU cost column names and plant display labels in plants_df order."""
    from model import safe_plant_key

    ccs_cols, ccu_cols, labels = [], [], []
    for name in plants_df["Name"]:
        key = safe_plant_key(name)
        ccs_cols.append(f"plant_{key}_ccs_cost_eur_tco2")
        ccu_cols.append(f"plant_{key}_ccu_cost_eur_tmeoh")
        labels.append(str(name).replace(" ", "\n"))
    return ccs_cols, ccu_cols, labels


def fig4_plant_costs(
    results_df: pd.DataFrame,
    plants_path: str = "data/plants_clean.csv",
    out_path: str = "results/fig4_plant_costs.png",
    show: bool = False,
    debug: bool = False,
) -> plt.Figure | None:
    """
    Two-panel figure:
      (1) Per-plant levelized CCS [EUR/tCO₂] and CCU [EUR/t MeOH] boxplots (all scenarios).
      (2) Replacement levelized methanol cost vs number of plants financed/replaced (KPI1).
    """
    plants_df = pd.read_csv(plants_path)
    ccs_cols, ccu_cols, plant_labels = _plant_cost_columns(plants_df)
    needed = set(ccs_cols + ccu_cols + ["EPR_design", "KPI1", "replacement_cost"])
    missing = needed - set(results_df.columns)
    if missing:
        raise KeyError(
            f"results missing columns: {missing}. "
            "Re-run controller.py to populate per-plant cost outcomes."
        )

    df = results_df.copy()
    df["KPI1"] = pd.to_numeric(df["KPI1"], errors="coerce")
    df["replacement_cost"] = pd.to_numeric(df["replacement_cost"], errors="coerce")

    mit = df.loc[df["EPR_design"] == "Mitigation"]
    rec = df.loc[df["EPR_design"] == "Recovery"]
    repl = df.loc[df["EPR_design"] == "Replacement"]

    fig, (ax_plants, ax_repl) = plt.subplots(2, 1, figsize=(10, 9))
    n_plants = len(plant_labels)
    box_width = 0.34
    offsets = (-box_width / 2, box_width / 2)
    ccs_data, ccu_data = [], []

    for i in range(n_plants):
        ccs_vals = mit[ccs_cols[i]].dropna().values
        ccu_vals = rec[ccu_cols[i]].dropna().values
        ccs_data.append(ccs_vals if len(ccs_vals) else [])
        ccu_data.append(ccu_vals if len(ccu_vals) else [])

    bp_ccs = ax_plants.boxplot(
        ccs_data,
        positions=[i + offsets[0] for i in range(n_plants)],
        widths=box_width * 0.9,
        patch_artist=True,
        manage_ticks=False,
    )
    bp_ccu = ax_plants.boxplot(
        ccu_data,
        positions=[i + offsets[1] for i in range(n_plants)],
        widths=box_width * 0.9,
        patch_artist=True,
        manage_ticks=False,
    )
    _style_boxplot(bp_ccs, [FIG4_CCS_COLOR] * n_plants)
    _style_boxplot(bp_ccu, [FIG4_CCU_COLOR] * n_plants)

    ax_plants.set_xticks(range(n_plants))
    ax_plants.set_xticklabels(plant_labels, fontsize=9, rotation=35, ha="right")
    ax_plants.set_ylabel("Levelized cost [EUR/t]", fontsize=13)
    ax_plants.set_title(
        "Per-plant levelized costs across scenarios\n"
        "CCS (Mitigation) and CCU (Recovery)",
        fontsize=13,
    )
    ax_plants.tick_params(axis="y", labelsize=11)
    ax_plants.grid(axis="y", linestyle="--", alpha=0.4)
    ax_plants.legend(
        handles=[
            plt.Rectangle((0, 0), 1, 1, facecolor=FIG4_CCS_COLOR, alpha=0.88, edgecolor="0.25"),
            plt.Rectangle((0, 0), 1, 1, facecolor=FIG4_CCU_COLOR, alpha=0.88, edgecolor="0.25"),
        ],
        labels=["CCS [EUR/tCO₂]", "CCU [EUR/t MeOH]"],
        fontsize=10,
        loc="upper right",
    )

    kpi_levels = [k for k in range(11) if (repl["KPI1"] == k).any()]
    repl_data = [
        repl.loc[np.isclose(repl["KPI1"], k), "replacement_cost"].dropna().values
        for k in kpi_levels
    ]
    if not any(len(d) for d in repl_data):
        if debug:
            print("fig4_plant_costs: no replacement_cost data")
        plt.close(fig)
        return None

    bp_repl = ax_repl.boxplot(
        repl_data,
        positions=list(kpi_levels),
        widths=0.55,
        patch_artist=True,
        manage_ticks=False,
    )
    _style_boxplot(bp_repl, [FIG2_HIGH_COLOR] * len(kpi_levels))

    ax_repl.set_xticks(kpi_levels)
    ax_repl.set_xticklabels([str(int(k)) for k in kpi_levels], fontsize=11)
    ax_repl.set_xlabel("Plants financed / replaced [n]", fontsize=13)
    ax_repl.set_ylabel("Levelized methanol cost [EUR/t MeOH]", fontsize=13)
    ax_repl.set_title(
        "Replacement hub cost vs fleet size\n"
        "subsidized cases only (replacement_cost)",
        fontsize=13,
    )
    ax_repl.tick_params(axis="y", labelsize=11)
    ax_repl.grid(axis="y", linestyle="--", alpha=0.4)

    fig.suptitle("Plant and replacement levelized costs", fontsize=14, y=1.01)
    fig.tight_layout()

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    if debug:
        print(f"Saved {out_path}")
    if show:
        plt.show()
    return fig


def main(show: bool = False, debug: bool = False) -> list[str]:
    """Generate all EMA figures."""
    results_df = load_ema_results()
    saved: list[str] = []

    fig = fig1_upstream_impacts(results_df, show=show, debug=debug)
    if fig is not None:
        saved.append("results/fig1_upstream_impacts.png")
        if not show:
            plt.close(fig)

    fig = fig2_carbon_treated(results_df, show=show, debug=debug)
    if fig is not None:
        saved.append("results/fig2_carbon_treated.png")
        if not show:
            plt.close(fig)

    fig = fig3_finance_robustness(results_df, show=show, debug=debug)
    if fig is not None:
        saved.append("results/fig3_finance_robustness.png")
        if not show:
            plt.close(fig)

    fig = fig4_plant_costs(results_df, show=show, debug=debug)
    if fig is not None:
        saved.append("results/fig4_plant_costs.png")
        if not show:
            plt.close(fig)

    return saved


if __name__ == "__main__":
    paths = main(show=False, debug=True)
    if paths:
        print("Saved:", ", ".join(paths))
    else:
        print("No figures saved.")
