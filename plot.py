"""EMA result figures — each plot function saves one figure; main() generates all."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

EPR_FEE_LEVELS = [100, 200, 300, 400, 500]
EPR_DESIGN_ORDER = ["Mitigation", "Recovery", "Replacement"]
MEOH_TO_CO2EQ = 44.0 / 32.0  # kt MeOH → kt CO₂eq (full oxidation stoichiometry)
FIG2_HIGH_COLOR = "#62A7A6"  # high celc / pmethanol (Recovery & Replacement)
FIG2_LOW_COLOR = "#DE4968"  # low celc / pmethanol (Recovery & Replacement)
FIG3_CRC_HIGH_COLOR = "black"  # high CRC & ETS (Mitigation)
FIG3_CRC_LOW_COLOR = "0.55"  # low CRC & ETS (Mitigation)
FIG2_KPI7_COLOR = plt.cm.magma(0.65)  # mean residual fossil CO₂ per KPI14 bin
MAC_COVERED_COLOR = "0.55"  # abatement cost covered by carbon price
MAC_UNCOVERED_COLOR = FIG2_KPI7_COLOR  # residual abatement cost
MAC_DEFAULT_COSTS = [90.0, 100.0, 110.0, 120.0, 135.0, 150.0, 155.0, 165.0, 180.0, 195.0]
MAC_DEFAULT_WIDTHS = [420.0, 400.0, 380.0, 350.0, 320.0, 300.0, 180.0, 160.0, 170.0, 150.0]
MAC_HIGHLIGHT_COUNT = 6
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
    "KPI12": "Plastic supply [Mtpl p.a.]",
    "KPI13": "EPR fee [€/tpl]",
    "KPI14": "Available subsidies [M€ p.a.]",
    "KPI15": "Remaining subsidies [M€ p.a.]",
    "KPI16": "Granulate price increase [%]",
    "KPI17": "Products price increase [%]",
}


def _policy_panel_title(policy: str) -> str:
    """Panel title with policy name in italics."""
    return rf"$\it{{{policy}}}$"


def _style_conceptual_arrow_axes(
    ax,
    x_max: float,
    y_max: float,
    n_x_ticks: int = 6,
    n_y_ticks: int = 5,
    debug: bool = False,
) -> None:
    """Arrow-style x/y axes with tick marks but no numeric labels."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

    arrow_kw = dict(arrowstyle="-|>", color="black", lw=1.2, shrinkA=0, shrinkB=0)
    ax.annotate("", xy=(x_max, 0.0), xytext=(0.0, 0.0), arrowprops=arrow_kw, clip_on=False)
    ax.annotate("", xy=(0.0, y_max), xytext=(0.0, 0.0), arrowprops=arrow_kw, clip_on=False)

    ax.set_xticks(np.linspace(0.0, x_max, n_x_ticks))
    ax.set_yticks(np.linspace(0.0, y_max, n_y_ticks))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.tick_params(
        axis="both",
        which="major",
        direction="out",
        length=5,
        width=0.9,
        labelbottom=False,
        labelleft=False,
    )
    if debug:
        print(f"_style_conceptual_arrow_axes: x_max={x_max}, y_max={y_max}")


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
    label: str | None = None,
    linestyle: str = "-",
) -> None:
    """Solid/dashed line of % financed vs KPI14 bin centres."""
    grouped = grouped.sort_values("x")
    plot_kw = dict(
        color=color,
        linewidth=2.2,
        linestyle=linestyle,
        zorder=4,
    )
    if label is not None:
        plot_kw["label"] = label
    ax.plot(grouped["x"], grouped["pct_financed"], **plot_kw)


def _fig3_shared_legend_handles() -> list[Line2D]:
    """All fig3 line styles — Mitigation (CRC/ETS) plus CCU elc × MeOH combinations."""
    return [
        Line2D(
            [0], [0], color=FIG3_CRC_HIGH_COLOR, lw=2.2, linestyle="-",
            label="CRC & ETS > 150 €/t",
        ),
        Line2D(
            [0], [0], color=FIG3_CRC_LOW_COLOR, lw=2.2, linestyle="-",
            label="CRC & ETS < 150 €/t",
        ),
        Line2D(
            [0], [0], color=FIG2_HIGH_COLOR, lw=2.2, linestyle="-",
            label="Elc. price > 50 €/MWh, MeOH price > 700 €/t",
        ),
        Line2D(
            [0], [0], color=FIG2_HIGH_COLOR, lw=2.2, linestyle="--",
            label="Elc. price > 50 €/MWh, MeOH price < 700 €/t",
        ),
        Line2D(
            [0], [0], color=FIG2_LOW_COLOR, lw=2.2, linestyle="-",
            label="Elc. price < 50 €/MWh, MeOH price > 700 €/t",
        ),
        Line2D(
            [0], [0], color=FIG2_LOW_COLOR, lw=2.2, linestyle="--",
            label="Elc. price < 50 €/MWh, MeOH price < 700 €/t",
        ),
    ]


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


def _visual_jitter(
    x: np.ndarray,
    y: np.ndarray,
    x_span: float,
    y_span: float,
    frac: float = 0.008,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Small random displacement for scatter visibility only (not physical variation)."""
    rng = np.random.default_rng(seed)
    x_scale = frac * (x_span if x_span > 0 else 1.0)
    y_scale = frac * (y_span if y_span > 0 else 1.0)
    return x + rng.normal(0.0, x_scale, len(x)), y + rng.normal(0.0, y_scale, len(y))


def fig1_upstream_impacts(
    results_df: pd.DataFrame,
    out_path: str = "results/fig1_upstream_impacts.png",
    scatter_jitter_frac: float = 0.006,
    show: bool = False,
    debug: bool = False,
) -> plt.Figure | None:
    """
    Two-panel figure:
      (1) KPI14 vs KPI12 by EPR fee — deterministic lines + scenario scatter.
      (2) KPI16 & KPI17 vs EPR fee — grouped boxplots, shared y-axis [%].

    Scatter points in panel 1 are slightly jittered for visibility only.
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
    kpi12_span = float(df["KPI12"].max() - df["KPI12"].min())
    kpi14_span = float(df["KPI14"].max() - df["KPI14"].min())

    for fee in fee_levels:
        sub = df.loc[df["EPR_fee"] == fee].sort_values("KPI12")
        if sub.empty:
            continue

        color = FEE_COLORS.get(fee, plt.cm.magma(0.5))
        x_plot, y_plot = _visual_jitter(
            sub["KPI12"].values,
            sub["KPI14"].values,
            kpi12_span,
            kpi14_span,
            frac=scatter_jitter_frac,
            seed=int(fee),
        )
        ax_supply.scatter(
            x_plot,
            y_plot,
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
            label=f"{fee} €/tpl",
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
    ax_supply.grid(axis="both", linestyle="--", alpha=0.4)

    beccs_auction_subsidy = 163.0  # [M€ p.a.]
    ax_supply.axhline(
        beccs_auction_subsidy,
        color="black",
        linestyle="--",
        linewidth=1.5,
        zorder=5,
    )
    ax_supply.annotate(
        "BECCS auction\nsubsidy 2029-2046",
        xy=(1.95, 50),
        xytext=(4, 4),
        textcoords="offset points",
        fontsize=10,
        ha="left",
        va="bottom",
        color="black",
    )

    ax_supply.legend(title="EPR fee", fontsize=9, title_fontsize=9, loc="best")

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
    ax_prices.set_xlabel("EPR fee [€/tpl]", fontsize=13)
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
        0.035,
        "Lower boxes =\nProduct price increase [%]",
        transform=ax_prices.transAxes,
        fontsize=10,
        va="bottom",
        ha="left",
    )
    ax_prices.text(
        0.40,
        0.85,
        "Upper boxes =\nGranulate price increase [%]",
        transform=ax_prices.transAxes,
        fontsize=10,
        va="top",
        ha="left",
    )

    fig.tight_layout()

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    if debug:
        print(
            f"fig1_upstream_impacts: scatter jitter frac={scatter_jitter_frac} "
            f"(~{scatter_jitter_frac * kpi12_span:.3f} Mt, "
            f"~{scatter_jitter_frac * kpi14_span:.1f} M€)"
        )
        print(f"Saved {out_path}")
    if show:
        plt.show()
    return fig


def _boxplot_by_kpi14(
    df: pd.DataFrame,
    y_col: str = "y",
    overlay_col: str | None = None,
    fossil_frac_col: str | None = None,
    n_bins: int = 5,
    debug: bool = False,
) -> tuple[list[np.ndarray], list[str], list[int], list[float], list[float]]:
    """Quantile-bin KPI14; return box data, labels, positions, overlay means, fossil frac means."""
    use_cols = ["KPI14", y_col]
    if overlay_col is not None:
        use_cols.append(overlay_col)
    if fossil_frac_col is not None:
        use_cols.append(fossil_frac_col)
    frame = df[use_cols].dropna(subset=["KPI14", y_col]).sort_values("KPI14")
    if frame.empty:
        return [], [], [], [], []

    if len(frame) >= n_bins:
        frame = frame.copy()
        frame["_bin"] = pd.qcut(frame["KPI14"], q=n_bins, duplicates="drop")
        grouped = (
            frame.groupby("_bin", observed=True)["KPI14"]
            .mean()
            .sort_values()
        )
    else:
        grouped = pd.Series({0: frame["KPI14"].mean()}, name="KPI14")
        frame = frame.copy()
        frame["_bin"] = 0

    all_data: list[np.ndarray] = []
    labels: list[str] = []
    positions: list[int] = []
    overlay_means: list[float] = []
    fossil_frac_means: list[float] = []
    for i, bin_key in enumerate(grouped.index):
        sub = frame.loc[frame["_bin"] == bin_key]
        vals = sub[y_col].values
        all_data.append(vals if len(vals) else np.array([]))
        labels.append(f"{grouped.iloc[i]:.0f}")
        positions.append(i)
        if overlay_col is not None:
            overlay_means.append(float(sub[overlay_col].mean()))
        if fossil_frac_col is not None:
            fossil_frac_means.append(float(sub[fossil_frac_col].mean(skipna=True)))

    if debug:
        counts = [len(d) for d in all_data]
        print(f"_boxplot_by_kpi14: {len(frame)} rows -> {len(all_data)} bins, n={counts}")
    return all_data, labels, positions, overlay_means, fossil_frac_means


def fig2_carbon_treated(
    results_df: pd.DataFrame,
    out_path: str = "results/fig2_carbon_treated.png",
    n_bins: int = 5,
    show: bool = False,
    debug: bool = False,
) -> plt.Figure | None:
    """
    Three policy panels: carbon treated [ktCO₂eq/a] vs KPI14 — 5 box plots per panel.

    Mitigation — KPI2 + KPI3 [ktCO₂/a]; Recovery & Replacement — (KPI4 + KPI5) × 44/32.
    All scenarios included (no CRC/ETS/pmethanol splits); KPI14 quantile bins per panel.
    Mean KPI7 (residual fossil CO₂, ktCO₂/a ≈ ktCO₂eq) overlaid as dots per bin.
    Box fill: grayscale by mean fossil fraction in bin (Mitigation: KPI2/(KPI2+KPI3);
    Recovery/Replacement: KPI4/(KPI4+KPI5); darker = more fossil).
    """
    needed = {"EPR_design", "KPI2", "KPI3", "KPI4", "KPI5", "KPI7", "KPI14"}
    if not needed.issubset(results_df.columns):
        raise KeyError(f"results missing columns: {needed - set(results_df.columns)}")

    df = results_df[list(needed)].copy()
    df["KPI14"] = pd.to_numeric(df["KPI14"], errors="coerce")
    for col in ("KPI2", "KPI3", "KPI4", "KPI5", "KPI7"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["EPR_design", "KPI14"])

    panel_y = {
        "Mitigation": lambda d: d["KPI2"] + d["KPI3"],
        "Recovery": lambda d: (d["KPI4"] + d["KPI5"]) * MEOH_TO_CO2EQ,
        "Replacement": lambda d: (d["KPI4"] + d["KPI5"]) * MEOH_TO_CO2EQ,
    }
    fig, axes = plt.subplots(1, 3, figsize=(10, 5), sharey=True)
    any_panel = False

    for ax, policy in zip(axes, EPR_DESIGN_ORDER):
        base = df.loc[df["EPR_design"] == policy].copy()
        if base.empty:
            ax.set_title(_policy_panel_title(policy), fontsize=13)
            if debug:
                print(f"fig2_carbon_treated: no rows for {policy}")
            continue

        base["y"] = panel_y[policy](base)
        if policy == "Mitigation":
            total_c = base["KPI2"] + base["KPI3"]
            base["fossil_frac"] = np.where(total_c > 0, base["KPI2"] / total_c, np.nan)
        else:
            total_m = base["KPI4"] + base["KPI5"]
            base["fossil_frac"] = np.where(total_m > 0, base["KPI4"] / total_m, np.nan)

        box_data, x_labels, positions, kpi7_means, fossil_means = _boxplot_by_kpi14(
            base,
            y_col="y",
            overlay_col="KPI7",
            fossil_frac_col="fossil_frac",
            n_bins=n_bins,
            debug=debug,
        )
        if not box_data or not any(len(d) for d in box_data):
            ax.set_title(_policy_panel_title(policy), fontsize=13)
            if debug:
                print(f"fig2_carbon_treated: no box data for {policy}")
            continue

        if debug:
            for pos, label, ff in zip(positions, x_labels, fossil_means):
                ff_str = f"{ff:.4f}" if np.isfinite(ff) else "nan"
                print(
                    f"fig2_carbon_treated [{policy}] bin {pos + 1} "
                    f"(KPI14~{label} M€): fossil_frac={ff_str}"
                )

        colors = [
            plt.cm.gray_r(f) if np.isfinite(f) else (0.75, 0.75, 0.75, 1.0)
            for f in fossil_means
        ]
        bp = ax.boxplot(
            box_data,
            positions=positions,
            widths=0.55,
            patch_artist=True,
            manage_ticks=False,
        )
        _style_boxplot(bp, colors)

        if kpi7_means:
            ax.scatter(
                positions,
                kpi7_means,
                s=72,
                c=[FIG2_KPI7_COLOR] * len(positions),
                edgecolors="black",
                linewidths=0.8,
                zorder=5,
            )

        ax.set_xticks(positions)
        ax.set_xticklabels(x_labels, fontsize=10)
        ax.set_title(_policy_panel_title(policy), fontsize=13)
        ax.set_xlabel(f"{KPI_LABELS['KPI14']}", fontsize=12)
        ax.tick_params(axis="y", labelsize=11)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        any_panel = True

    if not any_panel:
        plt.close(fig)
        if debug:
            print("fig2_carbon_treated: no data to plot")
        return None

    axes[0].set_ylabel("Carbon capture capacity [ktCO₂eq p.a.]", fontsize=13)
    # fig.suptitle(
    #     "Total carbon treated vs available subsidies\n"
    #     f"{n_bins} KPI14 quantile bins per policy; box shade = fossil fraction in bin, "
    #     "dots = mean KPI7 (residual fossil CO₂); MeOH as CO₂eq (× 44/32)",
    #     fontsize=14,
    #     y=1.03,
    # )
    sm = plt.cm.ScalarMappable(cmap=plt.cm.gray_r, norm=plt.Normalize(vmin=0.0, vmax=1.0))
    sm.set_array([])
    fig.tight_layout(rect=[0, 0, 0.88, 0.96])
    cbar_ax = fig.add_axes([0.89, 0.14, 0.02, 0.72]) # Adjusts colorbar position and size
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("Fossil fraction (mean)", fontsize=12)
    cbar.ax.tick_params(labelsize=12)
    axes[-1].legend(
        handles=[
            Line2D(
                [0], [0], marker="s", color="w", markerfacecolor="0.5",
                markeredgecolor="0.25", markersize=10, label="Carbon capture capacity\n(CCS/BECCS/methanol)",
            ),
            Line2D(
                [0], [0], marker="o", color="w", markerfacecolor=FIG2_KPI7_COLOR,
                markeredgecolor="black", markersize=9, label="Residual fossil CO₂ (mean)",
            ),
        ],
        loc="upper right",
        fontsize=9,
        framealpha=0.9,
    )

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
    n_bins: int = 8,
    show: bool = False,
    debug: bool = False,
) -> plt.Figure | None:
    """
    Three policy panels: % of scenarios financing 10 plants (KPI1 = 10) vs KPI14.

    Mitigation — CRC & ETS high/low (black/gray). Recovery & Replacement — four
    elc × MeOH combinations (colour + linestyle). Shared legend below all panels.
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
            (lambda d: (d["CRC"] > 150) & (d["ETS"] > 150), FIG3_CRC_HIGH_COLOR, "-"),
            (lambda d: (d["CRC"] < 150) & (d["ETS"] < 150), FIG3_CRC_LOW_COLOR, "-"),
        ],
        "Recovery": [
            (lambda d: (d["celc"] > 50) & (d["pmethanol"] > 700), FIG2_HIGH_COLOR, "-"),
            (lambda d: (d["celc"] > 50) & (d["pmethanol"] < 700), FIG2_HIGH_COLOR, "--"),
            (lambda d: (d["celc"] < 50) & (d["pmethanol"] > 700), FIG2_LOW_COLOR, "-"),
            (lambda d: (d["celc"] < 50) & (d["pmethanol"] < 700), FIG2_LOW_COLOR, "--"),
        ],
        "Replacement": [
            (lambda d: (d["celc"] > 50) & (d["pmethanol"] > 700), FIG2_HIGH_COLOR, "-"),
            (lambda d: (d["celc"] > 50) & (d["pmethanol"] < 700), FIG2_HIGH_COLOR, "--"),
            (lambda d: (d["celc"] < 50) & (d["pmethanol"] > 700), FIG2_LOW_COLOR, "-"),
            (lambda d: (d["celc"] < 50) & (d["pmethanol"] < 700), FIG2_LOW_COLOR, "--"),
        ],
    }

    for ax, policy in zip(axes, EPR_DESIGN_ORDER):
        base = df.loc[df["EPR_design"] == policy].copy()
        if base.empty:
            ax.set_title(_policy_panel_title(policy), fontsize=13)
            if debug:
                print(f"fig3_finance_robustness: no rows for {policy}")
            continue

        plotted = 0
        for mask_fn, color, linestyle in panel_config[policy]:
            sub = base.loc[mask_fn(base)].dropna(subset=["KPI14", "KPI1"])
            if len(sub) < 2:
                if debug:
                    print(f"fig3_finance_robustness {policy}: n={len(sub)}, skip line")
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
            _plot_pct_line(ax, grouped, color, label=None, linestyle=linestyle)
            plotted += 1

        ax.set_title(_policy_panel_title(policy), fontsize=13)
        ax.set_xlabel(KPI_LABELS["KPI14"], fontsize=12)
        ax.tick_params(labelsize=11)
        ax.grid(axis="both", linestyle="--", alpha=0.4)
        ax.set_ylim(0.0, 100.0)
        if plotted:
            any_panel = True

    if not any_panel:
        plt.close(fig)
        if debug:
            print("fig3_finance_robustness: no data to plot")
        return None

    axes[0].set_ylabel("Carbon capture robustness [%]", fontsize=13)
    fig.tight_layout(rect=[0, 0.075, 1, 1])
    fig.legend(
        handles=_fig3_shared_legend_handles(),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.06),
        bbox_transform=fig.transFigure,
        ncol=3,
        fontsize=9,
        framealpha=0.9,
        borderaxespad=0.0,
    )

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0.05)
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


def fig_mac_stylized(
    out_path: str = "results/mac_curve_stylized.png",
    costs: list[float] | None = None,
    widths: list[float] | None = None,
    carbon_price: float = 80.0,
    highlight_count: int = MAC_HIGHLIGHT_COUNT,
    show: bool = False,
    debug: bool = False,
) -> plt.Figure:
    """
    Stylized marginal abatement cost curve: stacked rectangles ordered by
    levelized cost (low → high). Gray base = cost covered by carbon price;
    magma upper section = uncovered abatement cost. The cheapest ``highlight_count``
    options use a dashed hatch on their red sections only.
    """
    if costs is None:
        costs = list(MAC_DEFAULT_COSTS)
    if widths is None:
        widths = list(MAC_DEFAULT_WIDTHS)
    if len(costs) != len(widths):
        raise ValueError("costs and widths must have the same length")

    order = np.argsort(costs)
    costs = [float(costs[i]) for i in order]
    widths = [float(widths[i]) for i in order]

    fig, ax = plt.subplots(figsize=(8, 5))
    x_left = 0.0
    edge_kw = {"edgecolor": "black", "linewidth": 0.8}

    for i, (cost, width) in enumerate(zip(costs, widths)):
        covered_h = min(carbon_price, cost)
        if covered_h > 0:
            ax.add_patch(
                Rectangle(
                    (x_left, 0.0),
                    width,
                    covered_h,
                    facecolor=MAC_COVERED_COLOR,
                    **edge_kw,
                )
            )
        if cost > carbon_price:
            uncovered_kw = {
                **edge_kw,
                "facecolor": MAC_UNCOVERED_COLOR,
            }
            if i < highlight_count:
                uncovered_kw["hatch"] = "----"
            ax.add_patch(
                Rectangle(
                    (x_left, carbon_price),
                    width,
                    cost - carbon_price,
                    **uncovered_kw,
                )
            )
        x_left += width

    total_width = sum(widths)
    y_top = max(costs) * 1.08
    ax.set_xlim(0.0, total_width * 1.05)
    ax.set_ylim(0.0, y_top * 1.05)
    ax.axhline(
        carbon_price,
        color="black",
        linestyle="--",
        linewidth=1.0,
        alpha=0.55,
        zorder=5,
    )
    ax.set_xlabel("Capacity [ktCO2eq p.a.]", fontsize=13)
    ax.set_ylabel("Levelized CCS/methanol cost [€/t]", fontsize=13)
    _style_conceptual_arrow_axes(ax, total_width, y_top, debug=debug)
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    if debug:
        print(
            f"fig_mac_stylized: bars={list(zip(widths, costs))}, "
            f"carbon_price={carbon_price}, highlighted={highlight_count}"
        )
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

    fig = fig_mac_stylized(show=show, debug=debug)
    saved.append("results/mac_curve_stylized.png")
    if not show:
        plt.close(fig)

    return saved


if __name__ == "__main__":
    paths = main(show=False, debug=True)
    if paths:
        print("Saved:", ", ".join(paths))
    else:
        print("No figures saved.")
