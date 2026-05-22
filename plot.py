"""EMA result figures: KPI boxplots by EPR design and EPR fee."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

EPR_DESIGN_ORDER = ["Mitigation", "Recovery", "Replacement"]
FEE_SENSITIVITY_KPIS = ["KPI1", "KPI6", "KPI11", "KPI16"]

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
    "KPI10": "New power capacity [MWe]",
    "KPI11": "New power [TWh/a]",
    "KPI12": "Plastic supply [Mtpl/a]",
    "KPI13": "EPR fee [EUR/tpl]",
    "KPI14": "Available subsidies [MEUR/a]",
    "KPI15": "Remaining subsidies [MEUR/a]",
    "KPI16": "Granulate price increase [%]",
    "KPI17": "Products price increase [%]",
}

DESIGN_COLORS = {
    design: plt.cm.magma(t)
    for design, t in zip(
        EPR_DESIGN_ORDER,
        np.linspace(0.25, 0.85, len(EPR_DESIGN_ORDER)),
    )
}


def _style_boxplot(bp, colors):
    """Apply face colors to boxplot patches (magma palette)."""
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


def _fee_bin_midpoint(interval):
    return 0.5 * (interval.left + interval.right)


def plot_kpi_boxplots_by_design(df, kpi_keys, title, out_path, ncols=3, debug=False):
    """Boxplots of KPIs grouped by EPR_design (one panel per KPI)."""
    n_kpi = len(kpi_keys)
    nrows = int(np.ceil(n_kpi / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.8 * nrows))
    axes = np.atleast_1d(axes).ravel()
    box_colors = [DESIGN_COLORS[d] for d in EPR_DESIGN_ORDER]

    for ax, kpi in zip(axes, kpi_keys):
        data = [
            df.loc[df["EPR_design"] == design, kpi].dropna().values
            for design in EPR_DESIGN_ORDER
        ]
        bp = ax.boxplot(
            data, tick_labels=EPR_DESIGN_ORDER, widths=0.55, patch_artist=True
        )
        _style_boxplot(bp, box_colors)
        ax.set_title(KPI_LABELS.get(kpi, kpi), fontsize=12)
        ax.set_ylabel(kpi, fontsize=11)
        ax.tick_params(axis="x", labelsize=10, rotation=15)
        ax.tick_params(axis="y", labelsize=10)

    for ax in axes[n_kpi:]:
        ax.set_visible(False)

    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=DESIGN_COLORS[d], edgecolor="0.25", alpha=0.88)
        for d in EPR_DESIGN_ORDER
    ]
    fig.legend(handles, EPR_DESIGN_ORDER, loc="upper center", ncol=3, fontsize=11, frameon=False)
    fig.suptitle(title, fontsize=14, y=1.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    if debug:
        print(f"Saved {out_path}")
    return fig


def plot_kpi_boxplots_by_fee(
    df,
    design,
    kpi_keys,
    out_path,
    n_fee_bins=5,
    debug=False,
):
    """Boxplots of KPIs (y) vs binned EPR_fee (x) for one EPR_design."""
    sub = df.loc[df["EPR_design"] == design].copy()
    if sub.empty:
        return None

    n_unique = sub["EPR_fee"].nunique()
    n_bins = min(n_fee_bins, max(2, n_unique))
    try:
        sub["fee_bin"] = pd.qcut(sub["EPR_fee"], q=n_bins, duplicates="drop")
    except ValueError:
        sub["fee_bin"] = pd.cut(sub["EPR_fee"], bins=n_bins)

    bin_order = sorted(sub["fee_bin"].dropna().unique(), key=_fee_bin_midpoint)
    tick_labels = [f"{int(b.left)}–{int(b.right)}" for b in bin_order]
    fee_colors = [DESIGN_COLORS[design]] * len(bin_order)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.ravel()
    for ax, kpi in zip(axes, kpi_keys):
        data = [
            sub.loc[sub["fee_bin"] == fee_bin, kpi].dropna().values
            for fee_bin in bin_order
        ]
        bp = ax.boxplot(data, tick_labels=tick_labels, widths=0.55, patch_artist=True)
        _style_boxplot(bp, fee_colors)
        ax.set_title(KPI_LABELS.get(kpi, kpi), fontsize=12)
        ax.set_ylabel(kpi, fontsize=11)
        ax.set_xlabel("EPR fee [EUR/tpl]", fontsize=11)
        ax.tick_params(axis="x", labelsize=10, rotation=20)
        ax.tick_params(axis="y", labelsize=10)

    fig.suptitle(f"{design}: KPIs vs EPR fee", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    if debug:
        print(f"Saved {out_path}")
    return fig


def plot_all_ema_kpi_figures(results_df, show=True, debug=False):
    """Generate and optionally display all EMA KPI boxplot figures."""
    mass_kpis = [f"KPI{i}" for i in range(1, 10)]
    other_kpis = [f"KPI{i}" for i in range(10, 18)]

    plot_kpi_boxplots_by_design(
        results_df,
        mass_kpis,
        title="Mass & carbon KPIs by EPR design",
        out_path="results/kpi_boxplots_mass_by_design.png",
        ncols=3,
        debug=debug,
    )
    plot_kpi_boxplots_by_design(
        results_df,
        other_kpis,
        title="Power, policy & economics KPIs by EPR design",
        out_path="results/kpi_boxplots_policy_by_design.png",
        ncols=4,
        debug=debug,
    )

    for design in EPR_DESIGN_ORDER:
        slug = design.lower()
        plot_kpi_boxplots_by_fee(
            results_df,
            design,
            FEE_SENSITIVITY_KPIS,
            out_path=f"results/kpi_boxplots_{slug}_by_fee.png",
            debug=debug,
        )

    
    if show:
        plt.show()


def load_ema_results(
    path="results/results.csv",
    experiments_path="results/experiments.csv",
    outcomes_path="results/outcomes.csv",
):
    """Load EMA results: combined CSV if present, else merge experiments + outcomes."""
    from pathlib import Path

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


if __name__ == "__main__":
    results_df = load_ema_results()
    plot_all_ema_kpi_figures(results_df, show=True, debug=True)
