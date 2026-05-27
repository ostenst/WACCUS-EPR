"""EMA result figures: KPI boxplots by EPR design and EPR fee."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

EPR_DESIGN_ORDER = ["Mitigation", "Recovery", "Replacement"]
EPR_FEE_LEVELS = [100, 200, 300, 400, 500]
FEE_SENSITIVITY_KPIS = ["KPI1", "KPI6", "KPI11", "KPI16"]

FEE_COLORS = {
    fee: plt.cm.magma(t)
    for fee, t in zip(EPR_FEE_LEVELS, np.linspace(0.2, 0.9, len(EPR_FEE_LEVELS)))
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

# Within-scenario regret: extend by adding entries (kpi + min/max sense)
DEFAULT_REGRET_SPECS = {
    "regret_carbon": {
        "kpi": "KPI7",
        "sense": "min",
        "ylabel": "Regret — residual fossil CO₂ [kt/a]",
    },
    "regret_power": {
        "kpi": "KPI11",
        "sense": "min",
        "ylabel": "Regret — new power [TWh/a]",
    },
    "regret_prices": {
        "kpi": "KPI16",
        "sense": "min",
        "ylabel": "Regret — granulate price increase [%]",
    },
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
        ax.set_ylabel(KPI_LABELS.get(kpi, kpi), fontsize=12)
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
    """Boxplots of KPIs (y) vs categorical EPR_fee levels for one EPR_design."""
    sub = df.loc[df["EPR_design"] == design].copy()
    if sub.empty:
        return None

    fee_levels = sorted(pd.to_numeric(sub["EPR_fee"], errors="coerce").dropna().unique())
    tick_labels = [str(int(fee)) for fee in fee_levels]
    fee_colors = [DESIGN_COLORS[design]] * len(fee_levels)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.ravel()
    for ax, kpi in zip(axes, kpi_keys):
        data = [
            sub.loc[sub["EPR_fee"] == fee, kpi].dropna().values
            for fee in fee_levels
        ]
        bp = ax.boxplot(data, tick_labels=tick_labels, widths=0.55, patch_artist=True)
        _style_boxplot(bp, fee_colors)
        ax.set_ylabel(KPI_LABELS.get(kpi, kpi), fontsize=12)
        ax.set_xlabel("EPR fee [EUR/tpl]", fontsize=11)
        ax.tick_params(axis="x", labelsize=10, rotation=20)
        ax.tick_params(axis="y", labelsize=10)

    fig.suptitle(f"{design}: KPIs vs EPR fee", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    if debug:
        print(f"Saved {out_path}")
    return fig


def calculate_regret(
    results_df,
    regret_specs,
    policies=None,
    scenario_col="scenario_id",
    policy_col="EPR_design",
    debug=False,
):
    """
    Within-scenario regret vs the best policy on each KPI.

    regret_specs: {name: {"kpi": str, "sense": "min"|"max", "ylabel": str (optional)}}
      - sense "min": lower KPI is better → regret = KPI(chosen) − KPI(best)
      - sense "max": higher KPI is better → regret = KPI(best) − KPI(chosen)

    Returns one row per (scenario, policy) with regret columns (≥ 0).
    """
    if policies is None:
        policies = EPR_DESIGN_ORDER
    if scenario_col not in results_df.columns:
        raise KeyError(f"Column {scenario_col!r} missing — run controller.py first.")

    rows = []
    for scenario_id, group in results_df.groupby(scenario_col):
        by_policy = group.set_index(policy_col)
        for policy in policies:
            if policy not in by_policy.index:
                continue
            row = {scenario_col: scenario_id, policy_col: policy}
            if "EPR_fee" in group.columns:
                row["EPR_fee"] = group["EPR_fee"].iloc[0]
            for regret_name, spec in regret_specs.items():
                kpi = spec["kpi"]
                sense = spec["sense"]
                values = by_policy[kpi]
                if sense == "min":
                    best = values.min()
                    row[regret_name] = float(values.loc[policy] - best)
                elif sense == "max":
                    best = values.max()
                    row[regret_name] = float(best - values.loc[policy])
                else:
                    raise ValueError(f"{regret_name}: sense must be 'min' or 'max', got {sense!r}")
            rows.append(row)

    regret_df = pd.DataFrame(rows)
    if debug:
        for name in regret_specs:
            print(f"{name}: mean={regret_df[name].mean():.4g}, max={regret_df[name].max():.4g}")
    return regret_df


def plot_regret_by_policy(
    regret_df,
    regret_specs,
    out_path="results/regret_by_policy.png",
    policy_col="EPR_design",
    fee_col="EPR_fee",
    fee_levels=None,
    show=False,
    debug=False,
):
    """
    Boxplots of regret by EPR_fee (magma colors), grouped by policy within each fee.

    At each fee level on the x-axis: three boxes (Mitigation, Recovery, Replacement).
    """
    if fee_col not in regret_df.columns:
        raise KeyError(f"Column {fee_col!r} missing — include EPR_fee in regret_df.")

    regret_df = regret_df.copy()
    regret_df[fee_col] = regret_df[fee_col].astype(int)

    if fee_levels is None:
        fee_levels = [f for f in EPR_FEE_LEVELS if f in regret_df[fee_col].unique()]
        if not fee_levels:
            fee_levels = sorted(regret_df[fee_col].unique())

    regret_names = list(regret_specs.keys())
    n_panels = len(regret_names)
    ncols = min(3, n_panels)
    nrows = int(np.ceil(n_panels / ncols))
    n_policies = len(EPR_DESIGN_ORDER)
    box_width = 0.22
    group_gap = 1.0

    fig, axes = plt.subplots(nrows, ncols, figsize=(5.0 * ncols, 4.2 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax, regret_name in zip(axes, regret_names):
        all_data = []
        positions = []
        colors = []
        for i, fee in enumerate(fee_levels):
            center = i * group_gap
            offsets = np.linspace(
                -(n_policies - 1) / 2 * box_width,
                (n_policies - 1) / 2 * box_width,
                n_policies,
            )
            for j, design in enumerate(EPR_DESIGN_ORDER):
                mask = (regret_df[fee_col] == fee) & (regret_df[policy_col] == design)
                vals = regret_df.loc[mask, regret_name].dropna().values
                all_data.append(vals if len(vals) else [])
                positions.append(center + offsets[j])
                colors.append(FEE_COLORS.get(fee, plt.cm.magma(0.5)))

        bp = ax.boxplot(
            all_data,
            positions=positions,
            widths=box_width,
            patch_artist=True,
            manage_ticks=False,
        )
        _style_boxplot(bp, colors)

        ax.set_xticks([i * group_gap for i in range(len(fee_levels))])
        ax.set_xticklabels([str(f) for f in fee_levels], fontsize=11)
        ax.set_xlabel("EPR fee [EUR/tpl]", fontsize=12)
        ylabel = regret_specs[regret_name].get("ylabel", regret_name)
        ax.set_title(ylabel, fontsize=12)
        ax.set_ylabel(regret_name, fontsize=11)
        ax.tick_params(axis="y", labelsize=10)
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", linestyle="--", alpha=0.5)

        policy_handles = [
            plt.Line2D(
                [0], [0], marker="s", linestyle="None", markersize=8,
                markerfacecolor="0.75", markeredgecolor="0.25",
                label=d[:4],
            )
            for d in EPR_DESIGN_ORDER
        ]
        ax.legend(
            handles=policy_handles,
            title="Policy (L→R per fee)",
            loc="upper right",
            fontsize=9,
            title_fontsize=9,
        )

    for ax in axes[n_panels:]:
        ax.set_visible(False)

    fee_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=FEE_COLORS[f], edgecolor="0.25", alpha=0.88)
        for f in fee_levels
    ]
    fig.legend(
        fee_handles,
        [f"{f} EUR/tpl" for f in fee_levels],
        loc="upper center",
        ncol=len(fee_levels),
        fontsize=11,
        frameon=False,
        title="EPR fee (box colour)",
        title_fontsize=11,
    )
    fig.suptitle("Within-scenario regret by EPR fee and policy", fontsize=14, y=1.06)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    if debug:
        print(f"Saved {out_path}")
    if show:
        plt.show()
    return fig


def plot_replacement_cost_across_scenarios(
    results_df,
    out_path="results/replacement_cost_by_scenario.png",
    policy_col="EPR_design",
    fee_col="EPR_fee",
    cost_col="replacement_cost",
    show=False,
    debug=False,
):
    """Boxplots of subsidized Replacement levelized cost [EUR/t] by EPR fee across scenarios."""
    sub = results_df.loc[results_df[policy_col] == "Replacement"].copy()
    sub[cost_col] = pd.to_numeric(sub[cost_col], errors="coerce")
    sub = sub.dropna(subset=[cost_col])
    if sub.empty:
        if debug:
            print("plot_replacement_cost_across_scenarios: no subsidized Replacement runs")
        return None

    sub[fee_col] = sub[fee_col].astype(int)
    fee_levels = [f for f in EPR_FEE_LEVELS if f in sub[fee_col].unique()]
    if not fee_levels:
        fee_levels = sorted(sub[fee_col].unique())

    data = [sub.loc[sub[fee_col] == fee, cost_col].values for fee in fee_levels]
    colors = [FEE_COLORS.get(fee, plt.cm.magma(0.5)) for fee in fee_levels]

    fig, ax = plt.subplots(figsize=(8, 5))
    bp = ax.boxplot(
        data,
        tick_labels=[str(f) for f in fee_levels],
        widths=0.55,
        patch_artist=True,
    )
    _style_boxplot(bp, colors)
    ax.set_xlabel("EPR fee [EUR/tpl]", fontsize=12)
    ax.set_ylabel("Levelized cost (EUR/t methanol)", fontsize=12)
    ax.set_title(
        "Replacement cost (subsidized case only) across scenarios",
        fontsize=13,
    )
    ax.tick_params(labelsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    if debug:
        print(f"Saved {out_path} ({len(sub)} subsidized Replacement runs)")
    if show:
        plt.show()
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

    if "replacement_cost" in results_df.columns:
        plot_replacement_cost_across_scenarios(results_df, debug=debug)

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
    plot_all_ema_kpi_figures(results_df, show=False, debug=True)
    if "scenario_id" in results_df.columns:
        regret_df = calculate_regret(results_df, DEFAULT_REGRET_SPECS, debug=True)
        plot_regret_by_policy(regret_df, DEFAULT_REGRET_SPECS, show=True, debug=True)
    else:
        print("No scenario_id — skip regret plots (run controller.py first).")
