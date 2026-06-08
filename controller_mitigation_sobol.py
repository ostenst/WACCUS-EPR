"""Sobol sensitivity for WACCUS-EPR Mitigation (EMA + SALib)."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from model import WACCUS_EPR, get_CoolProp, shipping_adjustment
from ema_workbench import (
    Model,
    RealParameter,
    CategoricalParameter,
    ScalarOutcome,
    Constant,
    Samplers,
    ema_logging,
    perform_experiments,
)
from ema_workbench.em_framework import get_SALib_problem
from SALib.analyze import sobol

from plot import KPI_LABELS

# Load and adjust data (mirrors controller.py)
plants_df = pd.read_csv("data/plants_clean.csv")
shipping_costs = pd.read_csv("data/shipping_costs.csv")
compression_costs = pd.read_csv("data/compression_costs.csv")
thermo_props = get_CoolProp()

SEK_to_EUR = 0.091
NOK_to_EUR = 0.089
shipping_costs = shipping_adjustment(shipping_costs, scaling=0.67, debug=False)
cost_columns = [col for col in shipping_costs.columns if col != "distance"]
shipping_costs[cost_columns] = shipping_costs[cost_columns] * SEK_to_EUR

def ema_WACCUS_EPR(**kwargs):
    """EMA entry point: return scalar KPIs only."""
    results = WACCUS_EPR(**kwargs)
    return {f"KPI{i}": float(results[f"KPI{i}"]) for i in range(1, 18)}


model = Model("WACCUSEPRMitigationSobol", function=ema_WACCUS_EPR)

model.constants = [
    Constant("EPR_design", "Mitigation"),
    Constant("plants_df", plants_df),
    Constant("shipping_costs", shipping_costs),
    Constant("compression_costs", compression_costs),
    Constant("thermo_props", thermo_props),
    Constant("SEK_to_EUR", SEK_to_EUR),
    Constant("NOK_to_EUR", NOK_to_EUR),
    Constant("profit", 0.10),
    Constant("plot_results", False),
    Constant("CPI2015", 314.21),
    Constant("CPI2025", 417.96),
    Constant("capex_ref_capture_keur", 3.7e9 * 0.091 / 1000),  # [kEUR]
    Constant("capacity_ref_capture_kt_per_yr", 400.0),  # [ktCO2/yr]
    Constant("capex_ref_loading_sek", 63000000),  # [SEK]
    Constant("capacity_ref_loading_kt_per_yr", 150.0),  # [ktCO2/yr]
    Constant("capture_rate", 0.90),
    Constant("eta_is", 0.80),
    Constant("q_synthesis", 0.087),
    Constant("q_distill", 0.20),
    Constant("ADJUST_CEPCI", True),  # [-]
    Constant("CEPCI_target", 900),  # [-]
    Constant("CEPCI_reference", 600),
    Constant("CEPCI_capture_reference", 900),
    Constant("CEPCI_HP_reference", 816),
    Constant("LHV_methanol", 19.8),
    Constant("n_replacement_cases", 10),
    Constant("FLH_gasifier", 8000.0),
    Constant("RW_EVAP_MJ_PER_KG", 2.5),
    Constant("LHV_H2_MJ_PER_KMOL", 243.0),
    Constant("LHV_CO_MJ_PER_KMOL", 286.0),
    Constant("LHV_CH3OH", 21.1),
    Constant("recycle_ratio", 3.0),
    Constant("n_steam_assumed", 1.0),
    Constant("air_ratio_combustor", 1.2),
    Constant("q_wgs_mj_per_kmol", 43.0),
    Constant("gasified_carbon_fraction", 0.70),
    Constant("eta_boiler", 0.85),
    Constant("capex_ref_sorting_msek", 650.0),  # [MSEK]
    Constant("capacity_ref_sorting_t_waste_per_yr", 200_000.0),  # [t waste/yr]
    Constant("capex_ref_gasification_meur", 749_729_639 * 1e-6),  # [MEUR]
    Constant("capacity_ref_gasification_t_methanol_per_yr", 237_000.0),  # [t methanol/yr]
    Constant("CEPCI_sorting_ref", 900),
    Constant("CEPCI_opex_sorting_ref", 816),
    Constant("CEPCI_gasification_ref", 600),
    Constant("CEPCI_opex_gasification_ref", 900),
    Constant("CEPCI_h2_ref", 600),
]

# Former levers moved here; no policy levers (Sobol on uncertainties only)
model.uncertainties = [
    CategoricalParameter("EPR_products", [True, False]),
    CategoricalParameter("EPR_fee", [100, 200, 300, 400, 500]),
    RealParameter("baseline_granulates", 1_200_000, 1_300_000),
    RealParameter("inc_granulates", 0.00, 0.30),
    RealParameter("shift_granulates", 0.00, 0.40),
    RealParameter("price_granulates", 11_000, 15_000),
    RealParameter("baseline_products", 700_000, 884_393),
    RealParameter("inc_products", 0.00, 0.30),
    RealParameter("shift_products", 0.00, 0.40),
    RealParameter("stringent_products", 0.00, 1.00),
    RealParameter("price_products", 36_000, 56_000),
    RealParameter("q_reb", 2.7, 3.7),
    RealParameter("q_hex", 0.62, 0.66),  # [MWth/MWreb]
    RealParameter("q_electrolyzer", 0.150, 0.158),  # [MWth/MWel]
    RealParameter("p_capture", 0.08, 0.12),
    RealParameter("p_condition", 0.30, 0.45),
    RealParameter("COP", 2.5, 3.5),
    RealParameter("eta_electrolyzer", 0.675, 0.725),
    RealParameter("heat_optimism", 0.00, 1.00),
    RealParameter("k", 0.65, 0.69),
    RealParameter("dr", 0.06, 0.09),
    RealParameter("t", 20, 30),
    RealParameter("capex_ref_hp_keur_per_mwth", 800, 920),  # [kEUR/MWth]
    RealParameter("capex_ref_h2_keur_per_mwel", 2000, 3000),  # [kEUR/MWel]
    RealParameter("capex_ref_train_eur", 8_000_000, 9_000_000),  # [EUR]
    RealParameter("capex_ref_synthesis_meur", 1.60, 2.00),  # [MEUR]
    RealParameter("opex_var_sorting_sek_per_t_waste", 190.0, 210.0),  # [SEK/t waste]
    RealParameter("opex_var_gasification_eur_per_mwh_fuel", 1.35, 1.45),  # [EUR/MWh_fuel]
    RealParameter("opex_fix", 0.02, 0.04),  # [-]
    RealParameter("camine", 40, 50),
    RealParameter("celc", 30, 100),
    RealParameter("cheat", 0.50, 0.95),
    RealParameter("pmethanol", 550, 850),
    RealParameter("storage_cost", 20, 80),
    RealParameter("transport_cost_factor", 0.90, 1.10),  # [-]
    RealParameter("CRC", 50, 250),
    RealParameter("ETS", 50, 250),
    CategoricalParameter(
        "shipping_case",
        ["optimist_0.5Mt", "pessimist_0.5Mt", "optimist_1Mt", "pessimist_1Mt"],
    ),
    CategoricalParameter("storage", ["oygarden", "kalundborg"]),
]

model.levers = []

model.outcomes = [ScalarOutcome(f"KPI{i}") for i in range(1, 18)]

SOBOL_KPIS = ["KPI6", "KPI11", "KPI16"]


def analyze_sobol(results, ooi, debug=False):
    """SALib Sobol analysis for one outcome."""
    _, outcomes = results
    problem = get_SALib_problem(model.uncertainties)
    y = outcomes[ooi]
    if debug:
        print(f"\nSobol analyze {ooi}: n={len(y)}, D={problem['num_vars']}")
    sobol_indices = sobol.analyze(problem, y)
    sobol_stats = pd.DataFrame(
        {key: sobol_indices[key] for key in ["ST", "ST_conf", "S1", "S1_conf"]},
        index=problem["names"],
    )
    sobol_stats = sobol_stats.sort_values(by="ST", ascending=False)
    s2 = pd.DataFrame(
        sobol_indices["S2"], index=problem["names"], columns=problem["names"]
    )
    s2_conf = pd.DataFrame(
        sobol_indices["S2_conf"], index=problem["names"], columns=problem["names"]
    )
    return sobol_stats, s2, s2_conf, problem


def plot_sobol_stats(sobol_stats, kpi, out_path, debug=False):
    """Horizontal bar chart of total-order Sobol indices (ST)."""
    sorted_stats = sobol_stats.sort_values(by="ST", ascending=True)
    n_params = len(sorted_stats)
    fig_h = max(6.0, 0.32 * n_params)
    fig, ax = plt.subplots(figsize=(8, fig_h))
    y_pos = np.arange(n_params)
    color = plt.cm.magma(0.65)
    ax.barh(
        y_pos,
        sorted_stats["ST"],
        xerr=sorted_stats["ST_conf"],
        color=color,
        ecolor="0.35",
        capsize=3,
        height=0.7,
    )
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_stats.index, fontsize=11)
    ax.set_xlabel("Total Sobol index (ST)", fontsize=12)
    ax.set_ylabel("Parameter", fontsize=12)
    kpi_title = KPI_LABELS.get(kpi, kpi)
    ax.set_title(f"Mitigation — {kpi_title}", fontsize=13)
    ax.set_xlim(left=0)
    ax.grid(axis="x", linestyle="--", alpha=0.6)
    ax.tick_params(labelsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    if debug:
        print(f"Saved {out_path}")
    return fig


if __name__ == "__main__":
    ema_logging.log_to_stderr(ema_logging.INFO)
    # Power of 2; total runs ≈ n_scenarios × (2D + 2) for D uncertain parameters
    n_scenarios = 16
    n_policies = 0

    results = perform_experiments(
        model,
        n_scenarios,
        n_policies,
        uncertainty_sampling=Samplers.SOBOL,
        lever_sampling=Samplers.SOBOL,
    )
    experiments, outcomes = results

    experiments.to_csv("results/experiments_mitigation_sobol.csv", index=False)
    pd.DataFrame(outcomes).to_csv("results/outcomes_mitigation_sobol.csv", index=False)

    print(f"Completed {len(experiments)} Mitigation Sobol runs.")

    for kpi in SOBOL_KPIS:
        sobol_stats, s2, s2_conf, _ = analyze_sobol(results, kpi, debug=True)
        print(f"\n--- {kpi} ({KPI_LABELS.get(kpi, kpi)}) — top ST ---")
        print(sobol_stats.head(10).round(4))

        sobol_stats.to_csv(f"results/sobol_stats_mitigation_{kpi}.csv")
        s2.to_csv(f"results/sobol_s2_mitigation_{kpi}.csv")
        s2_conf.to_csv(f"results/sobol_s2_conf_mitigation_{kpi}.csv")

        plot_sobol_stats(
            sobol_stats,
            kpi,
            out_path=f"results/sobol_mitigation_{kpi}.png",
            debug=True,
        )

    plt.show()
    print("Done. Sobol CSVs and figures saved under results/.")
