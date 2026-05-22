"""Sobol sensitivity for WACCUS-EPR Recovery (EMA + SALib)."""

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

EPR_DESIGN = "Recovery"
DESIGN_SLUG = "recovery"

plants_df = pd.read_csv("data/plants_clean.csv")
shipping_costs = pd.read_csv("data/shipping_costs.csv")
truck_costs = pd.read_csv("data/truck_costs.csv")
compression_costs = pd.read_csv("data/compression_costs.csv")
thermo_props = get_CoolProp()

SEK_to_EUR = 0.091
NOK_to_EUR = 0.089
shipping_costs = shipping_adjustment(shipping_costs, scaling=0.67, debug=False)
cost_columns = [col for col in shipping_costs.columns if col != "distance"]
shipping_costs[cost_columns] = shipping_costs[cost_columns] * SEK_to_EUR
truck_costs["EUR/ton"] = truck_costs["SEK/ton"] * SEK_to_EUR
truck_costs = truck_costs.drop(columns=["SEK/ton"])


def ema_WACCUS_EPR(**kwargs):
    results = WACCUS_EPR(**kwargs)
    return {f"KPI{i}": float(results[f"KPI{i}"]) for i in range(1, 18)}


model = Model("WACCUSEPRRecoverySobol", function=ema_WACCUS_EPR)

model.constants = [
    Constant("EPR_design", EPR_DESIGN),
    Constant("plants_df", plants_df),
    Constant("shipping_costs", shipping_costs),
    Constant("truck_costs", truck_costs),
    Constant("compression_costs", compression_costs),
    Constant("thermo_props", thermo_props),
    Constant("SEK_to_EUR", SEK_to_EUR),
    Constant("NOK_to_EUR", NOK_to_EUR),
    Constant("profit", 0.10),
    Constant("plot_results", False),
    Constant("ADJUST_CEPCI", False),
    Constant("CPI2015", 314.21),
    Constant("CPI2025", 417.96),
    Constant("CAPEXref_capture", 3.7e9 * 0.091 / 1000),
    Constant("CAPACITY_CAPTURE_REF_KT_PER_YR", 400.0),
    Constant("CAPEXref_synthesis", 1.8749),
    Constant("CAPEXref_loading", 63000000),
    Constant("CAPEXref_train", 8610000),
    Constant("capture_rate", 0.90),
    Constant("q_hex", 0.64),
    Constant("q_electrolyzer", 0.154),
    Constant("eta_is", 0.80),
    Constant("q_synthesis", 0.087),
    Constant("q_distill", 0.20),
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
    Constant("CAPEX_sorting_ref_msek", 650.0),
    Constant("capacity_sorting_ref_t_per_yr", 200_000.0),
    Constant("CAPEX_gasification_ref_meur", 749_729_639 * 1e-6),
    Constant("capacity_gasification_ref_t_per_yr", 237_000.0),
    Constant("opex_var_sorting_sek_per_t_waste", 200.0),
    Constant("opex_var_gasification_eur_per_mwh_fuel", 1.4),
    Constant("CEPCI_sorting_ref", 900),
    Constant("CEPCI_opex_sorting_ref", 816),
    Constant("CEPCI_gasification_ref", 600),
    Constant("CEPCI_opex_gasification_ref", 900),
    Constant("CEPCI_h2_ref", 600),
]

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
    RealParameter("p_capture", 0.08, 0.12),
    RealParameter("p_condition", 0.30, 0.45),
    RealParameter("COP", 2.5, 3.5),
    RealParameter("eta_electrolyzer", 0.675, 0.725),
    RealParameter("eta_synthesis", 0.76, 0.82),
    RealParameter("heat_optimism", 0.00, 1.00),
    RealParameter("k", 0.65, 0.69),
    RealParameter("CEPCI_scenario", 850, 950),
    RealParameter("dr", 0.06, 0.09),
    RealParameter("t", 20, 30),
    RealParameter("CAPEXref_HP", 800, 920),
    RealParameter("CAPEXref_H2", 2000, 3000),
    RealParameter("OPEXfix", 0.02, 0.04),
    RealParameter("camine", 40, 50),
    RealParameter("celc", 30, 100),
    RealParameter("cheat", 0.50, 0.95),
    RealParameter("pmethanol", 550, 850),
    RealParameter("storage_cost", 20, 80),
    RealParameter("carbon_change", -0.15, 0.15),
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
    sorted_stats = sobol_stats.sort_values(by="ST", ascending=True)
    n_params = len(sorted_stats)
    fig_h = max(6.0, 0.32 * n_params)
    fig, ax = plt.subplots(figsize=(8, fig_h))
    y_pos = np.arange(n_params)
    ax.barh(
        y_pos,
        sorted_stats["ST"],
        xerr=sorted_stats["ST_conf"],
        color=plt.cm.magma(0.65),
        ecolor="0.35",
        capsize=3,
        height=0.7,
    )
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_stats.index, fontsize=11)
    ax.set_xlabel("Total Sobol index (ST)", fontsize=12)
    ax.set_ylabel("Parameter", fontsize=12)
    ax.set_title(f"{EPR_DESIGN} — {KPI_LABELS.get(kpi, kpi)}", fontsize=13)
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

    experiments.to_csv(f"results/experiments_{DESIGN_SLUG}_sobol.csv", index=False)
    pd.DataFrame(outcomes).to_csv(f"results/outcomes_{DESIGN_SLUG}_sobol.csv", index=False)

    print(f"Completed {len(experiments)} {EPR_DESIGN} Sobol runs.")

    for kpi in SOBOL_KPIS:
        sobol_stats, s2, s2_conf, _ = analyze_sobol(results, kpi, debug=True)
        print(f"\n--- {kpi} ({KPI_LABELS.get(kpi, kpi)}) — top ST ---")
        print(sobol_stats.head(10).round(4))

        sobol_stats.to_csv(f"results/sobol_stats_{DESIGN_SLUG}_{kpi}.csv")
        s2.to_csv(f"results/sobol_s2_{DESIGN_SLUG}_{kpi}.csv")
        s2_conf.to_csv(f"results/sobol_s2_conf_{DESIGN_SLUG}_{kpi}.csv")

        plot_sobol_stats(
            sobol_stats,
            kpi,
            out_path=f"results/sobol_{DESIGN_SLUG}_{kpi}.png",
            debug=True,
        )

    plt.show()
    print(f"Done. {EPR_DESIGN} Sobol outputs saved under results/.")
