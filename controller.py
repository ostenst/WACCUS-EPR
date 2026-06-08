"""
EMA open exploration: compare Mitigation / Recovery / Replacement within each scenario.

Per the EMA Workbench, a *scenario* is one draw from uncertainties (including
EPR_products and EPR_fee). A *policy* is one draw from levers (here: EPR_design).
With combine='factorial', each policy is evaluated on every scenario, so you can
compare KPIs across designs under the same parametrization.

See: https://emaworkbench.readthedocs.io/en/latest/indepth_tutorial/open-exploration.html
"""

import pandas as pd
from model import WACCUS_EPR, get_CoolProp, shipping_adjustment
from plot import calculate_regret, plot_regret_by_policy, plot_replacement_cost_across_scenarios
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

# Load and adjust data (mirrors model.py __main__)
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

POLICY_ORDER = ["Mitigation", "Recovery", "Replacement"]

# Within-scenario regret: add entries as needed (sense "min" = lower KPI is better)
REGRET_SPECS = {
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
    # Example: higher is better → sense "max"
    # "regret_plants": {"kpi": "KPI1", "sense": "max", "ylabel": "Regret — plants financed [n]"},
}


def ema_WACCUS_EPR(**kwargs):
    """EMA entry point: return scalar outcomes only (no bids / replacement_cases)."""
    results = WACCUS_EPR(**kwargs)
    out = {f"KPI{i}": float(results[f"KPI{i}"]) for i in range(1, 18)}
    rc = results.get("replacement_cost", float("nan"))
    out["replacement_cost"] = float(rc) if pd.notna(rc) else float("nan")
    return out


def uncertainty_column_names(model):
    return [p.name for p in model.uncertainties]


def add_scenario_ids(df, uncertainty_cols):
    """Integer id for each unique uncertainty parametrization (policy held out)."""
    out = df.copy()
    out["scenario_id"] = out.groupby(uncertainty_cols, dropna=False).ngroup()
    return out


model = Model("WACCUSEPR", function=ema_WACCUS_EPR)

model.constants = [
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
    Constant("frac_combustor_bio", 0.75),
    Constant("frac_energy", 0.70),
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

# [X] Uncertainties — deep uncertainty + EPR context (fixed within scenario)
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

# [L] Levers — EPR design only (three policies compared within each scenario)
model.levers = [
    CategoricalParameter("EPR_design", POLICY_ORDER),
]

# Outcomes — KPIs returned by WACCUS_EPR (see model.py)
model.outcomes = [
    ScalarOutcome("KPI1"),   # [n] CHP plants financed / replaced
    ScalarOutcome("KPI2"),   # [ktCO2f/yr] fossil CO2 stored (Mitigation)
    ScalarOutcome("KPI3"),   # [ktCO2b/yr] biogenic CO2 stored (Mitigation)
    ScalarOutcome("KPI4"),   # [ktMeOHf/yr] fossil methanol
    ScalarOutcome("KPI5"),   # [ktMeOHb/yr] biogenic methanol
    ScalarOutcome("KPI6"),   # [ktC/yr] carbon treated
    ScalarOutcome("KPI7"),   # [ktCO2f/yr] residual fossil CO2
    ScalarOutcome("KPI8"),   # [ktCO2f/yr] hub combustor fossil CO2 (Replacement)
    ScalarOutcome("KPI9"),   # [ktCO2b/yr] hub combustor biogenic CO2 (Replacement)
    ScalarOutcome("KPI10"),  # [MWe] new power capacity
    ScalarOutcome("KPI11"),  # [TWh/yr] new power
    ScalarOutcome("KPI12"),  # [Mtpl/yr] targeted plastic supply
    ScalarOutcome("KPI13"),  # [EUR/tpl] EPR fee
    ScalarOutcome("KPI14"),  # [MEUR/yr] available subsidies
    ScalarOutcome("KPI15"),  # [MEUR/yr] remaining subsidies
    ScalarOutcome("KPI16"),  # [%] granulate price increase
    ScalarOutcome("KPI17"),  # [%] products price increase
    ScalarOutcome("replacement_cost"),  # [EUR/t methanol] subsidized Replacement only
]

if __name__ == "__main__":
    ema_logging.log_to_stderr(ema_logging.INFO)
    n_scenarios = 100
    n_policies = len(POLICY_ORDER)  # 3: Mitigation, Recovery, Replacement

    results = perform_experiments(
        model,
        n_scenarios,
        n_policies,
        uncertainty_sampling=Samplers.LHS,
        lever_sampling=Samplers.FF,  # all three EPR_design policies per scenario
        combine="factorial",
    )
    experiments, outcomes = results

    outcomes_df = pd.DataFrame(outcomes)
    results_df = pd.concat([experiments.reset_index(drop=True), outcomes_df], axis=1)

    unc_cols = uncertainty_column_names(model)
    results_df = add_scenario_ids(results_df, unc_cols)

    experiments.to_csv("results/experiments.csv", index=False)
    outcomes_df.to_csv("results/outcomes.csv", index=False)
    results_df.to_csv("results/results.csv", index=False)

    regret_df = calculate_regret(
        results_df, REGRET_SPECS, policies=POLICY_ORDER, debug=True
    )
    regret_df.to_csv("results/regret_by_policy.csv", index=False)

    plot_regret_by_policy(
        regret_df,
        REGRET_SPECS,
        out_path="results/regret_by_policy.png",
        debug=True,
    )
    plot_replacement_cost_across_scenarios(results_df, debug=True)

    n_exp = len(experiments)
    print(f"Completed {n_exp} runs ({n_scenarios} scenarios × {n_policies} policies).")
    print("\nOutcome means (all runs):")
    print(outcomes_df.mean().round(2).to_string())
    print("\nMean within-scenario regret by policy:")
    print(
        regret_df.groupby("EPR_design")[list(REGRET_SPECS.keys())]
        .mean()
        .round(4)
        .to_string()
    )
    repl = results_df.loc[
        results_df["EPR_design"] == "Replacement", "replacement_cost"
    ].dropna()
    if len(repl):
        print(f"\nSubsidized Replacement cost: n={len(repl)}, mean={repl.mean():.1f} EUR/t methanol")
    print("\nSaved results/results.csv, regret_by_policy.csv, replacement_cost_by_scenario.png")
    print("Run plot.py for remaining KPI boxplots.")
