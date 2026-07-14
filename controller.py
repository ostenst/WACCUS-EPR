"""
EMA open exploration: compare Mitigation / Recovery / Replacement within each scenario.

Per the EMA Workbench, a *scenario* is one draw from uncertainties (including
EPR_products and EPR_fee). A *policy* is one draw from levers (here: EPR_design).
With combine='factorial', each policy is evaluated on every scenario, so you can
compare KPIs across designs under the same parametrization.

See: https://emaworkbench.readthedocs.io/en/latest/indepth_tutorial/open-exploration.html
"""

import pandas as pd
from model import WACCUS_EPR, get_CoolProp, shipping_adjustment, plant_outcome_column_names
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
compression_costs = pd.read_csv("data/compression_costs.csv")
thermo_props = get_CoolProp()

SEK_to_EUR = 0.091
shipping_costs = shipping_adjustment(shipping_costs, scaling=0.67, debug=False)
cost_columns = [col for col in shipping_costs.columns if col != "distance"]
shipping_costs[cost_columns] = shipping_costs[cost_columns] * SEK_to_EUR
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
    for key in ("replacement_cost", "replacement_cost_financed"):
        val = results.get(key, float("nan"))
        out[key] = float(val) if pd.notna(val) else float("nan")
    for col in plant_outcome_column_names(plants_df):
        val = results.get(col, float("nan"))
        out[col] = float(val) if pd.notna(val) else float("nan")
    return out


def uncertainty_column_names(model):
    return [p.name for p in model.uncertainties]


def add_scenario_ids(df, uncertainty_cols):
    """Integer id for each unique uncertainty parametrization (policy held out)."""
    out = df.copy()
    out["scenario_id"] = out.groupby(uncertainty_cols, dropna=False).ngroup()
    return out


def calculate_regret(
    results_df,
    regret_specs,
    policies=None,
    scenario_col="scenario_id",
    policy_col="EPR_design",
    debug=False,
):
    """Within-scenario regret vs the best policy on each KPI."""
    if policies is None:
        policies = POLICY_ORDER
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


model = Model("WACCUSEPR", function=ema_WACCUS_EPR)

model.constants = [
    Constant("plot_results", False),
    Constant("n_replacement_cases", 10),
    Constant("plants_df", plants_df),
    Constant("shipping_costs", shipping_costs),
    Constant("compression_costs", compression_costs),
    Constant("thermo_props", thermo_props),

    Constant("SEK_to_EUR", SEK_to_EUR),
    Constant("profit", 0.10),
    Constant("CPI2015", 314.21),
    Constant("CPI2025", 417.96),

    Constant("eta_boiler", 0.85),
    Constant("capture_rate", 0.90),
    Constant("eta_is", 0.80),
    
    Constant("FLH_gasifier", 8000.0),
    Constant("LHV_H2_MJ_PER_KMOL", 243.0),
    Constant("LHV_CO_MJ_PER_KMOL", 286.0),
    Constant("LHV_CH3OH", 21.1),
    Constant("recycle_ratio", 3.0),

    # Reference CAPEX + capacity — scales as (capacity / capacity_ref)^k
    Constant("capex_ref_capture_keur", 3.7e9 * 0.091 / 1000),  # [kEUR]
    Constant("capacity_ref_capture_kt_per_yr", 400.0),  # [ktCO2/yr]
    Constant("capex_ref_loading_sek", 63000000),  # [SEK]
    Constant("capacity_ref_loading_kt_per_yr", 150.0),  # [ktCO2/yr]
    Constant("capex_ref_sorting_msek", 650.0),  # [MSEK]
    Constant("capacity_ref_sorting_t_waste_per_yr", 200_000.0),  # [t waste/yr]
    Constant("capex_ref_gasification_meur", 749_729_639 * 1e-6),  # [MEUR]
    Constant("capacity_ref_gasification_t_methanol_per_yr", 237_000.0),  # [t methanol/yr]

    Constant("ADJUST_CEPCI", True),  # [-] escalate overnight CAPEX to CEPCI_target
    Constant("CEPCI_target", 900),  # [-] cost evaluation year
    Constant("CEPCI_reference", 600),
    Constant("CEPCI_capture_reference", 900),
    Constant("CEPCI_HP_reference", 816),
    Constant("CEPCI_sorting_ref", 900),
    Constant("CEPCI_opex_sorting_ref", 816),
    Constant("CEPCI_gasification_ref", 600),
    Constant("CEPCI_opex_gasification_ref", 900),
    Constant("CEPCI_h2_ref", 900),
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

    RealParameter("COP", 2.0, 3.0),
    RealParameter("eta_electrolyzer", 0.63, 0.71),
    RealParameter("q_reb", 2.7, 3.7),
    RealParameter("q_hex", 0.60, 0.68),  # [MWth/MWreb]
    RealParameter("p_capture", 0.08, 0.12),
    RealParameter("p_condition", 0.35, 0.45),
    RealParameter("heat_optimism", 0.00, 1.00), # [0-100%] recovery from electrolyzers

    RealParameter("gasified_carbon_fraction", 0.65, 0.75), # [-] 0.70 central value, fraction of C to gasifier branch [Ecoplanta]
    RealParameter("frac_combustor_pl", 0.15, 0.35), # [-] 0.25 central value, share of combustor C that is plastic
    RealParameter("frac_energy", 0.55, 0.65), # [-] 0.60 central value, share of gasified fuel energy ending up in methanol [check Alberto Alamia]

    RealParameter("celc", 30, 70), # SEA Scenarier över Sveriges energisystem
    RealParameter("cheat", 0.50, 0.95),
    RealParameter("pmethanol", 550, 850),
    RealParameter("CRC", 50, 250),
    RealParameter("ETS", 50, 250),
    RealParameter("camine", 35, 55), # Ramboll SEK/tCO2
    CategoricalParameter(
        "shipping_case",
        ["optimist_0.5Mt", "pessimist_0.5Mt", "optimist_1Mt", "pessimist_1Mt"],
    ),
    CategoricalParameter("storage", ["oygarden", "kalundborg"]),
    RealParameter("storage_cost", 5, 58), # GlobalCCSInstitute report 
    RealParameter("transport_cost_factor", 0.80, 1.20),  # [-] CCS + replacement haul transport
    RealParameter("cost_correlation_truck", 0.10, 0.20),  # [EUR/(t·km)] truck transport correlation (Ouvrey)
    RealParameter("cost_correlation_pipeline", 0.01, 0.03),  # [EUR/(t·km)] pipeline transport correlation (Ouvrey)

    RealParameter("k", 0.62, 0.72),
    RealParameter("dr", 0.06, 0.09),
    RealParameter("t", 20, 30),
    RealParameter("capex_ref_hp_keur_per_mwth", 800, 920),  # [kEUR/MWth]
    RealParameter("capex_ref_h2_keur_per_mwel", 410, 690),  # [kEUR/MWel]
    RealParameter("capex_ref_train_eur", 8_000_000, 9_000_000),  # [EUR] fixed train @ 15 wagons × 60 t/wagon
    RealParameter("capex_ref_synthesis_meur", 1.60, 2.00),  # [MEUR] ∝ (m_methanol [t/d])^-0.315
    RealParameter("opex_var_sorting_sek_per_t_waste", 190.0, 210.0),  # [SEK/t waste]
    RealParameter("opex_var_gasification_eur_per_mwh_fuel", 1.35, 1.45),  # [EUR/MWh_fuel]
    RealParameter("opex_fix", 0.03, 0.05),  # [-] fixed OPEX as fraction of overnight CAPEX
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
    ScalarOutcome("KPI10"),  # Mitigation/Recovery: new equipment capacity [MWe]; Replacement: hub + site electrical demand [MWel]
    ScalarOutcome("KPI11"),  # [TWh/yr] new power
    ScalarOutcome("KPI12"),  # [Mtpl/yr] targeted plastic supply
    ScalarOutcome("KPI13"),  # [EUR/tpl] EPR fee
    ScalarOutcome("KPI14"),  # [MEUR/yr] available subsidies
    ScalarOutcome("KPI15"),  # [MEUR/yr] remaining subsidies
    ScalarOutcome("KPI16"),  # [%] granulate price increase
    ScalarOutcome("KPI17"),  # [%] products price increase
    ScalarOutcome("replacement_cost"),  # [EUR/t MeOH] subsidized Replacement only (cost gap > 0); NaN if profitable vs pmethanol
    ScalarOutcome("replacement_cost_financed"),  # [EUR/t MeOH] financed Replacement hub (KPI1 > 0); NaN if no affordable case
] + [ScalarOutcome(col) for col in plant_outcome_column_names(plants_df)]

if __name__ == "__main__":
    ema_logging.log_to_stderr(ema_logging.INFO)
    n_scenarios = 10000
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
    repl_sub = results_df.loc[
        results_df["EPR_design"] == "Replacement", "replacement_cost"
    ].dropna()
    repl_fin = results_df.loc[
        results_df["EPR_design"] == "Replacement", "replacement_cost_financed"
    ].dropna()
    if len(repl_sub):
        print(f"\nSubsidized Replacement cost: n={len(repl_sub)}, mean={repl_sub.mean():.1f} EUR/t methanol")
    if len(repl_fin):
        print(f"Financed Replacement cost: n={len(repl_fin)}, mean={repl_fin.mean():.1f} EUR/t methanol")
    print("\nSaved results/results.csv, regret_by_policy.csv")
    print("Run plot.py for figures.")
