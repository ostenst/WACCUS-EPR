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

# Load and adjust data (mirrors model.py __main__)
plants_df = pd.read_csv('data/plants_clean.csv')
shipping_costs = pd.read_csv('data/shipping_costs.csv')
truck_costs = pd.read_csv('data/truck_costs.csv')
compression_costs = pd.read_csv('data/compression_costs.csv')
thermo_props = get_CoolProp()

SEK_to_EUR = 0.091
NOK_to_EUR = 0.089
shipping_costs = shipping_adjustment(shipping_costs, scaling=0.67, debug=False)
cost_columns = [col for col in shipping_costs.columns if col != 'distance']
shipping_costs[cost_columns] = shipping_costs[cost_columns] * SEK_to_EUR
truck_costs["EUR/ton"] = truck_costs["SEK/ton"] * SEK_to_EUR
truck_costs = truck_costs.drop(columns=["SEK/ton"])

# Set up EMA model
model = Model("WACCUSEPR", function=WACCUS_EPR)

# [C] Constants — fixed parameters and dataframes
model.constants = [
    Constant("EPR_design", "Mitigation"),  # change to "Mitigation" or "Recovery"
    Constant("plants_df", plants_df),
    Constant("shipping_costs", shipping_costs),
    Constant("truck_costs", truck_costs),
    Constant("compression_costs", compression_costs),
    Constant("thermo_props", thermo_props),
    Constant("SEK_to_EUR", SEK_to_EUR),
    Constant("profit", 0.10),
    Constant("plot_results", False),
    Constant("CPI2015", 314.21),
    Constant("CPI2025", 417.96),
    Constant("CAPEXref_capture", 3550 * NOK_to_EUR * 1000),  # [kEUR] @400 ktCO2/yr
    Constant("CAPEXref_synthesis", 1.8749),              # [MEUR]
    Constant("CAPEXref_loading", 63000000),              # [SEK] @150 ktCO2/yr
    Constant("CAPEXref_train", 8610000),                 # [EUR]
    Constant("capture_rate", 0.90),
    Constant("q_hex", 0.64),
    Constant("q_electrolyzer", 0.154),
    Constant("eta_is", 0.80),
    Constant("q_synthesis", 0.087),
    Constant("q_distill", 0.20),
    Constant("CEPCI_reference", 600),
    Constant("LHV_methanol", 19.8),
]

# [X] Uncertainties — sampled parameters
model.uncertainties = [
    # Plastic supply
    RealParameter("baseline_granulates", 1200000, 1300000),  # [t/a]
    RealParameter("inc_granulates", 0.00, 0.30),             # [-]
    RealParameter("shift_granulates", 0.00, 0.40),           # [-]
    RealParameter("price_granulates", 11000, 15000),         # [SEK/tpl]
    RealParameter("baseline_products", 700000, 884393),      # [t/a]
    RealParameter("inc_products", 0.00, 0.30),               # [-]
    RealParameter("shift_products", 0.00, 0.40),             # [-]
    RealParameter("stringent_products", 0.00, 1.00),         # [-]
    RealParameter("price_products", 36000, 56000),           # [SEK/tpl]

    # Process performance
    RealParameter("q_reb", 2.7, 3.7),                        # [MJ/kgCO2]
    RealParameter("p_capture", 0.08, 0.12),                  # [MWh/tCO2]
    RealParameter("p_condition", 0.30, 0.45),                # [MJ/kgCO2]
    RealParameter("COP", 2.5, 3.5),                          # [MWth/MWel]
    RealParameter("eta_electrolyzer", 0.675, 0.725),         # [MWH2/MWel]
    RealParameter("eta_synthesis", 0.76, 0.82),              # [MWmethanol/MWH2+steam]
    RealParameter("heat_optimism", 0.00, 1.00),              # [-]

    # Economics
    RealParameter("k", 0.65, 0.69),                          # [-]
    RealParameter("CEPCI_scenario", 850, 950),               # [-]
    RealParameter("dr", 0.06, 0.09),                         # [-]
    RealParameter("t", 20, 30),                              # [yr]
    RealParameter("CAPEXref_HP", 800, 920),                  # [kEUR/MWth]
    RealParameter("CAPEXref_H2", 450, 650),                  # [kEUR/MWe]
    RealParameter("OPEXfix", 0.02, 0.04),                    # [-]
    RealParameter("camine", 40, 50),                         # [SEK/tCO2]
    RealParameter("celc", 30, 100),                          # [EUR/MWh]
    RealParameter("cheat", 0.50, 0.95),                      # [% of elc]
    RealParameter("pmethanol", 550, 700),                    # [EUR/t]
    RealParameter("storage_cost", 15, 25),                   # [EUR/tCO2]
    RealParameter("carbon_change", -0.10, 0.10),             # [-]
    RealParameter("CRC", 50, 250),                           # [EUR/tCO2]
    RealParameter("ETS", 50, 250),                           # [EUR/tCO2]

    # Transport
    CategoricalParameter("shipping_case", ["optimist_0.5Mt", "pessimist_0.5Mt",
                                           "optimist_1Mt", "pessimist_1Mt"]),
    CategoricalParameter("storage", ["oygarden", "kalundborg"]),
]

# [L] Levers — policy choices
model.levers = [
    CategoricalParameter("EPR_products", [True, False]),
    RealParameter("EPR_fee", 50, 200),  # [EUR/tpl]
]

# Outcomes — KPIs returned by WACCUS_EPR
model.outcomes = [
    ScalarOutcome("KPI1"),   # plastic supply [ktpl/yr]
    ScalarOutcome("KPI2"),   # EPR fee [EUR/tpl]
    ScalarOutcome("KPI3"),   # available subsidies [kEUR/yr]
    ScalarOutcome("KPI4"),   # awarded plants [n]
    ScalarOutcome("KPI5"),   # fossil CO2 captured [ktCO2/yr]
    ScalarOutcome("KPI6"),   # biogenic CO2 captured [ktCO2/yr]
    ScalarOutcome("KPI7"),   # total CO2 captured [ktCO2/yr]
    ScalarOutcome("KPI8"),   # methanol output [GWh/yr]
    ScalarOutcome("KPI9"),   # power penalty [GWh/yr]
    ScalarOutcome("KPI10"),  # heat penalty [GWh/yr]
    ScalarOutcome("KPI11"),  # waste energy treated [GWh/yr]
    ScalarOutcome("KPI12"),  # original energy efficiency [-]
    ScalarOutcome("KPI13"),  # new energy efficiency [-]
]

if __name__ == "__main__":
    ema_logging.log_to_stderr(ema_logging.INFO)
    n_scenarios = 100
    n_policies = 100

    results = perform_experiments(
        model, n_scenarios, n_policies,
        uncertainty_sampling=Samplers.LHS,
        lever_sampling=Samplers.LHS,
    )
    experiments, outcomes = results

    outcomes_df = pd.DataFrame(outcomes)
    experiments.to_csv("results/experiments.csv", index=False)
    outcomes_df.to_csv("results/outcomes.csv", index=False)
    print(outcomes_df)

    # Boxplots of selected KPIs
    kpis = {
        "KPI3": "Available subsidies [kEUR/yr]",
        "KPI4": "Awarded plants [n]",
        "KPI5": "Fossil CO₂ captured [ktCO₂/yr]",
        "KPI6": "Biogenic CO₂ captured [ktCO₂/yr]",
    }
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, (key, label) in zip(axes, kpis.items()):
        ax.boxplot(outcomes_df[key], widths=0.5)
        ax.set_ylabel(label, fontsize=12)
        ax.set_xticks([])
        ax.tick_params(labelsize=11)
    fig.suptitle("KPI distributions", fontsize=14)
    fig.tight_layout()

    # Correlation between KPI3 (subsidies) and KPI7 (total CO2 captured)
    x_data = outcomes_df["KPI3"]
    y_data = outcomes_df["KPI7"]
    x_label = "Available subsidies [kEUR/yr]"
    y_label = "Total CO₂ captured [ktCO₂/yr]"

    fig2, axes2 = plt.subplots(1, 3, figsize=(16, 4.5))

    # (a) Scatter plot
    axes2[0].scatter(x_data, y_data, s=12, alpha=0.5, color=plt.cm.magma(0.35))
    axes2[0].set_xlabel(x_label, fontsize=12)
    axes2[0].set_ylabel(y_label, fontsize=12)
    axes2[0].set_title("Scatter", fontsize=13)

    # (b) 2D histogram / heatmap
    h = axes2[1].hist2d(x_data, y_data, bins=20, cmap="magma")
    fig2.colorbar(h[3], ax=axes2[1], label="Count")
    axes2[1].set_xlabel(x_label, fontsize=12)
    axes2[1].set_ylabel(y_label, fontsize=12)
    axes2[1].set_title("2D histogram", fontsize=13)

    # (c) KDE contour plot
    from scipy.stats import gaussian_kde
    xy = np.vstack([x_data, y_data])
    kde = gaussian_kde(xy)
    xi = np.linspace(x_data.min(), x_data.max(), 100)
    yi = np.linspace(y_data.min(), y_data.max(), 100)
    Xi, Yi = np.meshgrid(xi, yi)
    Zi = kde(np.vstack([Xi.ravel(), Yi.ravel()])).reshape(Xi.shape)
    axes2[2].contourf(Xi, Yi, Zi, levels=15, cmap="magma")
    axes2[2].set_xlabel(x_label, fontsize=12)
    axes2[2].set_ylabel(y_label, fontsize=12)
    axes2[2].set_title("KDE contour", fontsize=13)

    for ax in axes2:
        ax.tick_params(labelsize=11)
    fig2.suptitle(f"KPI3 vs KPI7  (r = {x_data.corr(y_data):.3f})", fontsize=14)
    fig2.tight_layout()
    plt.show()
