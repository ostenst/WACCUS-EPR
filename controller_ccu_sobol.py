import numpy as np
from model import *
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from ema_workbench import (
    Model,
    RealParameter,
    IntegerParameter,
    CategoricalParameter,
    ScalarOutcome,
    ArrayOutcome,
    Constant,
    Samplers,
    ema_logging,
    perform_experiments
)
from ema_workbench.em_framework import get_SALib_problem
from SALib.analyze import sobol
import matplotlib.cm as cm
import matplotlib.colors as mcolors

# Load data
plants_df = pd.read_csv('data/plants.csv')
plant_distances = pd.read_csv('data/plant_distances.csv')
shipping_df = pd.read_csv('data/shipping_costs.csv')
truck_df = pd.read_csv('data/truck_costs.csv')
compression_df = pd.read_csv('data/compression_costs.csv')
thermo_props = get_CoolProp()

# Adjust dataframes
SEK_to_EUR = 0.091
plants_df = plants_df.merge(plant_distances, on='Name', how='left')
shipping_df = shipping_adjustment(shipping_df, scaling=0.67, debug=False)
cost_columns = [col for col in shipping_df.columns if col != 'distance']
shipping_df[cost_columns] = shipping_df[cost_columns] * SEK_to_EUR
truck_df["EUR/ton"] = truck_df["SEK/ton"] * SEK_to_EUR
truck_df = truck_df.drop(columns=["SEK/ton"])

model = Model("WACCUS", function=WACCUS_EPR)

model.uncertainties = [
    RealParameter("mKN39", 884393*0.70, 884393),    # 884393 [t/a] plastic products mappable under KN39
    RealParameter("pKN39", 36000, 56000),           # 46000 [SEK/tpl]
    RealParameter("mgranulates", 1200000, 1300000), # 1258597 [t/a] granulates
    RealParameter("pgranulates", 11000, 15000),     # 13000 [SEK/tpl]
    RealParameter("mbag", 3*10**-6, 7*10**-6),      # 5*10**-6 [t/bag]
    RealParameter("pbag", 3, 6),                    # 4 [SEK/bag]
    RealParameter("cfraction", 0.70, 0.80),         # 0.75 [-] cfraction of carbon in plastic
    RealParameter("circulated", 0.00, 0.20),        # 0.10 [-] fraction of granulates circulated

    RealParameter("FLH", 7700, 8300),               # 8000 [h/yr]
    RealParameter("capture_rate", 0.85, 0.95),      # 0.90 [-]
    RealParameter("Ccontent", 0.275, 0.325),        # 0.298 [kgC/kgf]
    RealParameter("carbon_change", -0.10, 0.10),    # 0.10 [-] fraction of biogenic carbon
    RealParameter("LHVf", 10, 12),                  # 11 [MJ/kgf]
    RealParameter("LHVmethanol", 18, 21),           # 19.8 [MJ/kg]

    RealParameter("q_reb", 2.7, 3.7),               # 3.5 [MJ/kgCO2]
    RealParameter("p_capture", 0.08, 0.12),           # 0.1 [MWh/tCO2]
    RealParameter("p_condition", 0.30, 0.45),       # 0.37 [MJ/kgCO2]
    RealParameter("n_is", 0.80, 0.85),              # 0.80 [-]
    RealParameter("n_electrolyzer", 0.675, 0.725),  # 0.699 [MWH2/MWel]
    RealParameter("q_electrolyzer", 0.125, 0.175),  # 0.154 [MWth/MWel]
    RealParameter("q_synthesis", 0.075, 0.095),     # 0.087 [MWsteam/MWH2]
    RealParameter("n_synthesis", 0.76, 0.82),       # 0.78 [MWmethanol/MWH2+steam]
    RealParameter("q_distill", 0.18, 0.22),         # 0.20 [MWth/MWH2+steam]
    RealParameter("COP", 2.5, 3.5),                 # 3 [MWth/MWel]
    RealParameter("heat_optimism", 0.00, 1.00),     # 0.70 [-] % of waste heat recoverable

    RealParameter("CAPEXref_capture", 300000, 340000),  # 3550*0.09*1000=319500 [kEUR] @400 ktCO2/yr
    RealParameter("CAPEXref_HP", 0.80, 0.92),                           # 0.86[MEUR/MWth]
    RealParameter("CAPEXref_H2", 450, 650),                             # 550 [kEUR/MWe]
    RealParameter("CAPEXref_synthesis", 1.600, 2.000),                  # 1.8749 [MEUR]
    RealParameter("CAPEXref_loading", 58000000, 68000000),              # 63000000 [SEK] @150 ktCO2/yr
    RealParameter("CAPEXref_train", 8000000, 9000000),                  # 8610000 [EUR]
    RealParameter("k", 0.65, 0.69),                                     # 0.67[-] economy-of-scale factor
    RealParameter("CEPCI", 850, 950),                                   # 900 [-]

    RealParameter("OPEXfix", 0.02, 0.04),           # 0.02 [-] % of base CAPEX
    RealParameter("dr", 0.06, 0.09),                # 0.075 [-]
    RealParameter("t", 20, 30),                     # 25 [yr]
    RealParameter("camine", 40, 50),                # 44 [SEK/tCO2]
    RealParameter("celc", 30, 100),                 # 60 [EUR/MWh]
    RealParameter("cheat", 0.50, 0.95),             # 0.75 [% of elc]
    RealParameter("CRC", 50, 250),                  # 100 [EUR/tCO2] although assumed always >= ETS prices!
    RealParameter("ETS", 50, 250),                  # 80 [EUR/tCO2]
    RealParameter("pmethanol", 550, 700),           # 625 [EUR/t]

    RealParameter("ship_uncertain", -0.15, 0.15),    # 0.10 [-]
    RealParameter("truck_uncertain", -0.15, 0.15),   # 0.10 [-]
    RealParameter("train_uncertain", -0.15, 0.15),   # 0.10 [-]
    CategoricalParameter("stockholm", [1,2,3]),      # 2 [Mt/yr]
    CategoricalParameter("malmo", [0.5,1,2]),        # 1 [Mt/yr]
    CategoricalParameter("gothenburg", [0.5,1,2]),   # 1 [Mt/yr]
    CategoricalParameter("storage", ["oygarden", "kalundborg"]),

    CategoricalParameter("tax", [50, 80, 110, 140, 170, 200, 230]), # 100 [EUR/tCO2]
    RealParameter("recyclable", 0.05, 0.30),                        # 0.15 [-] fraction recyclable
]

model.levers = [
    # CategoricalParameter("tax", [50, 100, 150, 200, 250, 300]),     # 100 [EUR/tCO2]
    # RealParameter("recyclable", 0.05, 0.30),                        # 0.15 [-] fraction recyclable
]

model.constants = [
    Constant("question", "granulates"), # ["granulates", "products", "both"]
    Constant("CCUS", "CCU"),            # ["CCS", "CCU"]
    Constant("plants_df", plants_df),
    Constant("shipping_df", shipping_df),
    Constant("truck_df", truck_df),
    Constant("compression_df", compression_df),
    Constant("thermo_props", thermo_props),
    Constant("SEK_to_EUR", SEK_to_EUR),
    Constant("profit", 0.10), # 10% profit margin
    Constant("print_auction", False),
]

model.outcomes = [
    ScalarOutcome("mass_taxed", ScalarOutcome.MAXIMIZE),
    ScalarOutcome("fund", ScalarOutcome.MAXIMIZE),
    ScalarOutcome("remaining_fund", ScalarOutcome.MAXIMIZE),

    ScalarOutcome("granulates_inc", ScalarOutcome.MINIMIZE),
    ScalarOutcome("products_inc", ScalarOutcome.MINIMIZE),
    ScalarOutcome("bag_inc", ScalarOutcome.MINIMIZE),

    ScalarOutcome("total_FCCS", ScalarOutcome.MAXIMIZE),
    ScalarOutcome("total_BECCS", ScalarOutcome.MAXIMIZE),
    ScalarOutcome("total_CCS", ScalarOutcome.MAXIMIZE),
    ScalarOutcome("total_FCCU", ScalarOutcome.MAXIMIZE),
    ScalarOutcome("total_BCCU", ScalarOutcome.MAXIMIZE),
    ScalarOutcome("total_CCU", ScalarOutcome.MAXIMIZE),
    ScalarOutcome("total_Ppenalty", ScalarOutcome.MAXIMIZE),
    ScalarOutcome("total_Qpenalty", ScalarOutcome.MAXIMIZE),
    ScalarOutcome("total_Qmethanol", ScalarOutcome.MAXIMIZE),

    ScalarOutcome("potential_CCS", ScalarOutcome.MAXIMIZE),
    ScalarOutcome("potential_CCU", ScalarOutcome.MAXIMIZE),
    ScalarOutcome("n_plants", ScalarOutcome.MAXIMIZE),
]

ema_logging.log_to_stderr(ema_logging.INFO)
n_scenarios = 40
n_policies = 0

# If Sobol sampling:
results = perform_experiments(model, n_scenarios, n_policies, uncertainty_sampling = Samplers.SOBOL, lever_sampling = Samplers.SOBOL)
experiments, outcomes = results

def analyze(results, ooi):
    """analyze results using SALib sobol, returns a dataframe"""
    _, outcomes = results

    problem = get_SALib_problem(model.uncertainties)
    y = outcomes[ooi]
    sobol_indices = sobol.analyze(problem, y)
    sobol_stats = {key: sobol_indices[key] for key in ["ST", "ST_conf", "S1", "S1_conf"]}
    sobol_stats = pd.DataFrame(sobol_stats, index=problem["names"])
    sobol_stats.sort_values(by="ST", ascending=False)
    s2 = pd.DataFrame(sobol_indices["S2"], index=problem["names"], columns=problem["names"])
    s2_conf = pd.DataFrame(
        sobol_indices["S2_conf"], index=problem["names"], columns=problem["names"]
    )
    return sobol_stats, s2, s2_conf, problem

sobol_stats, s2, s2_conf, problem = analyze(results, "total_CCU") # The outcome of interest
print(sobol_stats)
print(s2)
print(s2_conf)
sobol_stats = pd.DataFrame(sobol_stats, index=problem["names"])
sobol_stats.to_csv("results/sobol_stats_ccu.csv")
sobol_stats_sorted = sobol_stats.sort_values(by="ST", ascending=False)  # Ascending for better readability

# Create horizontal bar plot
plt.figure(figsize=(8, 10))  # Adjust figure size for better layout
sns.barplot(
    y=sobol_stats_sorted.index,  # Parameters on y-axis
    x=sobol_stats_sorted["ST"],  # Sobol indices on x-axis
    xerr=sobol_stats_sorted["ST_conf"],  # Confidence intervals as error bars
    capsize=0.2,
    color="crimson"
)
plt.ylabel("Parameter")
plt.xlabel("Total Sobol Index (ST)")
plt.title("Total-Order Sobol Indices with Confidence Intervals")
plt.grid(axis="x", linestyle="--", alpha=0.7)
plt.show()