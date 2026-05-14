import pandas as pd
import matplotlib.pyplot as plt
from model import get_CoolProp, compression_energy

# Read pre-calculated plant characterization
plants_clean = pd.read_csv("data/plants_clean.csv")
plants_clean = plants_clean.sort_values(by="m_tot", ascending=False).reset_index(drop=True)

rw = 2.5 # [MJ/kg] water evaporation
LHV_H2 = 243 # [MJ/kmolH2]
LHV_CO = 286 # [MJ/kmolCO]
FLH_gasifier = 8000 # [h/yr]

# ------------------------------------- GASIFICATION SINGLE -------------------------------------
# Try gasifying the largest plant, define all its data:
plant = plants_clean.iloc[0]
nC_pl = plant["nC_pl"] # [kmolC_pl/yr]
nH_pl = plant["nH_pl"] # [kmolH_pl/yr]
nO_pl = plant["nO_pl"] # [kmolO_pl/yr]
nC_bio = plant["nC_bio"] # [kmolC_bio/yr]
nH_bio = plant["nH_bio"] # [kmolH_bio/yr]
nO_bio = plant["nO_bio"] # [kmolO_bio/yr]
m_pl = plant["m_pl"] # [kg/yr]
m_bio = plant["m_bio"] # [kg/yr]
m_h2o = plant["m_h2o"] # [kg/yr]
m_ash = plant["m_ash"] # [kg/yr]
m_tot = plant["m_tot"] # [kg/yr]
LHV_pl = plant["LHV_pl"] # [MJ/kg]
LHV_bio = plant["LHV_bio"] # [MJ/kg]
LHV_tot = plant["LHV_tot"] # [MJ/kg]
print(plant)

# (1) Characterize the fuel in terms of CHxOy (wet basis)
E_input = LHV_tot * m_tot # [MJ/yr] available fuel LHV
x = (nH_pl + nH_bio) / (nC_pl + nC_bio) # [kmolH/kmolC] from dry mixed basis
y = (nO_pl + nO_bio) / (nC_pl + nC_bio) # [kmolO/kmolC]
print("C", x, "H", y)

# (2) Dry the fuel using steam [R2] CHxOy -> CHz1 + yH2O
n_steam_dry = y # [kmolH2O/kmolC]
z1 = x - 2 * n_steam_dry # [kmolH/kmolC]
print("CH", z1, "H2O", n_steam_dry)

# (3) Set up solver in [R3] CHz1 + 1*H2O + x0*O2 -> x1*CO + z2*H2 + x2*CO2
# Under a first guess, all hydrogen in z1 and 1*H2O ends up in H2.
# Also, 70% of the mixed waste (energy and d.a. mass) goes to unshifted syngas.
n_steam_assumed = 1 # [kmolH2O/kmolC]
z2 = z1 / 2 + n_steam_assumed # [kmolH2/kmolC]
n_H2 = z2 # [kmolH2/kmolC]
print("H2", n_H2)

E_input = LHV_pl * m_pl + LHV_bio * m_bio # [MJ/yr] available LHV
E_H2 = n_H2 * LHV_H2 * (nC_pl + nC_bio) * 0.70 # [MJ/yr]
E_CO = E_input * 0.70 - E_H2 # [MJ/yr]
print(E_H2 / (E_input * 0.70))

n_CO = E_CO / LHV_CO # [kmolCO/yr]
n_CO = n_CO / ((nC_pl + nC_bio) * 0.70) # [kmolCO/kmolC]
n_CO2 = 1 - n_CO # [kmolCO2/kmolC]
n_O2 = n_CO2 + (n_CO - 1) / 2 # [kmolO2/kmolC]
print(E_H2, n_CO, n_CO2, n_O2, n_H2 / n_CO)

# (4) Adjust the initial guess iteratively until H2:CO is close to 2:1
rWGS = 0 # [kmolrxn/kmolC] extent of [R4] H2 + CO2 -> CO + H2O
n_steam_wgs = 0 # [kmolH2O/kmolC]
E_product = LHV_H2 * n_H2 + LHV_CO * n_CO # [MJ/kmolC]
while n_H2 / n_CO > 2:
    rWGS += 0.001 # [kmolrxn/kmolC]
    n_H2 = n_H2 - rWGS # [kmolH2/kmolC]
    n_CO = n_CO + rWGS # [kmolCO/kmolC]
    n_CO2 = n_CO2 - rWGS # [kmolCO2/kmolC]
    n_steam_wgs = rWGS
    E_product = LHV_H2 * n_H2 + LHV_CO * n_CO # [MJ/kmolC]
    print("n_H2", n_H2, "n_CO", n_CO, "n_CO2", n_CO2, "n_O2", n_O2, "rWGS", rWGS, "n_H2/n_CO", n_H2 / n_CO)
    print("     E_product", E_product, "E_eff", E_product * ((nC_pl + nC_bio) * 0.70) / E_input)

# (5) Check steam and heat balances
q_evap = rw * 18 # [MJ/kmol] evaporation of water
q_wgs = 43 # [MJ/kmol] water gas shift reaction
n_steam_missing = n_steam_assumed - n_steam_dry - n_steam_wgs # [kmolH2O/kmolC]
print("n_steam_missing", n_steam_missing, "n_steam_assumed", n_steam_assumed, "n_steam_dry", n_steam_dry, "n_steam_wgs", n_steam_wgs)
Q_gasify = n_steam_missing * q_evap # [MJ/kmolC] additional steam needed for [R3]
Q_dry = n_steam_dry * q_evap # [MJ/kmolC] steam needed (and produced) in [R2]
Q_wgs = n_steam_wgs * q_wgs # [MJ/kmolC] steam produced in [R4]
print("Q_gasify", Q_gasify, "Q_dry", Q_dry, "Q_wgs", Q_wgs)
Q_steam_demand = (Q_gasify + Q_dry + Q_wgs) * (nC_pl + nC_bio) * 0.70 # [MJ/yr]
print("Q_steam_demand", Q_steam_demand)
print("Q_steam_demand/E_input", Q_steam_demand / E_input)
print("E eff", E_product * ((nC_pl + nC_bio) * 0.70) / E_input)

# (6) Calculate combusted remains
m_combustor = (m_pl + m_bio)*0.30 + m_ash + m_h2o # [kg/yr]
mCO2_fossil = nC_pl*0.30*44 # [kgCO2/yr] residual fossil CO2
mCO2_bio = nC_bio*0.30*44 # [kgCO2/yr] residual biogenic CO2
Q_residual_moisture = rw * m_h2o # [MJ/yr] residual moisture must be evaporated
Q_steam_available = E_input*0.30 - Q_residual_moisture # [MJ/yr]
print("Q_residual_moisture", Q_residual_moisture)
print("Q_steam_available", Q_steam_available)
print("Q_steam_demand/Q_steam_available", Q_steam_demand / Q_steam_available)

# Calculate O2
air_ratio = 1.2
nO2_combustor = (nC_pl + nC_bio)*0.30 # [kmolO2/yr] O2 needed for stoichiometric combustion
nO2_combustor = nO2_combustor / FLH_gasifier / 3600 * air_ratio # [kmolO2/s]
print("nO2_s demand for combustor", nO2_combustor)

# -------------- COMPRESSION AND METHANOL SYNTHESIS --------------
# q_synthesis = 0.087    # [MWsteam/MWH2]
# eta_synthesis = 0.78    # [MWmethanol/MWH2+steam]
# LHV_methanol = 19.8 # [MJ/kg]
eta_electrolyzer = 0.699 # [MWH2/MWel]

thermo_props = get_CoolProp()
nC_gasified = (nC_pl + nC_bio) * 0.70 # [kmolC/yr]
n_H2O_syngas = n_steam_wgs # [kmolH2O/kmolC] first estimate from [R4]

# Build syngas mix and the required H2 gas
n_species = {
    "H2": n_H2 * nC_gasified,    # [kmol/yr]
    "CO": n_CO * nC_gasified,    # [kmol/yr]
    "CO2": n_CO2 * nC_gasified,  # [kmol/yr]
    "H2O": n_H2O_syngas * nC_gasified,  # [kmol/yr]
}
n_tot = sum(n_species.values()) # [kmol/yr]
y_mix = {gas: n_flow / n_tot for gas, n_flow in n_species.items()} # [-]
n_mix_s = n_tot / FLH_gasifier / 3600 # [kmol/s]

# Produce H2 from AEL electrolyzer
nCO2_s = n_CO2 * nC_gasified / FLH_gasifier / 3600  # [kmolCO2/s]
nH2_s = nCO2_s * 3  # [kmolH2/s] synthesis stoichiometry for CO2 + 3H2 -> CH3OH + H2O
QH2 = nH2_s * LHV_H2                    # [MWth] 
PH2 = QH2/eta_electrolyzer              # [MWel] power demand

print(" ")
print("QH2", QH2, "MWth")
print("PH2", PH2, "MWel")
print("E_input", E_input/3600/FLH_gasifier, "MWth")
print("PH2/E_input", PH2 / (E_input/3600/FLH_gasifier), "MWel/MWth")
print(" ")

# Compress syngas mix and H2 gas
Wcomp_mix, Qcool_mix, P_mix, T_mix = compression_energy(
    n_mix_s, T1=40+273.15, P1=1.0, thermo_props=thermo_props,
    gas=y_mix, n_stages=3, pr=3.8, Tdiff=30, n_is=0.8, debug=True
)
print("Syngas compression totals: Wcomp", sum(Wcomp_mix), "Qcool", sum(Qcool_mix))

Wcomp_H2, Qcool_H2, P_H2, T_H2 = compression_energy(
    nH2_s, T1=75+273.15, P1=20, thermo_props=thermo_props,
    gas="H2", n_stages=2, pr=1.7, Tdiff=60, n_is=0.8, debug=True
)
recycle_ratio = 3
Wcomp_recycle = (sum(Wcomp_H2) + sum(Wcomp_mix)) * recycle_ratio
print("H2 compression totals: Wcomp", sum(Wcomp_H2), "Qcool", sum(Qcool_H2))
print(" ")

nCH3OH = nC_gasified # [kmolCH3OH/yr] produced methanol
mCH3OH = nCH3OH * 32 # [kg/yr]
print("mCH3OH", mCH3OH/FLH_gasifier/3600, "kg/s")
LHV_CH3OH = 21.1 # [MJ/kg]
QCH3OH = mCH3OH * LHV_CH3OH /3600 # [MWh/yr]
QCH3OH_s = QCH3OH / FLH_gasifier # [MWth]
print("QCH3OH", QCH3OH, "MWh/yr")
print("QCH3OH_s", QCH3OH_s, "MWhth")
print("Input fuel energy:", E_input/3600/FLH_gasifier, "MWth")
print("INPUT ENERGY:", QH2+Q_steam_demand/3600/FLH_gasifier)
print(Q_steam_demand/3600/FLH_gasifier)
print(" ")

# Time to check overall energy balances in MW
input_fuel = E_input/3600/FLH_gasifier
q_syngas = E_product * ((nC_pl + nC_bio) * 0.70)/3600/FLH_gasifier
power_input = PH2 + sum(Wcomp_mix) + sum(Wcomp_H2) + Wcomp_recycle
hydrogen_produced = QH2
print("input_fuel", input_fuel, "MWth")
print("q_syngas", q_syngas, "MWth")
print("power_input", power_input, "MWel")
print("power_hydrogen", PH2, "MWel")
print("hydrogen_produced", hydrogen_produced, "MWth")

print(" ")
q_synthesis_input = q_syngas + hydrogen_produced + sum(Wcomp_mix) + sum(Wcomp_H2) + Wcomp_recycle + Q_steam_demand/3600/FLH_gasifier
q_synthesis_output = QCH3OH_s
total_input = input_fuel + power_input
print("q_synthesis_input", q_synthesis_input, "MWth")
print("q_synthesis_output", q_synthesis_output, "MWth")
print("q_synthesis_output/q_synthesis_input", q_synthesis_output / q_synthesis_input)
print("eff_total", q_synthesis_output / total_input)
print(" >>> The total efficiency is comparable to Beiron (2026) at around 47% (or with DH: 63%)")
print(" >>> But we require relatively low power input, owing to the more optimistic syn gas composition (and thus reduced H2 demand!)")

steam_mw = Q_steam_demand / 3600 / FLH_gasifier  # [MWth] steam-related thermal rate (same basis as q_syngas)
ratio_synthesis = q_synthesis_output / q_synthesis_input
eff_total_plot = q_synthesis_output / total_input

# -------------- AUXILIARY OXYGEN AND HEAT PUMP/ELECTROLYZER DEMANDS ----------
print(" ")
nO2_demand = n_O2 * nC_gasified / FLH_gasifier / 3600  # [kmolO2/s]
nO2_electrolyzer = nH2_s / 2 # [kmolO2/s] electrolysis stoichiometry for 2H2O -> O2 + 2H2
print("nO2_demand", nO2_demand)
print("nO2_combustor", nO2_combustor)
print("nO2_electrolyzer", nO2_electrolyzer)
print("nO2_demand/nO2_electrolyzer", nO2_demand / nO2_electrolyzer)
print("(nO2_combustor+nO2_demand)/nO2_electrolyzer", (nO2_combustor + nO2_demand) / nO2_electrolyzer)
print(" or inversely: nO2_electrolyzer/(nO2_combustor+nO2_demand)", nO2_electrolyzer / (nO2_combustor + nO2_demand))
print("=> Seems like the available oxygen is sufficient for GASIFICATION demands, but this must be verified for all cumulative plant combinations!")
print("=> It is however not if we add OXY-COMBUSTION demands!")

# COSTS
# CAPEX elements: [Sorting],[Dry,Comb,Gasif,Clean,Upgrade(WGS),Comp(Syngas),Synth+Destill],[Electrolyzer],[Comp(H2)]
# OPEX elements: [fix %CAPEX], [variable energy], [variable non-energy]
fixate_CAPEX = 0.05 # [-] % of CAPEX
k = 0.67 # scaling factor
SEK_to_EUR = 0.091 # [EUR/SEK]
dr = 0.075 # [-]
lifetime = 25 # [yr]
annual_methanol = mCH3OH/1000 # [t/a]

def levelize_MEUR_methanol(CAPEX, annual_methanol, dr, lifetime):
    CRF = dr * (1 + dr)**lifetime / ((1 + dr)**lifetime - 1)
    CAPEX_annual = CAPEX * CRF * 10**6 # [EUR/yr]
    CAPEX_lev = CAPEX_annual / annual_methanol # [EUR/t]
    return CAPEX_lev

CAPEX_sorting_ref = 350 * SEK_to_EUR # [MEUR] @140ktwaste p.a. Brista
capacity_sorting_ref = 140000 # [t/a]
capacity_sorting_target = m_tot/1000 # [t/a]
CAPEX_sorting = CAPEX_sorting_ref * (capacity_sorting_target/capacity_sorting_ref)**k  # [MEUR]
CAPEX_sorting_lev = levelize_MEUR_methanol(CAPEX_sorting, annual_methanol, dr, lifetime) # [EUR/t]

OPEX_fix_sorting = CAPEX_sorting * fixate_CAPEX # [MEUR p.a.]
OPEX_var_sorting = 200 * SEK_to_EUR # [EUR/twaste] Brista
OPEX_var_sorting = OPEX_var_sorting * capacity_sorting_target * 10**-6 # [MEUR p.a.]
OPEX_sorting_lev = (OPEX_fix_sorting + OPEX_var_sorting) / (annual_methanol * 10**-6) # [EUR/t METHANOL]
print("\nOPEX_fix_sorting", OPEX_fix_sorting, "MEUR p.a.")
print("OPEX_var_sorting", OPEX_var_sorting, "MEUR p.a.")
print("CAPEX_sorting", CAPEX_sorting, "MEUR")
print("CAPEX_sorting_lev", CAPEX_sorting_lev, "EUR/t METHANOL")
print("OPEX_sorting_lev", OPEX_sorting_lev, "EUR/t METHANOL")

celc = 50 # [EUR/MWh] 
CAPEX_gasification_ref = 749729639 * 10**-6 # [MEUR] @237ktMETHANOL p.a. ECOPLANTA, and @70% carbon recovery
capacity_gasification_ref = 237000 # [t/a]
capacity_gasification_target = mCH3OH / 1000 # [t/a]
CAPEX_gasification = CAPEX_gasification_ref * (capacity_gasification_target/capacity_gasification_ref)**k # [MEUR]
CAPEX_gasification_lev = levelize_MEUR_methanol(CAPEX_gasification, annual_methanol, dr, lifetime) # [EUR/t METHANOL]

OPEX_fix_gasification = CAPEX_gasification * fixate_CAPEX # [MEUR p.a.]
OPEX_var_gasification = 1.4 # [EUR/MWhfuel] [Beiron, 2026]
OPEX_var_gasification = OPEX_var_gasification * (LHV_tot * m_tot)/3600 * 10**-6 # [MEUR p.a.]
OPEX_energy_gasification = (sum(Wcomp_mix) + sum(Wcomp_H2) + Wcomp_recycle) * FLH_gasifier * celc * 10**-6 # [MEUR p.a.] 
OPEX_gasification_lev = (OPEX_fix_gasification + OPEX_var_gasification + OPEX_energy_gasification) / (annual_methanol * 10**-6) # [EUR/t METHANOL]
print("\nOPEX_fix_gasification", OPEX_fix_gasification, "MEUR p.a.")
print("OPEX_var_gasification", OPEX_var_gasification, "MEUR p.a.")
print("OPEX_energy_gasification", OPEX_energy_gasification, "MEUR p.a.")
print("CAPEX_gasification", CAPEX_gasification, "MEUR")
print("CAPEX_gasification_lev", CAPEX_gasification_lev, "EUR/t METHANOL")
print("OPEX_gasification_lev", OPEX_gasification_lev, "EUR/t METHANOL")

CAPEXref_H2 = 550 # [kEUR/MWe] [Danish Agency Excel Renewable Fuels AEC100MW]
CAPEX_H2 = CAPEXref_H2 * PH2 * 10**-3 # [MEUR]
CAPEX_H2_lev = levelize_MEUR_methanol(CAPEX_H2, annual_methanol, dr, lifetime) # [EUR/t METHANOL]
OPEX_fix_H2 = CAPEX_H2 * fixate_CAPEX # [MEUR p.a.]
OPEX_var_H2 = PH2 * FLH_gasifier * celc * 10**-6 # [MEUR p.a.] 
OPEX_H2_lev = (OPEX_fix_H2 + OPEX_var_H2) / (annual_methanol * 10**-6) # [EUR/t METHANOL]
print("\nOPEX_fix_H2", OPEX_fix_H2, "MEUR p.a.")
print("OPEX_var_H2", OPEX_var_H2, "MEUR p.a.")
print("CAPEX_H2", CAPEX_H2, "MEUR")
print("CAPEX_H2_lev", CAPEX_H2_lev, "EUR/t METHANOL")
print("OPEX_H2_lev", OPEX_H2_lev, "EUR/t METHANOL")

cost_lev = CAPEX_sorting_lev + CAPEX_gasification_lev + CAPEX_H2_lev + OPEX_sorting_lev + OPEX_gasification_lev + OPEX_H2_lev # [EUR/t METHANOL]
print("===> cost_lev", cost_lev, "EUR/t METHANOL")


def plot_levelized_cost_contributions(
    cost_parts,
    part_labels,
    total_lev,
    out_path="results/levelized_cost_breakdown.png",
    debug=False,
):
    """Bar chart of EUR/t methanol contributions."""
    import os

    if debug:
        print("plot_levelized_cost_contributions inputs:", cost_parts, part_labels, total_lev)
    fig, ax = plt.subplots(figsize=(11, 5.5))
    colors = ["#4477AA", "#66CCEE", "#228833", "#CCBB44", "#EE6677", "#AA3377"]
    x = range(len(part_labels))
    bars = ax.bar(x, cost_parts, color=colors[: len(cost_parts)], edgecolor="black", linewidth=0.6)
    ax.axhline(total_lev, color="gray", linestyle="--", linewidth=1.2, label=f"Total = {total_lev:.0f} EUR/t")
    ax.set_xticks(list(x))
    ax.set_xticklabels(part_labels, fontsize=12)
    ax.set_ylabel("Levelized cost (EUR/t methanol)", fontsize=13)
    ax.set_title("Levelized cost contributions (methanol basis)", fontsize=14)
    ax.tick_params(axis="y", labelsize=11)
    ax.legend(loc="upper right", fontsize=11)
    ax.grid(axis="y", alpha=0.35)
    fig.tight_layout()
    for b in bars:
        h = b.get_height()
        ax.annotate(
            f"{h:.0f}",
            xy=(b.get_x() + b.get_width() / 2, h),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    d = os.path.dirname(out_path)
    if d:
        os.makedirs(d, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    if debug:
        print("plot_levelized_cost_contributions saved:", out_path)
    plt.close(fig)

combusted_t_per_a = ((m_pl + m_bio)*0.30 + m_ash + m_h2o) / 1000 # [t/yr]
added_H2_t_per_a = 2 * hydrogen_produced/(LHV_H2/3600) * FLH_gasifier /1000 # [t/yr]
consumed_O2_t_per_a = nO2_demand * 32 * 3600 * FLH_gasifier /1000 # [t/yr]
waste_t_per_a = capacity_sorting_target # [t/a]
methanol_t_per_a = annual_methanol # [t/a]
added_gasification_steam_t_per_a = n_steam_missing*(nC_pl + nC_bio)*0.70 * 18 /1000 # [t/yr]
mass_diff_t_per_a = (
    waste_t_per_a
    + added_H2_t_per_a
    + consumed_O2_t_per_a
    + added_gasification_steam_t_per_a
) - (methanol_t_per_a + combusted_t_per_a)


def plot_annual_mass_balance_bars(
    waste_t_per_a,
    methanol_t_per_a,
    combusted_t_per_a,
    added_H2_t_per_a,
    consumed_O2_t_per_a,
    added_gasification_steam_t_per_a,
    mass_diff_t_per_a,
    out_path="results/annual_mass_balance.png",
    debug=False,
):
    """Bar chart of annual mass streams (t/a) for the modeled balance."""
    import os

    masses = [
        waste_t_per_a,
        methanol_t_per_a,
        combusted_t_per_a,
        added_H2_t_per_a,
        consumed_O2_t_per_a,
        added_gasification_steam_t_per_a,
        mass_diff_t_per_a,
    ]
    labels = [
        "Waste\n(t/a)",
        "Methanol\n(t/a)",
        "Combusted\n(t/a)",
        "Added H2\n(t/a)",
        "Consumed O2\n(t/a)",
        "Gasif.\nsteam (t/a)",
        "Mass diff\n(t/a)",
    ]
    if debug:
        print("plot_annual_mass_balance_bars masses:", masses, labels)
    fig, ax = plt.subplots(figsize=(12.5, 5.5))
    colors = ["#4477AA", "#228833", "#CCBB44", "#66CCEE", "#EE6677", "#44AA99", "#888888"]
    x = range(len(labels))
    bars = ax.bar(x, masses, color=colors, edgecolor="black", linewidth=0.6)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel("Mass flow (t/a)", fontsize=13)
    ax.set_title("Annual mass balance (bar magnitudes)", fontsize=14)
    ax.tick_params(axis="y", labelsize=11)
    ax.grid(axis="y", alpha=0.35)
    fig.tight_layout()
    for b in bars:
        h = b.get_height()
        if h >= 0:
            y_annot = h
            va = "bottom"
            xytext = (0, 3)
        else:
            y_annot = h
            va = "top"
            xytext = (0, -12)
        ax.annotate(
            f"{h:,.0f}",
            xy=(b.get_x() + b.get_width() / 2, y_annot),
            xytext=xytext,
            textcoords="offset points",
            ha="center",
            va=va,
            fontsize=10,
        )
    d = os.path.dirname(out_path)
    if d:
        os.makedirs(d, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    if debug:
        print("plot_annual_mass_balance_bars saved:", out_path)
    plt.close(fig)


def plot_energy_balance_mw(
    input_fuel_mwth,
    q_syngas_mwth,
    hydrogen_produced_mwth,
    steam_mwth,
    q_synthesis_output_mwth,
    ph2_mwel,
    wcomp_mix_mwel,
    wcomp_h2_mwel,
    wcomp_recycle_mwel,
    power_input_mwel,
    ratio_synthesis,
    eff_total,
    out_path="results/energy_balance_mw.png",
    debug=False,
):
    """Thermal (MWth) and electrical (MWel) bar charts for the energy balance around synthesis."""
    import os

    if debug:
        print(
            "plot_energy_balance_mw",
            input_fuel_mwth,
            q_syngas_mwth,
            hydrogen_produced_mwth,
            steam_mwth,
            q_synthesis_output_mwth,
            ph2_mwel,
            wcomp_mix_mwel,
            wcomp_h2_mwel,
            wcomp_recycle_mwel,
            power_input_mwel,
        )
    fig, (ax_th, ax_el) = plt.subplots(2, 1, figsize=(11, 8), sharex=False)
    th_labels = [
        "Input fuel",
        "Syngas\n(LHV)",
        "H2 (electrolyzer\nLHV)",
        "Steam\ndemand",
        "Methanol\n(LHV)",
    ]
    th_vals = [
        input_fuel_mwth,
        q_syngas_mwth,
        hydrogen_produced_mwth,
        steam_mwth,
        q_synthesis_output_mwth,
    ]
    colors_th = ["#4477AA", "#228833", "#66CCEE", "#44AA99", "#CCBB44"]
    x_th = range(len(th_vals))
    bars_th = ax_th.bar(x_th, th_vals, color=colors_th, edgecolor="black", linewidth=0.6)
    ax_th.set_xticks(list(x_th))
    ax_th.set_xticklabels(th_labels, fontsize=12)
    ax_th.set_ylabel("Power (MWth)", fontsize=13)
    ax_th.set_title("Thermal energy rates", fontsize=14)
    ax_th.tick_params(axis="y", labelsize=11)
    ax_th.grid(axis="y", alpha=0.35)
    for b in bars_th:
        h = b.get_height()
        ax_th.annotate(
            f"{h:.1f}",
            xy=(b.get_x() + b.get_width() / 2, h),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    el_labels = ["Electrolyzer", "Comp.\nsyngas", "Comp.\nH2", "Recycle\n×3"]
    el_vals = [ph2_mwel, wcomp_mix_mwel, wcomp_h2_mwel, wcomp_recycle_mwel]
    colors_el = ["#EE6677", "#AA3377", "#882255", "#6699CC"]
    x_el = range(len(el_vals))
    bars_el = ax_el.bar(x_el, el_vals, color=colors_el, edgecolor="black", linewidth=0.6)
    ax_el.axhline(
        power_input_mwel,
        color="gray",
        linestyle="--",
        linewidth=1.2,
        label=f"Total electrical = {power_input_mwel:.1f} MWel",
    )
    ax_el.set_xticks(list(x_el))
    ax_el.set_xticklabels(el_labels, fontsize=12)
    ax_el.set_ylabel("Power (MWel)", fontsize=13)
    ax_el.set_title("Electrical energy rates", fontsize=14)
    ax_el.tick_params(axis="y", labelsize=11)
    ax_el.legend(loc="upper right", fontsize=11)
    ax_el.grid(axis="y", alpha=0.35)
    for b in bars_el:
        h = b.get_height()
        ax_el.annotate(
            f"{h:.1f}",
            xy=(b.get_x() + b.get_width() / 2, h),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    fig.suptitle(
        f"Energy balance (MW) — η_synth = Q_out / Q_in (Q_in mixes MWth + MWel in script) = {ratio_synthesis:.3f}   |   "
        f"η_total = Q_methanol / (fuel_th + power_el) = {eff_total:.3f}",
        fontsize=12,
        y=1.02,
    )
    fig.tight_layout()
    d = os.path.dirname(out_path)
    if d:
        os.makedirs(d, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    if debug:
        print("plot_energy_balance_mw saved:", out_path)
    plt.close(fig)


plot_levelized_cost_contributions(
    cost_parts=[
        CAPEX_sorting_lev,
        OPEX_sorting_lev,
        CAPEX_gasification_lev,
        OPEX_gasification_lev,
        CAPEX_H2_lev,
        OPEX_H2_lev,
    ],
    part_labels=[
        "Sorting\nCAPEX",
        "Sorting\nOPEX",
        "Gasif.\nCAPEX",
        "Gasif.\nOPEX",
        "H2\nCAPEX",
        "H2\nOPEX",
    ],
    total_lev=cost_lev,
    debug=False,
)
print("Saved levelized cost bar chart to results/levelized_cost_breakdown.png")

plot_annual_mass_balance_bars(
    waste_t_per_a=waste_t_per_a,
    methanol_t_per_a=methanol_t_per_a,
    combusted_t_per_a=combusted_t_per_a,
    added_H2_t_per_a=added_H2_t_per_a,
    consumed_O2_t_per_a=consumed_O2_t_per_a,
    added_gasification_steam_t_per_a=added_gasification_steam_t_per_a,
    mass_diff_t_per_a=mass_diff_t_per_a,
    debug=False,
)
print("Saved annual mass balance bar chart to results/annual_mass_balance.png")

plot_energy_balance_mw(
    input_fuel_mwth=input_fuel,
    q_syngas_mwth=q_syngas,
    hydrogen_produced_mwth=hydrogen_produced,
    steam_mwth=steam_mw,
    q_synthesis_output_mwth=q_synthesis_output,
    ph2_mwel=PH2,
    wcomp_mix_mwel=sum(Wcomp_mix),
    wcomp_h2_mwel=sum(Wcomp_H2),
    wcomp_recycle_mwel=Wcomp_recycle,
    power_input_mwel=power_input,
    ratio_synthesis=ratio_synthesis,
    eff_total=eff_total_plot,
    debug=False,
)
print("Saved energy balance bar chart to results/energy_balance_mw.png")

