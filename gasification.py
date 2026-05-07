import pandas as pd
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

# # ------------------------------------- GASIFICATION CUMULATIVE CASES (TOP 1->10) -------------------------------------
# n_cases = min(10, len(plants_clean))
# case_rows = []

# for n_case in range(1, n_cases + 1):
#     case_df = plants_clean.iloc[:n_case]

#     # Aggregate all included plants for this cumulative case
#     nC_pl = case_df["nC_pl"].sum() # [kmolC_pl/yr]
#     nH_pl = case_df["nH_pl"].sum() # [kmolH_pl/yr]
#     nO_pl = case_df["nO_pl"].sum() # [kmolO_pl/yr]
#     nC_bio = case_df["nC_bio"].sum() # [kmolC_bio/yr]
#     nH_bio = case_df["nH_bio"].sum() # [kmolH_bio/yr]
#     nO_bio = case_df["nO_bio"].sum() # [kmolO_bio/yr]
#     E_input = (case_df["LHV_pl"] * case_df["m_pl"]).sum() + (case_df["LHV_bio"] * case_df["m_bio"]).sum() # [MJ/yr]

#     # (1) Characterize cumulative fuel in CHxOy basis
#     x = (nH_pl + nH_bio) / (nC_pl + nC_bio) # [kmolH/kmolC]
#     y = (nO_pl + nO_bio) / (nC_pl + nC_bio) # [kmolO/kmolC]
#     n_steam_dry = y # [kmolH2O/kmolC]
#     z1 = x - 2 * n_steam_dry # [kmolH/kmolC]

#     # (2) Initial syngas guess with same assumptions as single case
#     n_steam_assumed = 1 # [kmolH2O/kmolC]
#     z2 = z1 / 2 + n_steam_assumed # [kmolH2/kmolC]
#     n_H2 = z2 # [kmolH2/kmolC]
#     E_H2 = n_H2 * LHV_H2 * (nC_pl + nC_bio) * 0.70 # [MJ/yr]
#     E_CO = E_input * 0.70 - E_H2 # [MJ/yr]
#     n_CO = (E_CO / LHV_CO) / ((nC_pl + nC_bio) * 0.70) # [kmolCO/kmolC]
#     n_CO2 = 1 - n_CO # [kmolCO2/kmolC]
#     n_O2 = n_CO2 + (n_CO - 1) / 2 # [kmolO2/kmolC]

#     # (3) WGS adjustment toward H2:CO = 2:1
#     rWGS = 0 # [kmolrxn/kmolC]
#     n_steam_wgs = 0 # [kmolH2O/kmolC]
#     E_product = LHV_H2 * n_H2 + LHV_CO * n_CO # [MJ/kmolC]
#     while n_H2 / n_CO > 2:
#         rWGS += 0.001 # [kmolrxn/kmolC]
#         n_H2 = n_H2 - rWGS # [kmolH2/kmolC]
#         n_CO = n_CO + rWGS # [kmolCO/kmolC]
#         n_CO2 = n_CO2 - rWGS # [kmolCO2/kmolC]
#         n_steam_wgs = rWGS
#         E_product = LHV_H2 * n_H2 + LHV_CO * n_CO # [MJ/kmolC]

#     E_product_total = E_product * ((nC_pl + nC_bio) * 0.70) # [MJ/yr]
#     E_eff = E_product_total / E_input # [-]

#     # (4) Steam and heat checks
#     q_evap = rw * 18 # [MJ/kmol]
#     q_wgs = 43 # [MJ/kmol]
#     n_steam_missing = n_steam_assumed - n_steam_dry - n_steam_wgs # [kmolH2O/kmolC]
#     Q_gasify = n_steam_missing * q_evap # [MJ/kmolC]
#     Q_dry = n_steam_dry * q_evap # [MJ/kmolC]
#     Q_wgs = n_steam_wgs * q_wgs # [MJ/kmolC]
#     Q_steam_demand = (Q_gasify + Q_dry + Q_wgs) * ((nC_pl + nC_bio) * 0.70) # [MJ/yr]
#     Q_steam_available = E_input * 0.30 # [MJ/yr]

#     case_rows.append({
#         "Case": f"Top {n_case}",
#         "Plants included": ", ".join(case_df["Name"].tolist()),
#         "n_H2 [kmol/kmolC]": n_H2,
#         "n_CO [kmol/kmolC]": n_CO,
#         "n_CO2 [kmol/kmolC]": n_CO2,
#         "n_O2 [kmol/kmolC]": n_O2,
#         "n_H2/n_CO [-]": n_H2 / n_CO,
#         "E_product [MJ/yr]": E_product_total,
#         "E_eff [-]": E_eff,
#         "Q_steam_demand [MJ/yr]": Q_steam_demand,
#         "Q_steam_available [MJ/yr]": Q_steam_available,
#         "Q_available/Q_demand [-]": Q_steam_available / Q_steam_demand,
#         "rWGS [kmolrxn/kmolC]": rWGS,
#     })

# cases_summary = pd.DataFrame(case_rows).round({
#     "n_H2 [kmol/kmolC]": 3,
#     "n_CO [kmol/kmolC]": 3,
#     "n_CO2 [kmol/kmolC]": 3,
#     "n_O2 [kmol/kmolC]": 3,
#     "n_H2/n_CO [-]": 3,
#     "E_product [MJ/yr]": 0,
#     "E_eff [-]": 3,
#     "Q_steam_demand [MJ/yr]": 0,
#     "Q_steam_available [MJ/yr]": 0,
#     "Q_available/Q_demand [-]": 3,
#     "rWGS [kmolrxn/kmolC]": 3,
# })

# print("\nCumulative gasification summary (Top 1 -> Top 10 by m_tot):")
# print(cases_summary.to_string(index=False))
# cases_summary.to_csv("results/gasification_trial.csv", index=False)

# -------------- COMPRESSION AND METHANOL SYNTHESIS --------------
print(" ")
q_synthesis = 0.087    # [MWsteam/MWH2]
eta_synthesis = 0.78    # [MWmethanol/MWH2+steam]
LHV_methanol = 19.8 # [MJ/kg]
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
# mH2 = nH2_s * 2                       # [kg/s] H2 production - do not delete, save for later
QH2 = nH2_s * LHV_H2                    # [MWth] 
PH2 = QH2/eta_electrolyzer              # [MWel] power demand

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
print("H2 compression totals: Wcomp", sum(Wcomp_H2), "Qcool", sum(Qcool_H2))

Qsteam_synthesis = q_synthesis * QH2                 # [MWth]  NOTE: MUST CHECK THIS IN DANISH ENERGY AGENCY! I FORGOT THE IMPORTANT CO PORTION!
Qmethanol = eta_synthesis * (QH2 + Qsteam_synthesis) # [MWth] NOTE: THIS IS JUST THE CO2-portion, NOT THE CO+H2 PORTION!
print("Qmethanol", Qmethanol, "MWth")
m_methanol = Qmethanol/LHV_methanol /1000*3600*24    # [t/day]
Qmethanol = Qmethanol * FLH_gasifier                 # [MWh/yr]

print(" ")
print("Qsteam_synthesis", Qsteam_synthesis*FLH_gasifier*3.6) # [MJ/yr] 3.6 is from MWh to MJ
print("Qsteam_synthesis/Q_steam_available", Qsteam_synthesis*FLH_gasifier*3.6 / Q_steam_available)
print("Qmethanol", Qmethanol*FLH_gasifier*3.6)
print("m_methanol", m_methanol)
print("=> Seems like the available steam is sufficient for all demands, but this must be verified for all cumulative plant combinations!")

# -------------- AUXILIARY OXYGEN AND HEAT PUMP/ELECTROLYZER DEMANDS ----------
print(" ")
nO2_demand = n_O2 * nC_gasified / FLH_gasifier / 3600  # [kmolO2/s]
nO2_electrolyzer = nH2_s / 2 # [kmolO2/s] electrolysis stoichiometry for 2H2O -> O2 + 2H2
print("nO2_demand", nO2_demand)
print("nO2_electrolyzer", nO2_electrolyzer)
print("nO2_demand/nO2_electrolyzer", nO2_demand / nO2_electrolyzer)
print("=> Seems like the available oxygen is sufficient for all demands, but this must be verified for all cumulative plant combinations!")

print("Electrolyzer power demand", PH2, "MWel")
print("Relative to E_input", PH2 / (E_input/3600/FLH_gasifier), "MWel/MWth")