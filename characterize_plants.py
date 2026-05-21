import pandas as pd
import numpy as np

plants_df = pd.read_csv("data/plants.csv")
plant_distances = pd.read_csv('data/plant_distances.csv') # NOTE: Should add inland distances to SiteZero
plants_df = plants_df.merge(plant_distances, on='Name', how='left')

rows = []
for _, plant in plants_df.iterrows():
    
    # Known data per plant
    Qsteam = plant["Qwaste"] # [MW LHV]
    Eb = plant["Biogenic"]*1000 # [tCO2/yr]
    Ep = plant["Fossil"]*1000 # [tCO2/yr]
   
    # Calculate molar flows, assuming (dry ash-free) plastic and biomass C,H,O ratios from Chahat (2024) and Beiron (2026):
    nC_tot = (Eb + Ep)*1000/44 # [kmolC_tot/yr]
    nC_pl = Ep / (Eb + Ep) * nC_tot # [kmolC_pl/yr] 
    nC_bio = Eb / (Eb + Ep) * nC_tot # [kmolC_bio/yr]

    nH_pl = nC_pl * 1.711 # [kmolH_pl/yr]
    nO_pl = nC_pl * 0.161 # [kmolO_pl/yr]
    ntot_pl = nC_pl + nH_pl + nO_pl # [kmol_pl/yr]

    # nH_bio = nC_bio * 1.44 # [kmolH_bio/yr] # I think this was wet biomasss... cf. Beiron (2026)
    # nO_bio = nC_bio * 0.66 # [kmolO_bio/yr]
    # nH_bio = nC_bio * 0.22 # [kmolH_bio/yr] # But this "dry" from Beiron is insane! I think its "without ANY oxygen!"
    # nO_bio = nC_bio * 0 # [kmolO_bio/yr]
    nH_bio = nC_bio * 1.23 # [kmolH_bio/yr] # Try this Table 5. forest residues: https://www.sciencedirect.com/science/article/pii/S0016236109004967#aep-section-id14
    nO_bio = nC_bio * 0.58 # [kmolO_bio/yr]

    ntot_bio = nC_bio + nH_bio + nO_bio # [kmol_bio/yr]

    # Estimating (dry ash-free) mass flows and then LHV using Dulong's formula (Hosokai et al., 2016):
    m_pl = nC_pl * 12 + nH_pl * 1 + nO_pl * 16 # [kg_pl/yr] d.a.
    m_bio = nC_bio * 12 + nH_bio * 1 + nO_bio * 16 # [kg_bio/yr] d.a.
    LHV_pl = 38.2*(nC_pl*12/m_pl) + 84.9*(nH_pl*1/m_pl - (nO_pl*16/m_pl)/8) - 0.62 # [MJ/kg d.a.]
    LHV_bio = 38.2*(nC_bio*12/m_bio) + 84.9*(nH_bio*1/m_bio - (nO_bio*16/m_bio)/8) - 0.62 # [MJ/kg d.a.]

    # Based on the dry waste (of plastic and biomass) we can calculate the additional water and ash using Hammar's (2022) values
    x_ash_dry = 0.206 # [kg_ash/kg_dry] calc. based on the DRY waste, Hammar
    x_h2o_frac = 0.360 # [kg_moisture/kg_total] calc. based on wet waste, Hammar
    m_ash = -x_ash_dry*(m_bio+m_pl) / (x_ash_dry - 1) # [kg_ash/yr] formula from Wolfram Alpha
    m_dry = m_bio + m_pl + m_ash
    m_h2o = -x_h2o_frac*m_dry / (x_h2o_frac - 1)
    m_tot = m_dry + m_h2o # [kg_waste/yr]

    x_pl = m_pl / m_tot # [kg_pl/kg_total]
    x_bio = m_bio / m_tot # [kg_bio/kg_total]
    x_ash = m_ash / m_tot # [kg_ash/kg_total]
    x_h2o = m_h2o / m_tot # [kg_water/kg_total]

    # From energy balance we can calculate the FLH and the total LHV of the fuel:
    rw = 2.5 # [MJ/kg] water evaporation
    eta_boiler = 0.85 # [MWsteam/MWfuel] Assumption from EteknikKompendie p.86, gives range: 0.80-0.92
     # Also check Danish Energy Agency p.96: Technology data - energy plants for electricity and district heating generation
    Qlhv = Qsteam / eta_boiler 
    FLH = (m_pl*LHV_pl + m_bio*LHV_bio - rw*m_h2o)/3600 / Qlhv # [h/yr]
    LHV_biowet = (LHV_bio*m_bio - rw*m_h2o) / (m_bio + m_h2o) # [MJ/kg ash-free]
    LHV_tot = (LHV_pl*m_pl + LHV_bio*m_bio - rw*m_h2o) / (m_tot)

    rows.append({
        "Name": plant["Name"], "FLH": FLH,
        "nC_pl": nC_pl, "nH_pl": nH_pl, "nO_pl": nO_pl,
        "nC_bio": nC_bio, "nH_bio": nH_bio, "nO_bio": nO_bio,
        "m_pl": m_pl, "m_bio": m_bio, "m_ash": m_ash,
        "m_h2o": m_h2o, "m_tot": m_tot,
        "x_pl": x_pl, "x_bio": x_bio, "x_ash": x_ash, "x_h2o": x_h2o,
        "LHV_pl": LHV_pl, "LHV_bio": LHV_bio,
        "LHV_biowet": LHV_biowet, "LHV_tot": LHV_tot,
        "Qlhv": Qlhv,
    })

# Merge computed columns into the plants dataframe
plants_clean = plants_df.merge(pd.DataFrame(rows), on="Name", how="left")
display_cols = {
    "Name": "Plant",
    "gasification_distance_km": "gasif_dist [km]",
    "FLH": "FLH [h/yr]",
    "m_pl": "m_pl [kg/yr]", "m_bio": "m_bio [kg/yr]",
    "m_ash": "m_ash [kg/yr]", "m_h2o": "m_h2o [kg/yr]", "m_tot": "m_tot [kg/yr]",
    "x_pl": "x_pl", "x_bio": "x_bio", "x_ash": "x_ash", "x_h2o": "x_h2o",
    "LHV_pl": "LHV_pl [MJ/kg]", "LHV_bio": "LHV_bio [MJ/kg]",
    "LHV_biowet": "LHV_bw [MJ/kg]", "LHV_tot": "LHV_tot [MJ/kg]",
    "Qlhv": "Qlhv [MW]",
}
summary = plants_clean[list(display_cols.keys())].rename(columns=display_cols)
round_rules = {
    "gasif_dist [km]": 0,
    "FLH [h/yr]": 0, "m_pl [kg/yr]": 0, "m_bio [kg/yr]": 0, "m_ash [kg/yr]": 0,
    "m_h2o [kg/yr]": 0, "m_tot [kg/yr]": 0,
    "x_pl": 3, "x_bio": 3, "x_ash": 3, "x_h2o": 3,
    "LHV_pl [MJ/kg]": 1, "LHV_bio [MJ/kg]": 1,
    "LHV_bw [MJ/kg]": 1, "LHV_tot [MJ/kg]": 1,
    "Qlhv [MW]": 1,
}
summary = summary.round(round_rules)

# Print a list of explanations for each column, for example: "m_pl [kg/yr] = dry ash-free plastic mass flow in the fuel"
explanations = {
    "gasif_dist [km]": "Road distance from plant to gasification site (plant_distances.csv)",
    "FLH [h/yr]": "Full load hours",
    "Qlhv [MW]": "Lower heating value of the total fuel",
    "nC_pl [kmol/yr]": "Molar flow of carbon in the dry ash-free plastic",
    "nH_pl [kmol/yr]": "Molar flow of hydrogen in the dry ash-free plastic",
    "nO_pl [kmol/yr]": "Molar flow of oxygen in the dry ash-free plastic",
    "nC_bio [kmol/yr]": "Molar flow of carbon in the dry ash-free biomass",
    "nH_bio [kmol/yr]": "Molar flow of hydrogen in the dry ash-free biomass",
    "nO_bio [kmol/yr]": "Molar flow of oxygen in the dry ash-free biomass",
    "m_pl [kg/yr]": "Dry ash-free plastic mass flow in the fuel",
    "m_bio [kg/yr]": "Dry ash-free biogenic mass flow in the fuel",
    "m_ash [kg/yr]": "Ash mass flow in the total fuel",
    "m_h2o [kg/yr]": "Moisture mass flow in the total fuel",
    "m_tot [kg/yr]": "Total mass flow in the total fuel",
    "x_pl [kg/kg]": "Mass fraction of plastic in the total fuel",
    "x_bio [kg/kg]": "Mass fraction of biogenic in the total fuel",
    "x_ash [kg/kg]": "Mass fraction of ash in the total fuel",
    "x_h2o [kg/kg]": "Mass fraction of moisture in the total fuel",
    "LHV_pl [MJ/kg]": "Lower heating value of the dry ash-free plastic",
    "LHV_bio [MJ/kg]": "Lower heating value of the dry ash-free biomass",
    "LHV_bw [MJ/kg]": "Lower heating value of the ash-free biomass INCLUDING moisture",
    "LHV_tot [MJ/kg]": "Lower heating value of the total fuel",
}
print("\nExplanations:")
for key, value in explanations.items():
    print(f"{key}: {value}")
print(summary.to_string(index=False))
print("\n")

for _, plant in plants_clean.iterrows():
    print(f"{plant['Name']}: {round(plant['FLH'], 0)} h/yr")

plants_clean.to_csv("data/plants_clean.csv", index=False)