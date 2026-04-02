import pandas as pd
import numpy as np

plants_df = pd.read_csv("data/plants.csv")

for _, plant in plants_df.iterrows():
    

    # Known data
    Qsteam = plant["Qwaste"] # [MW LHV]
    eta_boiler = 0.85 # [MWsteam/MWfuel] NOTE: This makes the final FLH calculation work! Beiron assumption. NO! Take from EteknikKompendie p.86, gives range: 0,80-0,92
    Qlhv = Qsteam / eta_boiler 
    # Qfgc = plant["Qfgc"] # [MW QFC]
    # Qhhv = Qlhv+Qfgc # [MW HHV]
    Eb = plant["Biogenic"]*1000 # [tCO2/yr]
    Ep = plant["Fossil"]*1000 # [tCO2/yr]
    print(plant["Name"], round(Eb/(Eb+Ep), 2))
    # rH2O = 2.5 # [MJ/kg] water evaporation heat @0C
    
    # # Assume carbon content of the mixed plastic fraction and full load hours
    # ep = 0.0898 * 3600 /1000 # [kgCO2/MJ=>tCO2/MWh] [Chahat, 2024]
    # FLH = 8000 # [h/yr]

    # # Calculate energy balances
    # Qp = Ep/(FLH*ep) # [MW LHV]
    # Qb = Qtot - Qp # [MW LHV]
    # eb = Eb/(FLH*Qb) # [tCO2/MWh biogenic]
    # print(eb / 3600 *1000)
    
    # # Solving molecular balance for the total combustion reaction: (see Notes)
    # mCO2 = (Ep + Eb) / (Qlhv * FLH) # [tCO2/MWh] @LHV basis
    # nCO2 = mCO2*1000 /44 # [kmolCO2/MWh]  
    # nCO2 = nCO2 * Qlhv /3600 # [kmolCO2/s]

    nC_tot = (Eb + Ep)*1000/44 # [kmolC_tot/yr]
    nC_pl = Ep / (Eb + Ep) * nC_tot # [kmolC_pl/yr] 
    nC_bio = Eb / (Eb + Ep) * nC_tot # [kmolC_bio/yr]

    # Assuming the plastic and biomass C,H,O ratios from Chahat (2024) and Beiron (2026):
    nH_pl = nC_pl * 1.711 # [kmolH_pl/yr]
    nO_pl = nC_pl * 0.161 # [kmolO_pl/yr]
    ntot_pl = nC_pl + nH_pl + nO_pl # [kmol_pl/yr]
    m_pl = nC_pl * 12 + nH_pl * 1 + nO_pl * 16 # [kg_pl/yr] d.a.
    # M_pl = 12*1+1*1.711+16*0.161 # [kg_pl/kmol_pl] a "mean", a bit sketchy!

    nH_bio = nC_bio * 1.44 # [kmolH_bio/yr]
    nO_bio = nC_bio * 0.66 # [kmolO_bio/yr]
    ntot_bio = nC_bio + nH_bio + nO_bio # [kmol_bio/yr]
    m_bio = nC_bio * 12 + nH_bio * 1 + nO_bio * 16 # [kg_bio/yr] d.a.
    # M_bio = 12*1+1*1.44+16*0.66 # [kg_bio/kmol_bio] 

    # Estimating dry, ash-free mass flows and then LHV (Dulong’s formula ​(Hosokai et al., 2016))​:
    LHV_pl = 38.2*(nC_pl*12/m_pl) + 84.9*(nH_pl*1/m_pl - (nO_pl*16/m_pl)/8) - 0.62 # [MJ/kg d.a.]
    LHV_bio = 38.2*(nC_bio*12/m_bio) + 84.9*(nH_bio*1/m_bio - (nO_bio*16/m_bio)/8) - 0.62 # [MJ/kg d.a.]

    # eCO2_pl = Ep / (m_pl * LHV_pl) # [tCO2/MJ]
    # eCO2_bio = Eb / (m_bio * LHV_bio) #[tCO2/MJ]

    # Based on the dry waste (of plastic and biomass) we can calculate the additional water and ash using Hammar's values
    x_ash_dry = 0.206 # [kg_ash/kg_dry] calc. based on the DRY waste, Hammar
    x_water = 0.360 # [kg_moisture/kg_total] calc. based on wet waste, Hammar
    m_ash = -x_ash_dry*(m_bio+m_pl) / (x_ash_dry - 1) # [kg_ash/yr] formula from Wolfram Alpha
    m_dry = m_bio + m_pl + m_ash
    m_water = -x_water*m_dry / (x_water - 1)
    m_total = m_dry + m_water # [kg_waste/yr]

    rw = 2.5 # [MJ/kg] water evaporation
    FLH = (m_pl*LHV_pl + m_bio*LHV_bio - rw*m_water)/3600 / Qlhv # [h/yr]
    # FLH = 8000
    # m_water = ( (m_pl*LHV_pl + m_bio*LHV_bio) - FLH*Qlhv*3600 ) /rw # [kg_w/a]
    LHV_tot = (LHV_pl*m_pl + LHV_bio*m_bio - rw*m_water) / (m_total)
    
    # Calculate mass flows in kg/s
    m_total_s = round(m_total / FLH / 3600, 2)  # kg/s
    m_pl_s = round(m_pl / FLH / 3600 /m_total_s, 2)        # kg/s
    m_bio_s = round(m_bio / FLH / 3600 /m_total_s, 2)      # kg/s
    m_ash_s = round(m_ash / FLH / 3600 /m_total_s, 2)      # kg/s
    m_water_s = round(m_water / FLH / 3600 /m_total_s, 2)  # kg/s
    Wpl = LHV_pl*m_pl *10**-6 # TJ/a
    Wbio = LHV_bio*m_bio *10**-6 # TJ/a
    Wbio_wet = (LHV_bio*m_bio - rw*m_water) *10**-6 # TJ/a

    # Print as a table row
    print(
        f"{plant['Name']:20} | "
        f"Total: {m_total_s:>6} | "
        f"Pl: {m_pl_s:>6} | "
        f"Bio: {m_bio_s:>6} | "
        f"Ash: {m_ash_s:>6} | "
        f"H2O: {m_water_s:>6} | "
        f"FLH: {round(FLH):>5} | "
        f"LHV: {round(LHV_tot, 2):>5} | "
        f"Wp: {round(Wpl, 2):>5} | "
        f"Wbdry: {round(Wbio, 2):>5} | "
        f"Wbwet: {round(Wbio_wet, 2):>5} | "
    )

    # # Now we miss 3 unknowns but have 3 equations:
    # a1 = nC_pl
    # a2 = nH_pl
    # a3 = nO_pl
    # b1 = nC_bio
    # b2 = nH_bio
    # b3 = nO_bio
    # c3 = nCO2
    # dh_H2OL = -285.8 # [MJ/kmol]
    # dh_H2Og = -241.8 # [MJ/kmol]
    # dh_CO2 = -393.5 # [MJ/kmol]

    # qLHV = -Qlhv # Check sign!
    # A = np.array([ 
    # [1, 2, -1], 
    # [-2, 0, 2],
    # [-dh_H2OL, 0, dh_H2Og]])
    # B = np.array(
    #     [2*c3-a3-b3, 
    #     a2+b2, 
    #     qLHV-c3*dh_CO2]
    #     )
    # X = np.linalg.solve(A, B)
    # c1 = X[0] # [kmolH2O(liq)/s] in fuel
    # c2 = X[1] # [kmolAir(gas)/s] added
    # c4 = X[2] # [kmolH2O(gas)/s] in flue gases
    # print(c1, c2, c4)
    # print("Mass of carbon in fuel:", (a1+b1)*12)
    # print("Mass of water in fuel:", c1*18)
