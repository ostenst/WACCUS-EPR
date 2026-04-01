import pandas as pd
import numpy as np

plants_df = pd.read_csv("data/plants.csv")

for _, plant in plants_df.iterrows():
    print(plant["Name"])

    # Known data
    Qlhv = plant["Qwaste"] # [MW LHV]
    # Qfgc = plant["Qfgc"] # [MW QFC]
    # Qhhv = Qlhv+Qfgc # [MW HHV]
    Eb = plant["Biogenic"]*1000 # [tCO2/yr]
    Ep = plant["Fossil"]*1000 # [tCO2/yr]
    # rH2O = 2.5 # [MJ/kg] water evaporation heat @0C
    
    # # Assume carbon content of the mixed plastic fraction and full load hours
    # ep = 0.0898 * 3600 /1000 # [kgCO2/MJ=>tCO2/MWh] [Chahat, 2024]
    FLH = 8000 # [h/yr]

    # # Calculate energy balances
    # Qp = Ep/(FLH*ep) # [MW LHV]
    # Qb = Qtot - Qp # [MW LHV]
    # eb = Eb/(FLH*Qb) # [tCO2/MWh biogenic]
    # print(eb / 3600 *1000)
    
    # # Solving molecular balance for the total combustion reaction: (see Notes)
    # mCO2 = (Ep + Eb) / (Qlhv * FLH) # [tCO2/MWh] @LHV basis
    # nCO2 = mCO2*1000 /44 # [kmolCO2/MWh]  
    # nCO2 = nCO2 * Qlhv /3600 # [kmolCO2/s]

    nC_pl = Ep / (Eb + Ep) # [kmolC_pl/kmolC_tot] NOTE: BUG: WRONG UNITS
    nC_bio = Eb / (Eb + Ep) # [kmolC_bio/kmolC_tot]

    # Assuming the plastic and biomass C,H,O ratios from Chatat (2024) and Beiron (2026):
    nH_pl = nC_pl * 1.711 # [kmolC_pl/kmolC_tot / kmolH/kmolC_pl]
    nO_pl = nC_pl * 0.161 # [kmolO/s plastic]
    nH_bio = nC_bio * 1.44 # [kmolH/s biogenic]
    nO_bio = nC_bio * 0.66 # [kmolO/s biogenic]

    # Estimating dry, ash-free mass flows and then LHV (Dulong’s formula ​(Hosokai et al., 2016))​:
    m_pl = nC_pl * 12 + nH_pl * 1 + nO_pl * 16 # [kg/s]
    m_bio = nC_bio * 12 + nH_bio * 1 + nO_bio * 16 # [kg/s]
    LHV_pl = 38.2*(nC_pl*12/m_pl) + 84.9*(nH_pl*1/m_pl - (nO_pl*16/m_pl)/8) - 0.62 # [MJ/kg d.a.]
    LHV_bio = 38.2*(nC_bio*12/m_bio) + 84.9*(nH_bio*1/m_bio - (nO_bio*16/m_bio)/8) - 0.62 # [MJ/kg d.a.]
    

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
