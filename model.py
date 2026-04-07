import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import searoute as sr
import CoolProp.CoolProp as CP

def get_CoolProp():

    """
    Returns:
        dict: Dictionary containing:
            - 'CO2': Dictionary with temperature as key and properties as values
            - 'H2': Dictionary with temperature as key and properties as values
            Properties include:
            - kappa: specific heat ratio (cp/cv)
            - cp: specific heat at constant pressure [kJ/kgK]
            - cv: specific heat at constant volume [kJ/kgK]
    """
    # Temperature range in Kelvin (from -20°C to 200°C)
    T_range = np.linspace(253.15, 473.15, 100)
    
    # Initialize dictionaries for each gas
    thermo_props = {
        'CO2': {
            'temperatures': T_range,
            'kappa': np.zeros_like(T_range),
            'cp': np.zeros_like(T_range),
            'cv': np.zeros_like(T_range)
        },
        'H2': {
            'temperatures': T_range,
            'kappa': np.zeros_like(T_range),
            'cp': np.zeros_like(T_range),
            'cv': np.zeros_like(T_range)
        }
    }
    
    # Calculate properties for CO2
    for i, T in enumerate(T_range):
        thermo_props['CO2']['kappa'][i] = CP.PropsSI('CPMASS', 'T', T, 'P', 1e5, 'CO2') / \
                                        CP.PropsSI('CVMASS', 'T', T, 'P', 1e5, 'CO2')
        thermo_props['CO2']['cp'][i] = CP.PropsSI('CPMASS', 'T', T, 'P', 1e5, 'CO2') / 1000  # Convert to kJ/kgK
        thermo_props['CO2']['cv'][i] = CP.PropsSI('CVMASS', 'T', T, 'P', 1e5, 'CO2') / 1000  # Convert to kJ/kgK
    
    # Calculate properties for H2
    for i, T in enumerate(T_range):
        thermo_props['H2']['kappa'][i] = CP.PropsSI('CPMASS', 'T', T, 'P', 1e5, 'Hydrogen') / \
                                       CP.PropsSI('CVMASS', 'T', T, 'P', 1e5, 'Hydrogen')
        thermo_props['H2']['cp'][i] = CP.PropsSI('CPMASS', 'T', T, 'P', 1e5, 'Hydrogen') / 1000  # Convert to kJ/kgK
        thermo_props['H2']['cv'][i] = CP.PropsSI('CVMASS', 'T', T, 'P', 1e5, 'Hydrogen') / 1000  # Convert to kJ/kgK
    
    return thermo_props

def shipping_adjustment(df, scaling=0.67, debug=False):
    """
    Adjust shipping costs by adding 0.5Mt capacity data and extrapolating 
    costs for 1st and 2nd shortest distances using linear regression.
    
    Args:
        df: DataFrame with shipping cost data
        debug: If True, print debug information
    
    Returns:
        Adjusted DataFrame with extrapolated values
    """
    if debug:
        print("Starting shipping adjustment...")
    
    # Add shipping costs for 0.5Mt
    capex_reference = df['optimist_1Mt']*1 # absolute costs for 1Mt
    capex_optimist = capex_reference * (0.5/1)**scaling
    df['optimist_0.5Mt'] = capex_optimist / 0.5 # specific costs for 0.5Mt

    capex_reference = df['pessimist_1Mt']*1 
    capex_pessimist = capex_reference * (0.5/1)**scaling
    df['pessimist_0.5Mt'] = capex_pessimist / 0.5 

    column_order = ['distance', 'optimist_0.5Mt', 'pessimist_0.5Mt', 'optimist_1Mt', 'pessimist_1Mt', 
                    'optimist_2Mt', 'pessimist_2Mt', 'optimist_3Mt', 'pessimist_3Mt']
    shipping_df = df[column_order]

    # Extrapolate costs for 1st and 2nd shortest distances using linear regression
    # Sort by distance to get shortest distances first
    shipping_sorted = shipping_df.sort_values('distance').reset_index(drop=True)

    # Use 3rd-6th shortest distances (indices 2-5) to predict 1st-2nd (indices 0-1)
    train_distances = shipping_sorted['distance'].iloc[2:6].values.reshape(-1, 1)
    target_distances = shipping_sorted['distance'].iloc[0:2].values.reshape(-1, 1)

    for mt in [0.5, 1, 2, 3]:
        for scenario in ['optimist', 'pessimist']:
            column = f"{scenario}_{mt}Mt"
            # Train on 3rd-6th distances
            train_costs = shipping_sorted[column].iloc[2:6].values
            reg = LinearRegression()
            reg.fit(train_distances, train_costs)
            predicted_costs = reg.predict(target_distances)
            
            # Update the dataframe - hard code replaced!
            shipping_sorted.loc[0, column] = predicted_costs[0]
            shipping_sorted.loc[1, column] = predicted_costs[1]

    shipping_df = shipping_sorted.sort_index()

    if debug:
        print("Debug - Extrapolated shipping costs:")
        print(shipping_df[['distance', 'optimist_0.5Mt', 'pessimist_0.5Mt']].round(2))
    
    return shipping_df

def compression_energy(m, T1, P1, thermo_props, gas='CO2', n_stages=4, pr=3.0, Tdiff=30, n_is=0.8, debug=False):
    """Multi-stage compression with intercooling. Returns per-stage lists of work [MW],
    cooling [MW], pressures [bar], and temperatures [K]."""
    Wcomp, Qcool, P_list, T_list = [], [], [P1], [T1]
    T, P = T1, P1
    temps = thermo_props[gas]['temperatures']

    for i in range(n_stages):
        kappa = np.interp(T, temps, thermo_props[gas]['kappa'])
        cp_in = np.interp(T, temps, thermo_props[gas]['cp'])
        P_next = P * pr
        T_is = T * (pr) ** ((kappa - 1) / kappa)
        T_act = T + (T_is - T) / n_is
        cp_out = np.interp(T_act, temps, thermo_props[gas]['cp'])
        W = m * (cp_in + cp_out) / 2 * (T_act - T) / 1000   # [MW]
        Q = 0 if i == n_stages - 1 else m * cp_out * (T_act - (T + Tdiff)) / 1000  # [MW]n no cooling in last stage
        Wcomp.append(W)
        Qcool.append(Q)
        P_list.append(P_next)
        T_list.append(T_act)
        T, P = T + Tdiff, P_next # NOTE: We neglect that temperatures wouldn't increase (but in reality needs intercooling and then final heating to synth temp).

    if debug:
        print(f"{gas} compression: {n_stages} stages, total W={sum(Wcomp):.2f} MW, total Q={sum(Qcool):.2f} MW")
    return Wcomp, Qcool, P_list, T_list

def levelize_kEUR(CAPEX, annual_CO2, x):
    """
    Levelize CAPEX in [kEUR] to [EUR/tCO2].

    """
    CRF = x["dr"] * (1 + x["dr"])**x["t"] / ((1 + x["dr"])**x["t"] - 1)
    CAPEX_annual = CAPEX * CRF  # [kEUR/yr]
    CAPEX_lev = CAPEX_annual / annual_CO2  # [EUR/tCO2]
    return CAPEX_lev

def plan_CCS(plant, c, x, l):

    # Burn fuel and capture/condition CO2
    annual_CO2 = plant["Total"] # [ktCO2/yr]
    FLH = plant["FLH"] # [h/yr]
    mCO2 = annual_CO2 * 1000 / FLH # [tCO2/h]
    mCO2 = mCO2 * 1000 / 3600 # [kgCO2/s]

    mCO2_captured = mCO2 * c["capture_rate"] # [kgCO2/s]
    Qreb = mCO2_captured * x["q_reb"]                       # [MW]
    Pcapture = x["p_capture"] * mCO2_captured/1000*3600     # [MW] 
    Pcondition = x["p_condition"] * mCO2_captured           # [MW]  

    # Penalize CHP and recover heat up to 100 % of original DH - use whatever power is available for HP
    Qsteam = plant["Qwaste"]
    P_old = plant["P"]
    Qdh_old = plant["Qdh"]
    P = P_old * (1 - Qreb/Qsteam)         # assuming live steam is used for reboiler
    Qdh = Qdh_old * (1 - Qreb/Qsteam)
    P = P - Pcapture - Pcondition

    Qrec_hex = c["q_hex"] * Qreb           # [MW] 
    Qdiff = Qdh_old - (Qdh + Qrec_hex)     # [MW] 
    if Qdiff < 0:
        raise ValueError
    Whp = Qdiff / x["COP"]             # [MW] may exceed P (grid purchase)
    Qdh = Qdh + Qrec_hex + Whp*x["COP"]    # restores Qdh to Qdh_old
    P -= Whp                           # negative P means grid power needed
    Ppenalty = (P_old - P) * FLH       # [MWh/yr] if negative P, the Ppenalty is inflated
    Qpenalty = (Qdh_old - Qdh) * FLH   # [MWh/yr]

    # Estimate CAPEX and on-site OPEX
    CEPCI_adjustment = x["CEPCI_scenario"] / c["CEPCI_reference"]
    CAPEX_capture = c["CAPEXref_capture"] * (annual_CO2/400) ** x["k"] * CEPCI_adjustment # [kEUR]
    CAPEX_capture_lev = levelize_kEUR(CAPEX_capture, annual_CO2, x) # [EUR/tCO2]
    CAPEX_HP = x["CAPEXref_HP"] * Whp*x["COP"] * CEPCI_adjustment # [kEUR] neglect HEX costs
    CAPEX_HP_lev = levelize_kEUR(CAPEX_HP, annual_CO2, x)            

    OPEX_fix = (CAPEX_capture * x["OPEXfix"]) / annual_CO2                           # [EUR/tCO2] 
    OPEX_makeup = x["camine"] * c['SEK_to_EUR']                                      # [EUR/tCO2]
    OPEX_energy = (Ppenalty*x["celc"] + Qpenalty*x["celc"]*x["cheat"]) / (annual_CO2 * 1000)  # [EUR/tCO2]
    OPEX = OPEX_fix + OPEX_makeup + OPEX_energy   

    # Calculate transport and storage costs based on (Ouvrey et al., 2024):
    loading_cost = 0
    truck_cost = 0
    pipeline_cost = 0
    rail_cost = 0
    shipping_cost = 0
    transport_cost = 0  # [EUR/tCO2]
    def _mode(val):
        return pd.notna(val) and str(val) != "None"

    def _loading_cost():
        CAPEX = c["CAPEXref_loading"] * c["SEK_to_EUR"] / 1000 * (annual_CO2 / 150) ** x["k"] * CEPCI_adjustment  # [kEUR]
        return levelize_kEUR(CAPEX, annual_CO2, x)

    if _mode(plant.get('Truck_distance')):
        loading_cost = _loading_cost()
        distance = float(plant['Truck_distance']) # [km]
        a1, a2 = 0.15, 5.58 
        UC = a1 + a2 / distance # [€/(t*km)]
        truck_cost = UC * distance # [€/tCO2]

    if _mode(plant.get('Pipeline_distance')):
        distance = float(plant['Pipeline_distance'])/1000 # [km]
        a1, a2, a3, a4 = 0.02, 260, 0.07, -0.61
        UC = a1 + a2 * (distance / 1)**a3 * (annual_CO2*1000*c["capture_rate"] / 1)**a4 # [€/(t*km)]
        pipeline_cost = UC * distance # [€/tCO2]

    if _mode(plant.get('Rail_distance')):
        loading_cost += _loading_cost()
        distance = float(plant['Rail_distance'])       # [km]
        cycle_time = (distance / 60 + 5) * 2           # [h] roundtrip (*2) @ 60 km/h + 5h unload
        capacity = 15 * 60 / cycle_time                # [tCO2/h] @15 wagons, 60 t/wagon

        CAPEX_train = c["CAPEXref_train"] * CEPCI_adjustment            # [EUR]
        CAPEXlev_train = levelize_kEUR(CAPEX_train/1000, annual_CO2, x) # [EUR/tCO2]
        OPEX_train = x["OPEXfix"]*CAPEX_train + 0.0269*(capacity*(distance*2*365)) # [EUR/yr] 1 roundtrip per day is more than enough!
        OPEX_train = OPEX_train / (annual_CO2*1000)                                # [EUR/tCO2]

        rail_cost = CAPEXlev_train + OPEX_train

    if _mode(plant.get('Oygarden_distance')):
        if x["storage"] == "oygarden":
            distance = float(plant['Oygarden_distance'])
        elif x["storage"] == "kalundborg":
            distance = float(plant['Kalundborg_distance'])
        df_ship = c["shipping_costs"].sort_values('distance')
        shipping_cost = float(np.interp(distance, df_ship['distance'].values, df_ship[x["shipping_case"]].values))

    transport_cost = loading_cost + truck_cost + pipeline_cost + rail_cost + shipping_cost
    cost_CCS = CAPEX_capture_lev + CAPEX_HP_lev + OPEX + transport_cost + x["storage_cost"]  # [EUR/tCO2]

    fossil = plant["Fossil"] / plant["Total"]                       # [tfossil/t] 
    biogenic = 1 - fossil                                           # [tbiogenic/t] 
    fossil -= x["carbon_change"]
    biogenic += x["carbon_change"]
    FCCS = annual_CO2 * fossil                                # [ktCO2/yr]
    BECCS = annual_CO2 * biogenic                             # [ktCO2/yr]
    if x["CRC"] < x["ETS"]:
        x["CRC"] = x["ETS"] # In such cases (~50%), CRCs are assumed integrated into the ETS
    strike_price = cost_CCS - biogenic * x["CRC"]                        # [EUR/tCO2] relative to a fossil ETS reference price

    cost_details = {
        "cost_CCS": cost_CCS,
        "CAPEX_capture_lev": CAPEX_capture_lev,
        "CAPEX_HP_lev": CAPEX_HP_lev,
        "OPEX_fix": OPEX_fix,
        "OPEX_makeup": OPEX_makeup,
        "OPEX_energy": OPEX_energy,
        "loading_cost": loading_cost,
        "truck_cost": truck_cost,
        "pipeline_cost": pipeline_cost,
        "rail_cost": rail_cost,
        "shipping_cost": shipping_cost,
        "storage_cost": x["storage_cost"],
    }
    return strike_price, FCCS, BECCS, Ppenalty, Qpenalty, cost_details

def plan_CCU(plant, c, x, l):
    
    # Burn fuel and capture/condition CO2
    annual_CO2 = plant["Total"] # [ktCO2/yr]
    FLH = plant["FLH"] # [h/yr]
    mCO2 = annual_CO2 * 1000 / FLH # [tCO2/h]
    mCO2 = mCO2 * 1000 / 3600 # [kgCO2/s]

    mCO2_captured = mCO2 * c["capture_rate"] # [kgCO2/s]
    Qreb = mCO2_captured * x["q_reb"]                       # [MW]
    Pcapture = x["p_capture"] * mCO2_captured/1000*3600     # [MW] 

    # Produce H2 from AEL electrolyzer
    LHV_H2 = 241.82                       # [kJ/molH2]
    nCO2 = mCO2_captured/44               # [kmol/s]
    nH2 = nCO2 * 3                        # [kmol/s] synthesis stoichiometry
    mH2 = nH2 * 2                         # [kg/s] 
    QH2 = nH2 * LHV_H2                    # [MWth] H2 production 
    PH2 = QH2/x["eta_electrolyzer"]         # [MWel] power demand

    # Compress CO2: 3 stages @ pr=3.8, from 40°C / 1 bar [Deng, 2019]
    Wcomp_CO2, Qcool_CO2, P_CO2, T_CO2 = compression_energy(
        mCO2_captured, T1=40+273.15, P1=1.0, thermo_props=c["thermo_props"],
        gas='CO2', n_stages=3, pr=3.8, Tdiff=30, n_is=c["eta_is"])

    # Compress H2: 2 stages @ pr=1.7, from 75°C / 20 bar [Jacobsson & Palmgren, 2025]
    Wcomp_H2, Qcool_H2, P_H2, T_H2 = compression_energy(
        mH2, T1=75+273.15, P1=20, thermo_props=c["thermo_props"],
        gas='H2', n_stages=2, pr=1.7, Tdiff=60, n_is=c["eta_is"])

    # Produce methanol (check Danish Agency Agency for method - we treat the whole synthesis plant as a single unit):
    LHV_methanol = 19.8 # [MJ/kg] [Formelsamling]
    Qsteam_synthesis = c["q_synthesis"] * QH2                 # [MWth] 
    Qmethanol = x["eta_synthesis"] * (QH2 + Qsteam_synthesis) # [MWth]
    m_methanol = Qmethanol/LHV_methanol /1000*3600*24         # [t/day]
    Qmethanol = Qmethanol * FLH                               # [MWh/yr]

    # Penalize CHP and recover Qdh
    Qsteam = plant["Qwaste"]
    P_old = plant["P"]
    Qdh_old = plant["Qdh"]
    P = P_old * (1 - Qreb/Qsteam - Qsteam_synthesis/Qsteam) # [MWel] live steam for synthesis
    P = P - Pcapture - PH2 - sum(Wcomp_CO2) - sum(Wcomp_H2)   # [MWel]
    Ppenalty = (P_old - P) * FLH                         # [MWh/yr] probably very positive

    Qdh = Qdh_old * (1 - Qreb/Qsteam - Qsteam_synthesis/Qsteam)
    Qdh = Qdh + sum(Qcool_CO2) + sum(Qcool_H2) # [MWth] the cooling is NECESSARY
    Qrec_hex = c["q_hex"] * Qreb           # [MWth] 
    Qrec_elec = c["q_electrolyzer"] * PH2   # [MWth]
    Qrec_distill = c['q_distill'] * (QH2 + Qsteam_synthesis) # [MWth] NOTE: optimistic assumption on heat recovery, from condensers at distillation
    Qavailable = Qrec_hex + (Qrec_elec + Qrec_distill)*x["heat_optimism"] # [MWth] assumed "free" heat exchange
    Qdiff = Qdh_old - (Qdh + Qavailable) # [MWth]
    Whp = 0
    if Qdiff < 0:
        Qdh = Qdh_old
    elif Qdiff > 0:
        Whp = Qdiff / x["COP"]                # [MW] may exceed P (grid purchase)
        Qdh = Qdh + Qavailable + Whp*x["COP"] # restores Qdh to Qdh_old
        P -= Whp                              # negative P means grid power needed
    Qpenalty = (Qdh_old - Qdh) * FLH   # [MWh/yr]
    print(Qdiff, Whp, Qpenalty)


    strike_price = 0
    FCCU = 0
    BCCU = 0
    Ppenalty = 0
    Qpenalty = 0
    Qmethanol = 0
    cost_details = {}
    return strike_price, FCCU, BCCU, Ppenalty, Qpenalty, Qmethanol, cost_details

def WACCUS_EPR(
    # [C] Constants
    EPR_design="Recovery", # [Mitigation, Recovery, Circularity]
    plants_df=None, 
    shipping_costs=None,
    truck_costs=None,          
    compression_costs=None,
    thermo_props=None,    
    SEK_to_EUR=0.091,
    profit=0.10,
    plot_results=False,

    CPI2015=314.21, # [SCB]
    CPI2025=417.96, # [SCB]
    CAPEXref_capture = 3550*0.09*1000,  # [MNOK]->[kEUR] @400 ktCO2/yr [Gassnova, Demonstrasjon av Fullskala CO2-Håndtering - Rapport for Avsluttet Forprosjekt]

    capture_rate = 0.90,    # [-] 
    q_hex = 0.64,           # [MWth/MWreb] [Beiron, 2022] assumed heat exhange from capture plant
    q_electrolyzer = 0.154,   # [MWth/MWel] [AEL tech, Fig2.1 MSc Jacobsson & Palmgren, 2025] OR [Danish Renwable Fuels 100MW AEC]
    eta_is = 0.80,
    q_synthesis = 0.087,    # [MWsteam/MWH2] about 0.08/(1-0.08)*QH2 [Danish Renewable Fuels Fig3, section 5.2 Methanol from Hydrogen and Carbon Dioxide]
    q_distill = 0.20,       # [MWth/MWH2+steam]
    CEPCI_reference = 600,  # [-] [University of Manchester, 2025] applies to reference CAPEX values

    # [X] Uncertainties
    baseline_granulates = 1258597, # [t/a] [IVL]
    inc_granulates = 0.10, # [-] [0-30%]
    shift_granulates = 0.20, # [-] [0-40%]
    price_granulates = 13000, # [SEK/tpl] [IVL]

    baseline_products = 884393, # [t/a] [IVL]
    inc_products = 0.10, # [-] [0-30%]
    shift_products = 0.20, # [-] [0-40%]
    stringent_products = 0.50, # [-] [0-100%]
    price_products = 46000, # [SEK/tpl] [IVL]

    q_reb = 3.5,            # [MJ/kgCO2] [2.5-3.5] [Soroodan, 2026]
    p_capture = 0.1,        # [MWh/tCO2] [Beiron, 2022]
    p_condition = 0.37,     # [MJ/kgCO2] [Kumar, 2023]
    COP = 3,                # [MWth/MWel]
    eta_electrolyzer = 0.699, # [MWH2/MWel] Table2.1 MSc Jacobsson & Palmgren (2025)
    eta_synthesis = 0.78,    # [MWmethanol/MWH2+steam]
    heat_optimism = 0.15,    # [-] [0-1.0] [0-100%] optimistic assumption on heat recovery, from condensers at distillation

    k = 0.67,                           # [-] [Stenström, 2025] assumed economy-of-scale factor
    CEPCI_scenario = 900,               # [-] [University of Manchester, 2025] applies to reference CAPEX values
    dr = 0.075,                         # [-]
    t = 25,                             # [yr]
    CAPEXref_HP = 860,                  # [kEUR/MWth] [Bergander & Hellander, 2025]
    CAPEXref_loading = 63000000,        # [SEK*] @150 ktCO2/yr excluding railway track [Koldioxid på tåg, 2024]
    CAPEXref_train = 8610000,           # [EUR*] an oversized train @15 wagons, cost = 4.98 *10**6 + 242*15 *10**3 [MSc Gunnarsson, 2025]
    OPEXfix = 0.04,                     # [-] % of base CAPEX, calculated from [Ramboll-Malmö, 2023]

    camine = 44,            # [SEK/tCO2] [Ramboll-Malmö, 2023]
    celc = 60,              # [EUR/MWh]
    cheat = 0.75,           # [% of elc]

    shipping_case = "pessimist_1Mt", # The main transport uncertainty! Dictates 1Mt, 2Mt, or 3Mt costs.
    storage = "oygarden",   # ["oygarden", "kalundborg"]
    storage_cost = 20, # [EUR/tCO2]

    carbon_change = 0.10,   # [-] [-0.10,0.10] [Malder, 2023] fraction of biogenic carbon
    CRC = 100,              # [EUR/tCO2]
    ETS = 80,               # [EUR/tCO2] Use this report: The EU-ETS Price Through 2030 and Beyond: A closer look at drivers, models and assumptions (https://www.ecologic.eu/19034)
    
    # [L] Levers
    EPR_products = True,
    EPR_fee = 100, # [EUR/tpl]
):
    # Store parameters in dicts
    c = {
        "plants_df": plants_df,
        "shipping_costs": shipping_costs,
        "truck_costs": truck_costs,
        "compression_costs": compression_costs,
        "thermo_props": thermo_props,
        "SEK_to_EUR": SEK_to_EUR,
        "profit": profit,

        "CAPEXref_capture": CAPEXref_capture,
        "CAPEXref_loading": CAPEXref_loading,
        "CAPEXref_train": CAPEXref_train,
        "capture_rate": capture_rate,
        "q_hex": q_hex,
        "q_electrolyzer": q_electrolyzer,
        "q_synthesis": q_synthesis,
        "q_distill": q_distill,
        "eta_is": eta_is,
        "CEPCI_reference": CEPCI_reference,
    }
    x = {
        "q_reb": q_reb,
        "p_capture": p_capture,
        "p_condition": p_condition,
        "COP": COP,
        "eta_electrolyzer": eta_electrolyzer,
        "eta_synthesis": eta_synthesis,
        "heat_optimism": heat_optimism,

        "k": k,
        "CEPCI_scenario": CEPCI_scenario,
        "dr": dr,
        "t": t,
        "CAPEXref_HP": CAPEXref_HP,
        "OPEXfix": OPEXfix,

        "camine": camine,
        "celc": celc,
        "cheat": cheat,

        "shipping_case": shipping_case,
        "storage": storage,
        "storage_cost": storage_cost,

        "carbon_change": carbon_change,
        "CRC": CRC,
        "ETS": ETS,
    }
    l = {}

    # (1) Simulate EPR fee
    plastic_supply = baseline_granulates * (1 + inc_granulates) * (1 - shift_granulates) # [tpl/yr]
    if EPR_products:
        plastic_supply += baseline_products * (1 + inc_products) * (1 - shift_products) * stringent_products # [tpl/yr]
    available_subsidies = plastic_supply * EPR_fee # [EUR/yr]

    granulate_inc = EPR_fee / (price_granulates * CPI2025/CPI2015 * SEK_to_EUR) # [-]
    products_inc = EPR_fee / (price_products * CPI2025/CPI2015 * SEK_to_EUR) # [-]
    print(f"Granulate inc: {granulate_inc}, Products inc: {products_inc}")

    # (2) Simulate EPR subsidies per case
    if EPR_design == "Mitigation":
        bids = []
        for _, plant in plants_df.iterrows():
            strike_price, FCCS, BECCS, Ppenalty, Qpenalty, cost_details = plan_CCS(plant, c, x, l)
            bids.append({"Name": plant["Name"], "Design": EPR_design, "strike_price": strike_price,
                            "FCCS": FCCS, "BECCS": BECCS, "Ppenalty": Ppenalty, "Qpenalty": Qpenalty, "cost_details": cost_details})

        # Plot cost breakdown for highest and lowest cost_CCS plants
        if plot_results:
                sorted_costs = sorted(bids, key=lambda r: r["cost_details"]["cost_CCS"])
                for case in [sorted_costs[0], sorted_costs[-1]]:
                    details = case["cost_details"]
                    labels = [k for k, v in details.items() if v > 0]
                    values = [v for v in details.values() if v > 0]

                    fig, ax = plt.subplots(figsize=(8, 4))
                    ax.barh(labels, values, color=plt.cm.magma(np.linspace(0.2, 0.8, len(values))))
                    ax.set_xlabel("Cost [EUR/tCO2]", fontsize=13)
                    ax.set_title(f"Cost breakdown — {case['Name']}", fontsize=14)
                    ax.tick_params(labelsize=12)
                    fig.tight_layout()

    elif EPR_design == "Recovery":
        bids = []
        for _, plant in plants_df.iterrows():
            strike_price, FCCU, BCCU, Ppenalty, Qpenalty, Qmethanol, cost_details = plan_CCU(plant, c, x, l)
            bids.append({"Name": plant["Name"], "Design": EPR_design, "strike_price": strike_price,
                            "FCCU": FCCU, "BCCU": BCCU, "Ppenalty": Ppenalty, "Qpenalty": Qpenalty, "Qmethanol": Qmethanol, "cost_details": cost_details})

    elif EPR_design == "Circularity":
        print("Circularity not implemented yet")
    
    # (3) Distribute subsidies and calculate KPIs
    bids.sort(key=lambda x: x['strike_price'])
    results = {}

    if EPR_design == "Mitigation":
        remaining_fund = available_subsidies
        awarded_plants = []
        for bid in bids:
            requested_amount = (bid['strike_price'] - x['ETS']) * (bid['FCCS'] + bid['BECCS'])*1000 # [EUR/yr]
            requested_amount *= (1 + c["profit"]) # [EUR/yr]
            awarded = requested_amount <= remaining_fund # [EUR/yr]
            if awarded:
                remaining_fund -= requested_amount
            awarded_plants.append({
                'Name': bid['Name'],
                'Design': EPR_design,
                'Awarded': awarded,
                'Strike price': bid['strike_price'],
                'FCCS': bid['FCCS'],
                'BECCS': bid['BECCS'],
                'Ppenalty': bid['Ppenalty'],
                'Qpenalty': bid['Qpenalty'],
                'Cost details': bid['cost_details'],
            })
        results['KPI1'] = plastic_supply * 10**-3 # [ktpl/yr]
        results['KPI2'] = EPR_fee # [EUR/tpl]
        results['KPI3'] = available_subsidies * 10**-3 # [kEUR/yr]
        awarded_only = [p for p in awarded_plants if p['Awarded']]
        results['KPI4'] = len(awarded_only) # [n plants]
        results['KPI5'] = sum(p['FCCS'] for p in awarded_only) # [ktCO2/yr]
        results['KPI6'] = sum(p['BECCS'] for p in awarded_only) # [ktCO2/yr]
        results['KPI7'] = results['KPI5'] + results['KPI6'] # [ktCO2/yr]
        results['KPI8'] = sum(p['Ppenalty'] for p in awarded_only) * 10**-3 # [GWh/yr]
        results['KPI9'] = sum(p['Qpenalty'] for p in awarded_only) * 10**-3 # [GWh/yr]
    
    return results

if __name__ == "__main__":

    # Read data
    plants_df = pd.read_csv('data/plants_clean.csv')
    shipping_costs = pd.read_csv('data/shipping_costs.csv')
    truck_costs = pd.read_csv('data/truck_costs.csv')
    compression_costs = pd.read_csv('data/compression_costs.csv')
    thermo_props = get_CoolProp()

    # Adjust dataframes: add 0.5Mt shipping costs, convert SEK to EUR
    SEK_to_EUR = 0.091
    shipping_costs = shipping_adjustment(shipping_costs, scaling=0.67, debug=False)
    cost_columns = [col for col in shipping_costs.columns if col != 'distance']
    shipping_costs[cost_columns] = shipping_costs[cost_columns] * SEK_to_EUR
    truck_costs["EUR/ton"] = truck_costs["SEK/ton"] * SEK_to_EUR
    truck_costs = truck_costs.drop(columns=["SEK/ton"])

    # Run the model
    results = WACCUS_EPR(
        EPR_design="Recovery", # Mitigation, Recovery, Circularity 
        plants_df=plants_df, 
        shipping_costs=shipping_costs,
        truck_costs=truck_costs,
        compression_costs=compression_costs,
        thermo_props=thermo_props,    
        SEK_to_EUR=SEK_to_EUR,
        profit=0.10,
        plot_results=True,
    )

    print("\n----------------------------------------These are the simulation results:----------------------------------------")
    for k, v in results.items():
        print(f"{k}: {v}")
    plt.show()
