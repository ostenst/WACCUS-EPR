import os
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
try:
    import searoute as sr
except ImportError:
    sr = None
import CoolProp.CoolProp as CP

def get_CoolProp():

    """
    Returns:
        dict: Dictionary containing:
            - 'CO2', 'H2', 'CO', 'H2O': Gas property dictionaries
            Properties include:
            - kappa: specific heat ratio (cp/cv)
            - cp: specific heat at constant pressure [kJ/kgK]
            - cv: specific heat at constant volume [kJ/kgK]
            - mw: molecular weight [kg/kmol]
    """
    # Temperature range in Kelvin (from -20°C to 200°C)
    T_range = np.linspace(253.15, 473.15, 100)
    
    # Initialize dictionaries for each gas
    gas_map = {
        'CO2': ['CO2'],
        'H2': ['Hydrogen', 'H2'],
        'CO': ['CO', 'CarbonMonoxide'],
        'H2O': ['Water', 'H2O'],
    }
    mw_map = {
        'CO2': 44.01,
        'H2': 2.016,
        'CO': 28.01,
        'H2O': 18.015,
    }
    thermo_props = {}
    for gas in gas_map:
        thermo_props[gas] = {
            'temperatures': T_range,
            'kappa': np.zeros_like(T_range),
            'cp': np.zeros_like(T_range),
            'cv': np.zeros_like(T_range),
            'mw': mw_map[gas],
        }
    
    # Calculate properties for all gases at 1 bar
    for gas, fluid_options in gas_map.items():
        cp_test = None
        cv_test = None
        selected_fluid = None
        for fluid in fluid_options:
            try:
                cp_test = CP.PropsSI('CPMASS', 'T', 300.0, 'P', 1e5, fluid)
                cv_test = CP.PropsSI('CVMASS', 'T', 300.0, 'P', 1e5, fluid)
                selected_fluid = fluid
                break
            except Exception:
                continue
        if selected_fluid is None:
            raise ValueError(f"Could not find CoolProp fluid for {gas}: tried {fluid_options}")
        for i, T in enumerate(T_range):
            # Water at 1 bar is not valid below melting point in this call.
            T_eval = max(T, 273.16) if gas == "H2O" else T
            cp_mass = CP.PropsSI('CPMASS', 'T', T_eval, 'P', 1e5, selected_fluid)
            cv_mass = CP.PropsSI('CVMASS', 'T', T_eval, 'P', 1e5, selected_fluid)
            thermo_props[gas]['kappa'][i] = cp_mass / cv_mass
            thermo_props[gas]['cp'][i] = cp_mass / 1000  # Convert to kJ/kgK
            thermo_props[gas]['cv'][i] = cv_mass / 1000  # Convert to kJ/kgK
    
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

def compression_energy(n, T1, P1, thermo_props, gas='CO2', n_stages=4, pr=3.0, Tdiff=30, n_is=0.8, debug=False):
    """Multi-stage compression with intercooling. Returns per-stage lists of work [MW],
    cooling [MW], pressures [bar], and temperatures [K]."""
    Wcomp, Qcool, P_list, T_list = [], [], [P1], [T1]
    T, P = T1, P1
    species = ["CO2", "H2", "CO", "H2O"]
    if isinstance(gas, dict):
        gas_mix = {sp: gas.get(sp, 0.0) for sp in species}
    else:
        gas_mix = {sp: 0.0 for sp in species}
        gas_mix[gas] = 1.0

    mix_sum = sum(gas_mix.values())
    if mix_sum <= 0:
        raise ValueError("gas mix must have positive molar flow or fraction")
    y_mix = {sp: gas_mix[sp] / mix_sum for sp in species}
    temps = thermo_props["CO2"]["temperatures"]

    for i in range(n_stages):
        T_in = T
        cp_mix_molar_in = 0
        cv_mix_molar_in = 0
        mw_mix = 0
        for gas_i, y_i in y_mix.items():
            cp_i = np.interp(T, temps, thermo_props[gas_i]['cp'])  # [kJ/kgK]
            cv_i = np.interp(T, temps, thermo_props[gas_i]['cv'])  # [kJ/kgK]
            mw_i = thermo_props[gas_i]['mw']  # [kg/kmol]
            cp_mix_molar_in += y_i * cp_i * mw_i  # [kJ/kmolK]
            cv_mix_molar_in += y_i * cv_i * mw_i  # [kJ/kmolK]
            mw_mix += y_i * mw_i  # [kg/kmol]
        kappa = cp_mix_molar_in / cv_mix_molar_in
        cp_in = cp_mix_molar_in / mw_mix  # [kJ/kgK]
        m_mass = n * mw_mix  # [kg/s]
        P_next = P * pr
        T_is = T * (pr) ** ((kappa - 1) / kappa)
        T_act = T + (T_is - T) / n_is
        cp_mix_molar_out = 0
        for gas_i, y_i in y_mix.items():
            cp_i = np.interp(T_act, temps, thermo_props[gas_i]['cp'])  # [kJ/kgK]
            mw_i = thermo_props[gas_i]['mw']  # [kg/kmol]
            cp_mix_molar_out += y_i * cp_i * mw_i  # [kJ/kmolK]
        cp_out = cp_mix_molar_out / mw_mix  # [kJ/kgK]
        W = m_mass * (cp_in + cp_out) / 2 * (T_act - T) / 1000   # [MW]
        Q = 0 if i == n_stages - 1 else m_mass * cp_out * (T_act - (T + Tdiff)) / 1000  # [MW]n no cooling in last stage
        Wcomp.append(W)
        Qcool.append(Q)
        P_list.append(P_next)
        T_list.append(T_act)
        T, P = T + Tdiff, P_next # NOTE: We neglect that temperatures wouldn't increase (but in reality needs intercooling and then final heating to synth temp).

        if debug:
            print(
                f"Stage {i+1}: Pin={P_list[i]:.2f} bar, Pout={P_next:.2f} bar, "
                f"Tin={T_in:.1f} K, Tout={T_act:.1f} K, cp={cp_in:.3f} kJ/kgK, "
                f"kappa={kappa:.3f}, W={W:.3f} MW, Q={Q:.3f} MW"
            )

    if debug:
        gas_label = f"mix({', '.join([f'{k}:{v:.3f}' for k, v in y_mix.items()])})"
        print(f"{gas_label} compression: {n_stages} stages, total W={sum(Wcomp):.2f} MW, total Q={sum(Qcool):.2f} MW")
    return Wcomp, Qcool, P_list, T_list

def compression_capex_eur(wcomp_list, compression_costs_df, debug=False):
    """Multi-stage compression train overnight CAPEX [EUR] [Deng, 2019]."""
    comp_df = compression_costs_df.set_index(compression_costs_df.columns[0])
    total = 0.0
    for i, W in enumerate(wcomp_list):
        W_kW = W * 1000  # [kW]
        a = comp_df.iloc[0, i]
        b = comp_df.iloc[1, i]
        coeff_c = comp_df.iloc[2, i]
        if i == 3:
            total += a + b * W_kW + coeff_c * W_kW**0.5
        else:
            total += a + b * W_kW**1.5 + coeff_c * W_kW**2
    if debug:
        print("compression_capex_eur", wcomp_list, "->", total)
    return total  # [EUR]


def levelize_kEUR(CAPEX, annual_mass_kt, x, debug=False):
    """Levelize overnight CAPEX [kEUR] on annual throughput [kt/a] → [EUR/t]."""
    CRF = x["dr"] * (1 + x["dr"]) ** x["t"] / ((1 + x["dr"]) ** x["t"] - 1)
    CAPEX_annual = CAPEX * CRF  # [kEUR/yr]
    CAPEX_lev = CAPEX_annual / annual_mass_kt  # [EUR/t]
    if debug:
        print("levelize_kEUR", CAPEX, annual_mass_kt, CAPEX_lev)
    return CAPEX_lev


def cepci_adjustment(c, cepci_ref, debug=False):
    """Escalate overnight costs from ``cepci_ref`` to ``CEPCI_target``; unity if ``ADJUST_CEPCI`` is False."""
    if not c["ADJUST_CEPCI"]:
        adj = 1.0
    else:
        adj = c["CEPCI_target"] / cepci_ref
    if debug:
        print("cepci_adjustment", cepci_ref, "->", adj)
    return adj

def plan_CCS(plant, c, x, l):

    # Burn fuel and capture/condition CO2
    annual_CO2 = plant["Total"] # [ktCO2/yr]
    FLH = plant["FLH"] # [h/yr]
    mCO2 = annual_CO2 * 1000 / FLH # [tCO2/h]
    mCO2 = mCO2 * 1000 / 3600 # [kgCO2/s]

    mCO2_captured = mCO2 * c["capture_rate"] # [kgCO2/s]
    annual_CO2 = annual_CO2 * c["capture_rate"] # [ktCO2/yr]
    Qreb = mCO2_captured * x["q_reb"]                       # [MW]
    Pcapture = x["p_capture"] * mCO2_captured/1000*3600     # [MW] 
    Pcondition = x["p_condition"] * mCO2_captured           # [MW]  

    # Penalize CHP power; heat: steam DH reduced by reboiler, Qfgc unchanged
    Qsteam = plant["Qwaste"]
    P_old = plant["P"]
    Qdh_steam_old = plant["Qdh"]
    Q_heat_target = Qdh_steam_old + plant["Qfgc"]  # [MWth] baseline DH + FGC obligation

    P = P_old * (1 - Qreb / Qsteam)  # [MWel] live steam for reboiler
    P = P - Pcapture - Pcondition  # [MWel]

    Q_delivered = Qdh_steam_old * (1 - Qreb / Qsteam) + plant["Qfgc"]  # [MWth] Qfgc unaffected by reboiler
    Qrec_hex = x["q_hex"] * Qreb  # [MWth]
    Qavailable = Qrec_hex  # [MWth] capture heat recovery
    Qdiff = Q_heat_target - (Q_delivered + Qavailable)  # [MWth]
    Whp = 0
    if Qdiff < 0:
        Q_delivered = Q_heat_target  # surplus vented; baseline obligation met
    elif Qdiff > 0:
        Whp = Qdiff / x["COP"]  # [MWel] may exceed P (grid purchase)
        Q_delivered = Q_delivered + Qavailable + Whp * x["COP"]  # restore to Q_heat_target
        P -= Whp  # negative P means grid power needed
    Ppenalty = (P_old - P) * FLH  # [MWh/yr]
    Qpenalty = (Q_heat_target - Q_delivered) * FLH  # [MWh/yr]
    Pnew_capacity = P_old - P  # [MWel] rated new electrical equipment

    # Estimate CAPEX and on-site OPEX
    CEPCI_adjustment = cepci_adjustment(c, c["CEPCI_reference"])
    CEPCI_capture_adjustment = cepci_adjustment(c, c["CEPCI_capture_reference"])
    CAPEX_capture = (
        c["capex_ref_capture_keur"] * (annual_CO2 / c["capacity_ref_capture_kt_per_yr"]) ** x["k"] * CEPCI_capture_adjustment
    )  # [kEUR]
    CAPEX_capture_lev = levelize_kEUR(CAPEX_capture, annual_CO2, x) # [EUR/tCO2]
    CEPCI_HP_adjustment = cepci_adjustment(c, c["CEPCI_HP_reference"])
    CAPEX_HP = x["capex_ref_hp_keur_per_mwth"] * Whp * x["COP"] * CEPCI_HP_adjustment  # [kEUR] 0.86 MEUR/MWth ref, CEPCI 2022→scenario
    CAPEX_HP_lev = levelize_kEUR(CAPEX_HP, annual_CO2, x)

    OPEX_fix = (CAPEX_capture + CAPEX_HP) * x["opex_fix"] / annual_CO2  # [EUR/tCO2]
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
    CAPEX_loading_kEUR = 0.0
    CAPEX_train_kEUR = 0.0

    def _mode(val):
        return pd.notna(val) and str(val) != "None"

    def _loading_capex_and_lev():
        capex = c["capex_ref_loading_sek"] * c["SEK_to_EUR"] / 1000 * (annual_CO2 / c["capacity_ref_loading_kt_per_yr"]) ** x["k"] * CEPCI_adjustment  # [kEUR]
        return capex, levelize_kEUR(capex, annual_CO2, x)

    if _mode(plant.get('Truck_distance')):
        CAPEX_loading_kEUR, loading_cost = _loading_capex_and_lev()
        distance = float(plant['Truck_distance']) # [km]
        a1, a2 = 0.15, 5.58 
        UC = a1 + a2 / distance # [€/(t*km)]
        truck_cost = UC * distance # [€/tCO2]

    if _mode(plant.get('Pipeline_distance')):
        distance = float(plant['Pipeline_distance'])/1000 # [km]
        a1, a2, a3, a4 = 0.02, 260, 0.07, -0.61
        UC = a1 + a2 * (distance / 1)**a3 * (annual_CO2*1000/ 1)**a4 # [€/(t*km)]
        pipeline_cost = UC * distance # [€/tCO2]

    if _mode(plant.get('Rail_distance')):
        capex_load, lev_load = _loading_capex_and_lev()
        CAPEX_loading_kEUR += capex_load
        loading_cost += lev_load
        distance = float(plant['Rail_distance'])       # [km]
        cycle_time = (distance / 60 + 5) * 2           # [h] roundtrip (*2) @ 60 km/h + 5h unload
        capacity = 15 * 60 / cycle_time                # [tCO2/h] @15 wagons, 60 t/wagon

        CAPEX_train = x["capex_ref_train_eur"] * CEPCI_adjustment            # [EUR]
        CAPEX_train_kEUR = CAPEX_train / 1000.0
        CAPEXlev_train = levelize_kEUR(CAPEX_train_kEUR, annual_CO2, x) # [EUR/tCO2]
        OPEX_train = x["opex_fix"]*CAPEX_train + 0.0269*(capacity*(distance*2*365)) # [EUR/yr] 1 roundtrip per day is more than enough!
        OPEX_train = OPEX_train / (annual_CO2*1000)                                # [EUR/tCO2]

        rail_cost = CAPEXlev_train + OPEX_train

    if _mode(plant.get('Oygarden_distance')):
        if x["storage"] == "oygarden":
            distance = float(plant['Oygarden_distance'])
        elif x["storage"] == "kalundborg":
            distance = float(plant['Kalundborg_distance'])
        df_ship = c["shipping_costs"].sort_values('distance')
        shipping_cost = float(np.interp(distance, df_ship['distance'].values, df_ship[x["shipping_case"]].values))

    transport_factor = x["transport_cost_factor"]  # [-]
    loading_cost *= transport_factor
    truck_cost *= transport_factor
    pipeline_cost *= transport_factor
    rail_cost *= transport_factor
    shipping_cost *= transport_factor
    transport_cost = loading_cost + truck_cost + pipeline_cost + rail_cost + shipping_cost  # [EUR/tCO2]
    cost_CCS = CAPEX_capture_lev + CAPEX_HP_lev + OPEX + transport_cost + x["storage_cost"]  # [EUR/tCO2]

    fossil = plant["Fossil"] / plant["Total"]                       # [tfossil/t] 
    biogenic = 1 - fossil                                           # [tbiogenic/t] 
    FCCS = annual_CO2 * fossil                                # [ktCO2/yr]
    BECCS = annual_CO2 * biogenic                             # [ktCO2/yr]
    if x["CRC"] < x["ETS"]:
        x["CRC"] = x["ETS"] # In such cases (~50%), CRCs are assumed integrated into the ETS
    strike_price = cost_CCS - biogenic * x["CRC"]                        # [EUR/tCO2] relative to a fossil ETS reference price

    opex_fix_eur_yr = OPEX_fix * annual_CO2 * 1000.0
    opex_makeup_eur_yr = OPEX_makeup * annual_CO2 * 1000.0
    opex_energy_eur_yr = OPEX_energy * annual_CO2 * 1000.0

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
        "annual_co2_kt_yr": annual_CO2,
        "capex_overnight_kEUR": {
            "capture": CAPEX_capture,
            "hp": CAPEX_HP,
            "loading": CAPEX_loading_kEUR,
            "train": CAPEX_train_kEUR,
        },
        "opex_annual_eur_yr": {
            "fix": opex_fix_eur_yr,
            "makeup": opex_makeup_eur_yr,
            "energy": opex_energy_eur_yr,
            "total": (OPEX * annual_CO2 * 1000.0),
        },
        "energy": {
            "annual_co2_kt_yr": annual_CO2,
            "p_chp_export_before_mwel": float(P_old),
            "p_chp_export_after_mwel": float(P),
            "p_ccs_penalty_mwel": float(P_old - P),
            "q_heat_target_mwth": float(Q_heat_target),
            "q_heat_delivered_mwth": float(Q_delivered),
            "qdh_mwth": float(Qdh_steam_old),
            "qfgc_mwth": float(plant["Qfgc"]),
            "qlhv_mwth": float(plant["Qlhv"]),
            "q_fuel_plot_mwth": float(plant["Qlhv"]) + float(plant["Qfgc"]),
            "q_boiler_loss_mwth": (1.0 - c["eta_boiler"]) * float(plant["Qlhv"]),
            "qreb_mwth": float(Qreb),
            "qrec_hex_mwth": float(Qrec_hex),
            "pcapture_mwel": float(Pcapture),
            "pcondition_mwel": float(Pcondition),
            "whp_mwel": float(Whp),
            "qhp_out_mwth": float(Whp) * x["COP"],
        },
    }
    return strike_price, FCCS, BECCS, Ppenalty, Pnew_capacity, Qpenalty, cost_details

def plan_CCU(plant, c, x, l):
    
    # Burn fuel and capture/condition CO2
    annual_CO2 = plant["Total"] # [ktCO2/yr]
    FLH = plant["FLH"] # [h/yr]
    mCO2 = annual_CO2 * 1000 / FLH # [tCO2/h]
    mCO2 = mCO2 * 1000 / 3600 # [kgCO2/s]

    mCO2_captured = mCO2 * c["capture_rate"] # [kgCO2/s]
    annual_CO2 = annual_CO2 * c["capture_rate"] # [ktCO2/yr]
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
        nCO2, T1=40+273.15, P1=1.0, thermo_props=c["thermo_props"],
        gas='CO2', n_stages=3, pr=3.8, Tdiff=30, n_is=c["eta_is"])

    # Compress H2: 2 stages @ pr=1.7, from 75°C / 20 bar [Jacobsson & Palmgren, 2025]
    Wcomp_H2, Qcool_H2, P_H2, T_H2 = compression_energy(
        nH2, T1=75+273.15, P1=20, thermo_props=c["thermo_props"],
        gas='H2', n_stages=2, pr=1.7, Tdiff=60, n_is=c["eta_is"])

    wcomp_co2_mwel = float(sum(Wcomp_CO2))  # [MWel]
    wcomp_h2_mwel = float(sum(Wcomp_H2))  # [MWel]
    wcomp_recycle_mwel = (wcomp_co2_mwel + wcomp_h2_mwel) * c["recycle_ratio"]  # [MWel]

    # Produce methanol (check Danish Agency Agency for method - we treat the whole synthesis plant as a single unit):
    nCH3OH = nCO2 # [kmolCH3OH/s] produced methanol
    mCH3OH = nCH3OH * 32 # [kgCH3OH/s]
    m_methanol = mCH3OH * 3600 * 24 /1000 # [tCH3OH/day]
    LHV_CH3OH = c["LHV_CH3OH"] # [MJ/kg]
    QCH3OH = mCH3OH * LHV_CH3OH # [MWth]
    Qmethanol = QCH3OH * FLH     # [MWh/yr]

    # Penalize CHP power; heat: steam DH reduced by reboiler, Qfgc unchanged
    Qsteam = plant["Qwaste"]
    P_old = plant["P"]
    Qdh_steam_old = plant["Qdh"]
    Q_heat_target = Qdh_steam_old + plant["Qfgc"]  # [MWth] baseline DH + FGC obligation

    P = P_old * (1 - Qreb / Qsteam)  # [MWel]
    P = P - Pcapture - PH2 - wcomp_recycle_mwel  # [MWel]

    Q_delivered = Qdh_steam_old * (1 - Qreb / Qsteam) + plant["Qfgc"]  # [MWth] Qfgc unaffected by reboiler
    Q_delivered = Q_delivered + sum(Qcool_CO2) + sum(Qcool_H2)  # [MWth] compressor cooling to network
    Qrec_hex = x["q_hex"] * Qreb  # [MWth]
    # Qrec_elec = x["q_electrolyzer"] * PH2  # [MWth]
    Qloss_elec = PH2 - QH2 # [MWth] electrolyzer heat loss
    Qavailable = Qrec_hex + Qloss_elec * x["heat_optimism"]  # [MWth] assumed "free" heat exchange
    Qdiff = Q_heat_target - (Q_delivered + Qavailable)  # [MWth]
    Whp = 0
    if Qdiff < 0:
        Q_delivered = Q_heat_target  # surplus vented; baseline obligation met
    elif Qdiff > 0:
        Whp = Qdiff / x["COP"]  # [MWel] may exceed P (grid purchase)
        Q_delivered = Q_delivered + Qavailable + Whp * x["COP"]  # restore to Q_heat_target
        P -= Whp  # negative P means grid power needed
    Ppenalty = (P_old - P) * FLH  # [MWh/yr] site electricity (incl. recycle compression)
    Qpenalty = (Q_heat_target - Q_delivered) * FLH  # [MWh/yr]
    Pnew_capacity = P_old - P  # [MWel] rated new electrical equipment

    # Estimate CAPEX and OPEX
    annual_methanol_kt = annual_CO2 * 32.0 / 44.0  # [kt methanol/a] from captured CO2 stoichiometry
    CEPCI_adjustment = cepci_adjustment(c, c["CEPCI_reference"])
    CEPCI_capture_adjustment = cepci_adjustment(c, c["CEPCI_capture_reference"])
    CAPEX_capture = (
        c["capex_ref_capture_keur"] * (annual_CO2 / c["capacity_ref_capture_kt_per_yr"]) ** x["k"] * CEPCI_capture_adjustment
    )  # [kEUR]
    CAPEX_capture_lev = levelize_kEUR(CAPEX_capture, annual_methanol_kt, x)  # [EUR/t methanol]
    CEPCI_HP_adjustment = cepci_adjustment(c, c["CEPCI_HP_reference"])
    CAPEX_HP = x["capex_ref_hp_keur_per_mwth"] * Whp * x["COP"] * CEPCI_HP_adjustment  # [kEUR] 0.86 MEUR/MWth ref, CEPCI 2022→scenario
    CAPEX_HP_lev = levelize_kEUR(CAPEX_HP, annual_methanol_kt, x)  # [EUR/t methanol]

    CAPEX_comp_CO2 = compression_capex_eur(Wcomp_CO2, c["compression_costs"]) / 1000 * CEPCI_adjustment  # [kEUR]
    CAPEX_comp_CO2_lev = levelize_kEUR(CAPEX_comp_CO2, annual_methanol_kt, x)  # [EUR/t methanol]
    CAPEX_comp_H2 = compression_capex_eur(Wcomp_H2, c["compression_costs"]) / 1000 * CEPCI_adjustment  # [kEUR]
    CAPEX_comp_H2_lev = levelize_kEUR(CAPEX_comp_H2, annual_methanol_kt, x)  # [EUR/t methanol]
    CAPEX_H2 = x["capex_ref_h2_keur_per_mwel"] * PH2 * CEPCI_adjustment  # [kEUR]
    CAPEX_H2_lev = levelize_kEUR(CAPEX_H2, annual_methanol_kt, x)  # [EUR/t methanol]
    CAPEX_synthesis = x["capex_ref_synthesis_meur"] * m_methanol ** (-0.315) * 1000 * CEPCI_adjustment  # [kEUR]
    CAPEX_synthesis_lev = levelize_kEUR(CAPEX_synthesis, annual_methanol_kt, x)  # [EUR/t methanol]

    CAPEX_total = CAPEX_capture + CAPEX_HP + CAPEX_comp_CO2 + CAPEX_comp_H2 + CAPEX_H2 + CAPEX_synthesis # [kEUR]
    CAPEX_total_lev = CAPEX_capture_lev + CAPEX_HP_lev + CAPEX_comp_CO2_lev + CAPEX_comp_H2_lev + CAPEX_H2_lev + CAPEX_synthesis_lev  # [EUR/t methanol]

    OPEX_fix = (CAPEX_total * x["opex_fix"]) / annual_methanol_kt  # [EUR/t methanol]
    OPEX_makeup = x["camine"] * c["SEK_to_EUR"] * 44.0 / 32.0  # [EUR/t methanol] from SEK/tCO2 basis
    OPEX_energy = (Ppenalty * x["celc"] + Qpenalty * x["celc"] * x["cheat"]) / (annual_methanol_kt * 1000)  # [EUR/t methanol]
    OPEX = OPEX_fix + OPEX_makeup + OPEX_energy  # [EUR/t methanol]

    methanol_cost = CAPEX_total_lev + OPEX  # [EUR/t methanol]
    strike_price = methanol_cost  # [EUR/t methanol]

    fossil = plant["Fossil"] / plant["Total"]                       # [tfossil/t] 
    biogenic = 1 - fossil                                           # [tbiogenic/t] 
    FCCU = annual_CO2 * fossil                                # [ktCO2/yr]
    BCCU = annual_CO2 * biogenic                              # [ktCO2/yr]

    opex_total_eur_yr = OPEX * annual_methanol_kt * 1000.0  # [EUR/yr]
    opex_fix_eur_yr = OPEX_fix * annual_methanol_kt * 1000.0  # [EUR/yr]
    opex_makeup_eur_yr = OPEX_makeup * annual_methanol_kt * 1000.0  # [EUR/yr]
    opex_energy_eur_yr = OPEX_energy * annual_methanol_kt * 1000.0  # [EUR/yr]

    cost_details = {
        "cost_methanol": methanol_cost,
        "cost_CCU": CAPEX_total_lev + OPEX,  # [EUR/t methanol]
        "CAPEX_capture_lev": CAPEX_capture_lev,
        "CAPEX_HP_lev": CAPEX_HP_lev,
        "CAPEX_comp_CO2_lev": CAPEX_comp_CO2_lev,
        "CAPEX_comp_H2_lev": CAPEX_comp_H2_lev,
        "CAPEX_H2_lev": CAPEX_H2_lev,
        "CAPEX_synthesis_lev": CAPEX_synthesis_lev,
        "OPEX_makeup": OPEX_makeup,
        "OPEX_energy": OPEX_energy,
        "OPEX_fix": OPEX_fix,
        "recycle_ratio": c["recycle_ratio"],
        "annual_methanol_t_yr": annual_methanol_kt,
        "capex_overnight_kEUR": {
            "capture": CAPEX_capture,
            "hp": CAPEX_HP,
            "comp_co2": CAPEX_comp_CO2,
            "comp_h2": CAPEX_comp_H2,
            "h2": CAPEX_H2,
            "synthesis": CAPEX_synthesis,
        },
        "opex_annual_eur_yr": {
            "fix": opex_fix_eur_yr,
            "makeup": opex_makeup_eur_yr,
            "energy": opex_energy_eur_yr,
            "total": opex_total_eur_yr,
        },
        "energy": {
            "annual_methanol_t_yr": annual_methanol_kt,
            "p_chp_export_before_mwel": float(P_old),
            "p_chp_export_after_mwel": float(P),
            "p_ccu_penalty_mwel": float(P_old - P),
            "q_heat_target_mwth": float(Q_heat_target),
            "q_heat_delivered_mwth": float(Q_delivered),
            "qdh_mwth": float(Qdh_steam_old),
            "qfgc_mwth": float(plant["Qfgc"]),
            "qlhv_mwth": float(plant["Qlhv"]),
            "q_fuel_plot_mwth": float(plant["Qlhv"]) + float(plant["Qfgc"]),
            "q_boiler_loss_mwth": (1.0 - c["eta_boiler"]) * float(plant["Qlhv"]),
            "qreb_mwth": float(Qreb),
            "pcapture_mwel": float(Pcapture),
            "ph2_mwel": float(PH2),
            "wcomp_co2_mwel": wcomp_co2_mwel,
            "wcomp_h2_mwel": wcomp_h2_mwel,
            "wcomp_recycle_mwel": wcomp_recycle_mwel,
            "whp_mwel": float(Whp),
            "qhp_out_mwth": float(Whp) * x["COP"],
            "qmethanol_out_mwth": float(QCH3OH),
            "qrec_available_mwth": float(Qavailable),
        },
    }
    return strike_price, FCCU, BCCU, Qmethanol, Ppenalty, Pnew_capacity, Qpenalty, cost_details


@dataclass
class FuelAggregate:
    """Summed extensive properties for plants in a cumulative replacement set."""

    n_c_pl: float  # [kmolC_pl/yr]
    n_h_pl: float  # [kmol/yr]
    n_o_pl: float  # [kmol/yr]
    n_c_bio: float  # [kmol/yr]
    n_h_bio: float  # [kmol/yr]
    n_o_bio: float  # [kmol/yr]
    m_pl: float  # [kg/yr]
    m_bio: float  # [kg/yr]
    m_h2o: float  # [kg/yr]
    m_ash: float  # [kg/yr]
    m_tot: float  # [kg/yr]
    e_input_mj_yr: float  # [MJ/yr]
    lhv_pl: float # [MJ/kg_da]
    lhv_bio: float # [MJ/kg_da]
    # lhv_tot_blend: float  # [MJ/kg_wet]


def aggregate_fuel(plants_slice: pd.DataFrame) -> FuelAggregate:
    """Sum extensive columns over ``plants_slice``; blended LHV_tot from Σ(LHV_tot·m_tot)/Σm_tot."""
    n_c_pl = float(plants_slice["nC_pl"].sum())  # [kmolC_pl/yr]
    n_h_pl = float(plants_slice["nH_pl"].sum())  # [kmol/yr]
    n_o_pl = float(plants_slice["nO_pl"].sum())  # [kmol/yr]
    n_c_bio = float(plants_slice["nC_bio"].sum())  # [kmol/yr]
    n_h_bio = float(plants_slice["nH_bio"].sum())  # [kmol/yr]
    n_o_bio = float(plants_slice["nO_bio"].sum())  # [kmol/yr]
    m_pl = float(plants_slice["m_pl"].sum())  # [kg/yr]
    m_bio = float(plants_slice["m_bio"].sum())  # [kg/yr]
    m_h2o = float(plants_slice["m_h2o"].sum())  # [kg/yr]
    m_ash = float(plants_slice["m_ash"].sum())  # [kg/yr]
    m_tot = float(plants_slice["m_tot"].sum())  # [kg/yr]
    e_input = float(
        (plants_slice["LHV_pl"] * plants_slice["m_pl"] + plants_slice["LHV_bio"] * plants_slice["m_bio"]).sum()
    )  # [MJ/yr]
    lhv_num_pl = float((plants_slice["LHV_pl"] * plants_slice["m_pl"]).sum())  # [MJ/yr] = Σ(LHV_tot·m_tot)
    lhv_pl = lhv_num_pl / m_pl # [MJ/kg_da]
    lhv_num_bio = float((plants_slice["LHV_bio"] * plants_slice["m_bio"]).sum())  # [MJ/yr] = Σ(LHV_tot·m_tot)
    lhv_bio = lhv_num_bio / m_bio # [MJ/kg_da]
    # lhv_num = float((plants_slice["LHV_tot"] * plants_slice["m_tot"]).sum())  # [MJ/yr] = Σ(LHV_tot·m_tot)
    # lhv_tot_blend = lhv_num / m_tot if m_tot > 0 else 0.0  # [MJ/kg_wet]
    return FuelAggregate(
        n_c_pl=n_c_pl,
        n_h_pl=n_h_pl,
        n_o_pl=n_o_pl,
        n_c_bio=n_c_bio,
        n_h_bio=n_h_bio,
        n_o_bio=n_o_bio,
        m_pl=m_pl,
        m_bio=m_bio,
        m_h2o=m_h2o,
        m_ash=m_ash,
        m_tot=m_tot,
        e_input_mj_yr=e_input,
        lhv_pl=lhv_pl,
        lhv_bio=lhv_bio,
    )

def plan_gasifier(
    fuel: FuelAggregate,
    c: dict,
    x: dict,
    debug: bool = False,
) -> Dict[str, Any]:
    """Run gasification → synthesis → costs for one aggregated fuel bundle (gasify_all simulate_case_new logic).

    Return dict keys use SI tags in names where helpful: *_mj_yr, *_mwth, *_mwel, *_t_yr, *_meur.
    """
    thermo_props = c["thermo_props"]
    compression_costs_df = c["compression_costs"]
    flh = c["FLH_gasifier"]  # [h/yr]
    frac_gasify = c["gasified_carbon_fraction"]  # [-]
    frac_combustor_pl = c["frac_combustor_pl"]  # [-]
    frac_energy = c["frac_energy"]  # [-]

    n_c_tot = fuel.n_c_pl + fuel.n_c_bio  # [kmolC/yr]
    if n_c_tot <= 0:
        raise ValueError("aggregate carbon moles must be positive")

    h_ratio_pl = fuel.n_h_pl / fuel.n_c_pl  # [kmolH/kmolC]
    o_ratio_pl = fuel.n_o_pl / fuel.n_c_pl  # [kmolO/kmolC]
    h_ratio_bio = fuel.n_h_bio / fuel.n_c_bio  # [kmolH/kmolC]
    o_ratio_bio = fuel.n_o_bio / fuel.n_c_bio  # [kmolO/kmolC]

    n_c_combustor = n_c_tot * (1.0 - frac_gasify)  # [kmolC/yr]
    n_c_combustor_bio = n_c_combustor * (1.0 - frac_combustor_pl)  # [kmolC/yr]
    n_c_combustor_pl = n_c_combustor - n_c_combustor_bio  # [kmolC/yr]

    n_c_gasifier_yr = n_c_tot - n_c_combustor  # [kmolC/yr]
    n_c_gasifier_bio = fuel.n_c_bio - n_c_combustor_bio  # [kmolC/yr]
    n_c_gasifier_pl = fuel.n_c_pl - n_c_combustor_pl  # [kmolC/yr]
    n_h_gasifier_yr = h_ratio_pl * n_c_gasifier_pl + h_ratio_bio * n_c_gasifier_bio  # [kmolH/yr]

    q_combustor_yr = (
        (n_c_combustor_pl * 12.0 + h_ratio_pl * n_c_combustor_pl * 1.0 + o_ratio_pl * n_c_combustor_pl * 16.0) * fuel.lhv_pl
        + (n_c_combustor_bio * 12.0 + h_ratio_bio * n_c_combustor_bio * 1.0 + o_ratio_bio * n_c_combustor_bio * 16.0) * fuel.lhv_bio
    )  # [MJ/yr]
    q_gasifier_yr = (
        (n_c_gasifier_pl * 12.0 + h_ratio_pl * n_c_gasifier_pl * 1.0 + o_ratio_pl * n_c_gasifier_pl * 16.0) * fuel.lhv_pl
        + (n_c_gasifier_bio * 12.0 + h_ratio_bio * n_c_gasifier_bio * 1.0 + o_ratio_bio * n_c_gasifier_bio * 16.0) * fuel.lhv_bio
    )  # [MJ/yr]
    if debug:
        print(
            "plan_gasifier input energy [MJ/yr]",
            q_combustor_yr,
            q_gasifier_yr,
            q_combustor_yr / (q_combustor_yr + q_gasifier_yr),
            fuel.e_input_mj_yr / (q_combustor_yr + q_gasifier_yr),
        )

    lhv_ch3oh = c["LHV_CH3OH"] * 32.0  # [MJ/kmol]
    q_gasifier = q_gasifier_yr / (flh * 3600.0)  # [MWth]
    q_combustor = q_combustor_yr / (flh * 3600.0)  # [MWth]
    q_methanol = q_gasifier * frac_energy  # [MWth]

    n_c_methanol_base = q_methanol / lhv_ch3oh  # [kmolC/s] methanol yield excluding any hydrogenation
    n_c_gasifier = n_c_gasifier_yr / (flh * 3600.0)  # [kmolC/s]
    n_co2_syngas = n_c_gasifier - n_c_methanol_base  # [kmolCO2/s]
    n_co_syngas = n_c_gasifier - n_co2_syngas  # [kmolCO/s]
    n_h2_syngas = 2.0 * n_co_syngas  # [kmolH2/s] before additional hydrogenation, our syngas must have the composition 2H2:1CO to create CH3OH

    n_h2o_gasifier = n_c_gasifier  # [kmolH2O/s] assume 1 mol H2O steam is added per mol CHyOx (like Beiron's initial guess)
    n_h2o_syngas = (n_h_gasifier_yr / (flh * 3600.0)) + 2.0 * n_h2o_gasifier - 2.0 * n_h2_syngas  # [kmolH/s]
    n_h2o_syngas /= 2.0  # [kmolH2O/s]

    if any(x < 0 for x in [n_h2_syngas, n_co_syngas, n_co2_syngas, n_h2o_syngas]):
        raise ValueError(
            f"Negative syngas component detected: "
            f"H2={n_h2_syngas}, CO={n_co_syngas}, CO2={n_co2_syngas}, H2O={n_h2o_syngas}"
        )
    n_syngas = n_h2_syngas + n_co_syngas + n_co2_syngas + n_h2o_syngas  # [kmol/s]
    q_syngas = n_h2_syngas * c["LHV_H2_MJ_PER_KMOL"] + n_co_syngas * c["LHV_CO_MJ_PER_KMOL"]  # [MWth]
    syngas_mix = {
        "H2": n_h2_syngas / n_syngas,
        "CO": n_co_syngas / n_syngas,
        "CO2": n_co2_syngas / n_syngas,
        "H2O": n_h2o_syngas / n_syngas,
    }

    wcomp_mix, _, _, _ = compression_energy(
        n_syngas,
        T1=40 + 273.15,
        P1=1.0,
        thermo_props=thermo_props,
        gas=syngas_mix,
        n_stages=3,
        pr=3.8,
        Tdiff=30,
        n_is=c["eta_is"],
        debug=debug,
    )
    n_h2_electrolyzer = n_co2_syngas * 3.0  # [kmolH2/s]
    wcomp_h2, _, _, _ = compression_energy(
        n_h2_electrolyzer,
        T1=75 + 273.15,
        P1=20,
        thermo_props=thermo_props,
        gas="H2",
        n_stages=2,
        pr=1.7,
        Tdiff=60,
        n_is=c["eta_is"],
        debug=debug,
    )
    w_mix_sum = float(sum(wcomp_mix))  # [MWel]
    w_h2_sum = float(sum(wcomp_h2))  # [MWel]
    w_recycle = (w_h2_sum + w_mix_sum) * c["recycle_ratio"]  # [MWel]

    n_c_methanol_total = n_c_methanol_base + n_co2_syngas  # [kmol/s]
    q_methanol_final = n_c_methanol_total * lhv_ch3oh  # [MWth]
    qh2_mwth = n_h2_electrolyzer * c["LHV_H2_MJ_PER_KMOL"]  # [MWth]
    ph2_mwel = qh2_mwth / x["eta_electrolyzer"]  # [MWel]

    w_power_mwel = ph2_mwel + w_recycle  # [MWel] used in eta_total denominator
    power_input_mwel = ph2_mwel + w_recycle  # [MWel] full hub electrical demand
    eta_total = q_methanol_final / (q_combustor + q_gasifier + w_power_mwel)  # [-]

    q_loss_gasifier = q_gasifier - q_syngas  # [MWth]
    q_loss_combustor = q_combustor  # [MWth]
    q_loss_electrolyzer = ph2_mwel - qh2_mwth  # [MWth]
    q_loss_synthesis = q_syngas + qh2_mwth + w_recycle - q_methanol_final  # [MWth]
    q_loss_total = q_loss_gasifier + q_loss_combustor + q_loss_electrolyzer + q_loss_synthesis  # [MWth]
    if debug:
        print(
            "plan_gasifier losses [MWth]",
            q_loss_total,
            (q_combustor + q_gasifier + w_power_mwel) - q_methanol_final,
        )

    m_ch3oh_kg_yr = n_c_methanol_total * 32.0 * flh * 3600.0  # [kg/yr]
    annual_methanol_t_yr = m_ch3oh_kg_yr / 1000.0  # [t/a]
    capacity_sorting_t_yr = fuel.m_tot / 1000.0  # [t waste/a]

    capex_sorting_at_ref = c["capex_ref_sorting_meur"] * (
        capacity_sorting_t_yr / c["capacity_ref_sorting_t_waste_per_yr"]
    ) ** x["k"]  # [MEUR]
    capex_sorting_meur = capex_sorting_at_ref * cepci_adjustment(c, c["CEPCI_sorting_ref"])  # [MEUR]
    opex_fix_sorting_meur = capex_sorting_meur * x["opex_fix"]  # [MEUR/a]
    opex_var_sorting_eur_per_t = (
        x["opex_var_sorting_sek_per_t_waste"] * c["SEK_to_EUR"] * cepci_adjustment(c, c["CEPCI_opex_sorting_ref"])
    )
    opex_var_sorting_meur = opex_var_sorting_eur_per_t * capacity_sorting_t_yr * 1e-6  # [MEUR/a]
    opex_sorting_meur_a = opex_fix_sorting_meur + opex_var_sorting_meur  # [MEUR/a]

    capex_gasif_at_ref = c["capex_ref_gasification_meur"] * (
        annual_methanol_t_yr / c["capacity_ref_gasification_t_methanol_per_yr"]
    ) ** x["k"]  # [MEUR] incl. syngas compression (ECOPLANTA)
    capex_gasif_meur = capex_gasif_at_ref * cepci_adjustment(c, c["CEPCI_gasification_ref"])  # [MEUR]
    opex_fix_gasif_meur = capex_gasif_meur * x["opex_fix"]  # [MEUR/a]
    opex_var_gasif_eur_per_mwh = (
        x["opex_var_gasification_eur_per_mwh_fuel"] * cepci_adjustment(c, c["CEPCI_opex_gasification_ref"])
    )
    opex_var_gasif_meur = opex_var_gasif_eur_per_mwh * fuel.e_input_mj_yr / 3600.0 * 1e-6  # [MEUR/a]
    opex_gasif_gross_meur_a = opex_fix_gasif_meur + opex_var_gasif_meur  # [MEUR/a] before heat credit
    opex_heat_recovery_meur_a = (q_loss_combustor + q_loss_electrolyzer * x["heat_optimism"]) * flh * x["celc"] * x["cheat"] * 1e-6  # [MEUR/a] combustor waste-heat credit
    opex_gasif_meur_a = opex_gasif_gross_meur_a - opex_heat_recovery_meur_a  # [MEUR/a] net gasification OPEX

    w_comp_mwel = w_recycle  # [MWel]
    cepci_h2_adj = cepci_adjustment(c, c["CEPCI_h2_ref"])  # [-]
    capex_compression_meur = (
        compression_capex_eur(wcomp_h2, compression_costs_df, debug=debug) * 1e-6 * cepci_h2_adj
    )  # [MEUR] H2 compressor only
    opex_fix_compression_meur = capex_compression_meur * x["opex_fix"]  # [MEUR/a]
    opex_energy_compression_meur = w_comp_mwel * flh * x["celc"] * 1e-6  # [MEUR/a]
    opex_compression_meur_a = opex_fix_compression_meur + opex_energy_compression_meur  # [MEUR/a]

    capex_electrolyzer_meur = x["capex_ref_h2_keur_per_mwel"] * ph2_mwel * 1e-3 * cepci_h2_adj  # [MEUR]
    opex_fix_electrolyzer_meur = capex_electrolyzer_meur * x["opex_fix"]  # [MEUR/a]
    opex_energy_electrolyzer_meur = ph2_mwel * flh * x["celc"] * 1e-6  # [MEUR/a]
    opex_electrolyzer_meur_a = opex_fix_electrolyzer_meur + opex_energy_electrolyzer_meur  # [MEUR/a]

    if debug:
        print(
            "plan_gasifier costs [MEUR]",
            capex_sorting_meur,
            capex_gasif_meur,
            capex_compression_meur,
            capex_electrolyzer_meur,
            "heat_recovery_opex",
            opex_heat_recovery_meur_a,
        )

    return {
        "e_input_mj_yr": fuel.e_input_mj_yr,  # [MJ/yr]
        "c_gasified_t_yr": n_c_gasifier_yr * 12.0 / 1000.0,  # [t/a]
        "c_combusted_t_yr": n_c_combustor * 12.0 / 1000.0,  # [t/a]
        "frac_bio_combustor": n_c_combustor_bio / n_c_combustor,  # [-]
        "frac_bio_gasifier": n_c_gasifier_bio / n_c_gasifier_yr,  # [-]

        "q_combustor_mwth": q_combustor,  # [MWth]
        "q_gasifier_mwth": q_gasifier,  # [MWth]
        "q_fuel_mwth": q_combustor + q_gasifier,  # [MWth]
        "q_syngas_mwth": q_syngas,  # [MWth]
        "q_hydrogen_mwth": qh2_mwth,  # [MWth]
        "w_hydrogen_mwel": ph2_mwel,  # [MWel]
        "q_methanol_mwth": q_methanol_final,  # [MWth]
        "w_comp_syngas_mwel": w_mix_sum,  # [MWel]
        "w_comp_h2_mwel": w_h2_sum,  # [MWel]
        "w_recycle_mwel": w_recycle,  # [MWel]
        "w_power_mwel": w_power_mwel,  # [MWel]
        "eta_total": eta_total,  # [-]
        "q_loss_combustor_mwth": q_loss_combustor,  # [MWth]
        "q_loss_gasifier_mwth": q_loss_gasifier,  # [MWth]
        "q_loss_electrolyzer_mwth": q_loss_electrolyzer,  # [MWth]
        "q_loss_synthesis_mwth": q_loss_synthesis,  # [MWth]
        "q_loss_total_mwth": q_loss_total,  # [MWth]

        "annual_methanol_t_yr": annual_methanol_t_yr,  # [t/a]
        "capex_sorting_meur": capex_sorting_meur,  # [MEUR]
        "opex_sorting_meur_a": opex_sorting_meur_a,  # [MEUR/a]
        "capex_gasif_meur": capex_gasif_meur,  # [MEUR]
        "opex_gasif_gross_meur_a": opex_gasif_gross_meur_a,  # [MEUR/a]
        "opex_heat_recovery_meur_a": opex_heat_recovery_meur_a,  # [MEUR/a] credit (combustor losses)
        "opex_gasif_meur_a": opex_gasif_meur_a,  # [MEUR/a] net after heat credit
        "capex_compression_meur": capex_compression_meur,  # [MEUR]
        "opex_compression_meur_a": opex_compression_meur_a,  # [MEUR/a]
        "capex_electrolyzer_meur": capex_electrolyzer_meur,  # [MEUR]
        "opex_electrolyzer_meur_a": opex_electrolyzer_meur_a,  # [MEUR/a]

        # Legacy aliases for KPI / plotting
        "input_fuel_mwth": q_combustor + q_gasifier,  # [MWth]
        "hydrogen_produced_mwth": qh2_mwth,  # [MWth]
        "methanol_out_mwth": q_methanol_final,  # [MWth]
        "ph2_mwel": ph2_mwel,  # [MWel]
        "wcomp_mix_mwel": w_mix_sum,  # [MWel]
        "wcomp_h2_mwel": w_h2_sum,  # [MWel]
        "wcomp_recycle_mwel": w_recycle,  # [MWel]
        "power_input_mwel": power_input_mwel,  # [MWel]
        "p_energy_mwh_yr": power_input_mwel * flh,  # [MWh/yr]
    }

def plan_replacements(plants_slice: pd.DataFrame, c: dict, x: dict, debug: bool = False) -> Dict[str, Any]:
    """Sum per-plant truck haul and heat-pump replacement costs over ``plants_slice``."""
    cepci_hp_adj = cepci_adjustment(c, c["CEPCI_HP_reference"])  # [-]
    a1_truck, a2_truck = 0.15, 5.58  # [EUR/(t·km), EUR/t] same as plan_CCS CO₂ truck (Ouvrey et al., 2024)

    opex_truck_eur_yr = 0.0
    capex_hp_meur = 0.0
    opex_hp_meur = 0.0
    opex_fix_hp_meur = 0.0
    opex_var_hp_meur = 0.0
    p_hp_capacity_mwel = 0.0
    p_site_deficit_mwel = 0.0
    p_energy_mwh_yr = 0.0
    per_plant: List[Dict[str, float]] = []

    for _, plant_row in plants_slice.iterrows():
        waste_mass_t_per_yr = float(plant_row["m_tot"]) / 1000.0  # [t/a]
        distance_km = float(plant_row["gasification_distance_km"])  # [km]
        flh_plant = float(plant_row["FLH"])  # [h/yr]

        uc_truck = a1_truck + a2_truck / distance_km  # [EUR/(t·km)]
        truck_eur_per_t = uc_truck * distance_km * x["transport_cost_factor"]  # [EUR/t waste]
        plant_opex_truck = waste_mass_t_per_yr * truck_eur_per_t  # [EUR/yr]

        q_lost_mwth = float(plant_row["Qdh"]) + float(plant_row["Qfgc"])  # [MWth]
        p_lost_mwel = float(plant_row["P"])  # [MWel]
        whp_mwel = q_lost_mwth / x["COP"]  # [MWel]
        plant_capex_hp = x["capex_ref_hp_keur_per_mwth"] * q_lost_mwth / 1000.0 * cepci_hp_adj  # [MEUR] kEUR/MWth → MEUR
        plant_opex_fix_hp = plant_capex_hp * x["opex_fix"]  # [MEUR/a]
        plant_opex_var_hp = (whp_mwel + p_lost_mwel) * flh_plant * x["celc"] * 1e-6  # [MEUR/a]
        plant_opex_hp = plant_opex_fix_hp + plant_opex_var_hp  # [MEUR/a]

        if debug:
            print(
                plant_row["Name"],
                f"truck={truck_eur_per_t:.2f} EUR/t",
                f"HP CAPEX={plant_capex_hp:.2f} MEUR",
            )

        fuel_mwth = float(plant_row["Qlhv"])  # [MWth]
        heat_out_mwth = q_lost_mwth  # [MWth]
        power_out_mwel = p_lost_mwel  # [MWel]
        part = {
            "plant_name": plant_row["Name"],
            "opex_truck_eur_yr": plant_opex_truck,
            "capex_hp_meur": plant_capex_hp,
            "opex_hp_meur": plant_opex_hp,
            "opex_fix_hp_meur": plant_opex_fix_hp,
            "opex_var_hp_meur": plant_opex_var_hp,
            "fuel_mwth": fuel_mwth,
            "heat_out_mwth": heat_out_mwth,
            "power_out_mwel": power_out_mwel,
            "heat_deficit_mwth": heat_out_mwth,
            "hp_elec_mwel": whp_mwel,
            "power_deficit_mwel": whp_mwel + power_out_mwel,
            "flh_h": flh_plant,
            "p_energy_mwh_yr": (whp_mwel + power_out_mwel) * flh_plant,
        }
        per_plant.append(part)
        p_hp_capacity_mwel += whp_mwel
        p_site_deficit_mwel += whp_mwel + power_out_mwel
        p_energy_mwh_yr += part["p_energy_mwh_yr"]
        opex_truck_eur_yr += plant_opex_truck
        capex_hp_meur += plant_capex_hp
        opex_hp_meur += plant_opex_hp
        opex_fix_hp_meur += plant_opex_fix_hp
        opex_var_hp_meur += plant_opex_var_hp

    return {
        "opex_truck_eur_yr": opex_truck_eur_yr,  # [EUR/yr]
        "capex_hp_meur": capex_hp_meur,  # [MEUR]
        "opex_hp_meur": opex_hp_meur,  # [MEUR/a]
        "opex_fix_hp_meur": opex_fix_hp_meur,  # [MEUR/a]
        "opex_var_hp_meur": opex_var_hp_meur,  # [MEUR/a]
        "p_hp_capacity_mwel": p_hp_capacity_mwel,  # [MWel]
        "p_site_deficit_mwel": p_site_deficit_mwel,  # [MWel] HP + lost CHP export at sites
        "p_energy_mwh_yr": p_energy_mwh_yr,  # [MWh/yr]
        "per_plant": per_plant,
    }


def levelize_meur_capex_to_eur_per_t_methanol(
    capex_meur: float,
    annual_methanol_t_per_yr: float,
    x: dict,
    debug: bool = False,
) -> float:
    """Capital recovery on overnight CAPEX [MEUR] → [EUR/t methanol]."""
    dr, lifetime = x["dr"], x["t"]
    crf = dr * (1 + dr) ** lifetime / ((1 + dr) ** lifetime - 1)
    capex_annual_eur = capex_meur * crf * 1e6  # [EUR/yr]
    lev = capex_annual_eur / annual_methanol_t_per_yr
    if debug:
        print("levelize_meur_capex_to_eur_per_t_methanol", capex_meur, annual_methanol_t_per_yr, lev)
    return lev


def levelize_all_replacement_costs(
    gasifier: Dict[str, Any],
    replacement: Dict[str, Any],
    x: dict,
    debug: bool = False,
) -> Dict[str, float]:
    """Levelize centralized hub + summed replacement costs on hub methanol [t/a]."""
    annual_methanol_t_per_yr = gasifier["annual_methanol_t_yr"]
    lev = {
        "eur_t_truck_opex": replacement["opex_truck_eur_yr"] / annual_methanol_t_per_yr,
        "eur_t_hp_capex": levelize_meur_capex_to_eur_per_t_methanol(
            replacement["capex_hp_meur"], annual_methanol_t_per_yr, x, debug=debug
        ),
        "eur_t_hp_opex": replacement["opex_hp_meur"] * 1e6 / annual_methanol_t_per_yr,
        "eur_t_sort_capex": levelize_meur_capex_to_eur_per_t_methanol(
            gasifier["capex_sorting_meur"], annual_methanol_t_per_yr, x, debug=debug
        ),
        "eur_t_sort_opex": gasifier["opex_sorting_meur_a"] * 1e6 / annual_methanol_t_per_yr,
        "eur_t_gasif_capex": levelize_meur_capex_to_eur_per_t_methanol(
            gasifier["capex_gasif_meur"], annual_methanol_t_per_yr, x, debug=debug
        ),
        "eur_t_gasif_opex": gasifier["opex_gasif_gross_meur_a"] * 1e6 / annual_methanol_t_per_yr,
        "eur_t_heat_recovery": -gasifier["opex_heat_recovery_meur_a"] * 1e6 / annual_methanol_t_per_yr,
        "eur_t_comp_capex": levelize_meur_capex_to_eur_per_t_methanol(
            gasifier["capex_compression_meur"], annual_methanol_t_per_yr, x, debug=debug
        ),
        "eur_t_comp_opex": gasifier["opex_compression_meur_a"] * 1e6 / annual_methanol_t_per_yr,
        "eur_t_electrolyzer_capex": levelize_meur_capex_to_eur_per_t_methanol(
            gasifier["capex_electrolyzer_meur"], annual_methanol_t_per_yr, x, debug=debug
        ),
        "eur_t_electrolyzer_opex": gasifier["opex_electrolyzer_meur_a"] * 1e6 / annual_methanol_t_per_yr,
    }
    lev["eur_t_total"] = sum(lev.values())
    if debug:
        print("levelize_all_replacement_costs", lev)
    return lev


def _gasifier_central_costs(gasifier: Dict[str, Any]) -> Dict[str, float]:
    """Hub overnight CAPEX and annual OPEX block for plotting."""
    return {
        "capex_sorting_meur": gasifier["capex_sorting_meur"],
        "capex_gasif_meur": gasifier["capex_gasif_meur"],
        "capex_compression_meur": gasifier["capex_compression_meur"],
        "capex_electrolyzer_meur": gasifier["capex_electrolyzer_meur"],
        "opex_sorting_meur_a": gasifier["opex_sorting_meur_a"],
        "opex_gasif_gross_meur_a": gasifier["opex_gasif_gross_meur_a"],
        "opex_heat_recovery_meur_a": gasifier["opex_heat_recovery_meur_a"],
        "opex_gasif_meur_a": gasifier["opex_gasif_meur_a"],
        "opex_compression_meur_a": gasifier["opex_compression_meur_a"],
        "opex_electrolyzer_meur_a": gasifier["opex_electrolyzer_meur_a"],
    }


def plot_replacement_cost_breakdown(
    n_plants: int,
    replacement: Dict[str, Any],
    central_costs: Dict[str, float],
    lev: Dict[str, float],
    annual_methanol_t_per_yr: float,
    out_path: str,
    debug: bool = False,
) -> None:
    """Three-panel cost figure: per-plant replacement, centralized hub, levelized EUR/t MeOH."""
    per_plant = replacement["per_plant"]
    plant_names = [p["plant_name"] for p in per_plant]
    n = len(plant_names)
    x_idx = np.arange(n)
    width = 0.26

    truck_opex_meur_a = [p["opex_truck_eur_yr"] * 1e-6 for p in per_plant]
    hp_opex_meur_a = [p["opex_hp_meur"] for p in per_plant]
    hp_capex_meur = [p["capex_hp_meur"] for p in per_plant]

    fig, (ax_dec, ax_cen, ax_lev) = plt.subplots(3, 1, figsize=(14, 14))

    b1 = ax_dec.bar(
        x_idx - width, truck_opex_meur_a, width, label="Truck OPEX", color="#66CCEE", edgecolor="black", linewidth=0.5
    )
    b2 = ax_dec.bar(
        x_idx, hp_opex_meur_a, width, label="Heat pump OPEX", color="#EE6677", edgecolor="black", linewidth=0.5
    )
    b3 = ax_dec.bar(
        x_idx + width, hp_capex_meur, width, label="Heat pump CAPEX (overnight)", color="#AA3377", edgecolor="black", linewidth=0.5
    )
    ax_dec.set_ylabel("Cost (MEUR/a or MEUR overnight)", fontsize=13)
    ax_dec.set_title("Decentralized replacement costs (per replaced plant)", fontsize=14)
    ax_dec.set_xticks(x_idx)
    ax_dec.set_xticklabels(plant_names, rotation=35, ha="right", fontsize=10)
    ax_dec.legend(loc="upper right", fontsize=10)
    ax_dec.grid(axis="y", alpha=0.35)
    ax_dec.tick_params(axis="y", labelsize=11)

    cen_labels = ["Sorting", "Gasification", "H2 compression", "Electrolyzer"]
    cen_x = np.arange(4)
    cen_w = 0.35
    capex_cen = [
        central_costs["capex_sorting_meur"],
        central_costs["capex_gasif_meur"],
        central_costs["capex_compression_meur"],
        central_costs["capex_electrolyzer_meur"],
    ]
    opex_cen = [
        central_costs["opex_sorting_meur_a"],
        central_costs["opex_gasif_meur_a"],
        central_costs["opex_compression_meur_a"],
        central_costs["opex_electrolyzer_meur_a"],
    ]
    ax_cen.bar(cen_x - cen_w / 2, capex_cen, cen_w, label="CAPEX (overnight MEUR)", color="#4477AA", edgecolor="black", linewidth=0.5)
    ax_cen.bar(cen_x + cen_w / 2, opex_cen, cen_w, label="OPEX (MEUR/a)", color="#228833", edgecolor="black", linewidth=0.5)
    ax_cen.set_ylabel("Cost (MEUR or MEUR/a)", fontsize=13)
    ax_cen.set_title(f"Centralized gasifier hub ({n_plants} plants replaced)", fontsize=14)
    ax_cen.set_xticks(cen_x)
    ax_cen.set_xticklabels(cen_labels, fontsize=12)
    ax_cen.legend(loc="upper right", fontsize=11)
    ax_cen.grid(axis="y", alpha=0.35)
    ax_cen.tick_params(axis="y", labelsize=11)

    lev_labels = [
        "Truck\nOPEX", "HP\nCAPEX", "HP\nOPEX", "Sort\nCAPEX", "Sort\nOPEX",
        "Gasif.\nCAPEX", "Gasif.\nOPEX", "Heat\nrec.", "H2 comp.\nCAPEX", "H2 comp.\nOPEX",
        "El.\nCAPEX", "El.\nOPEX",
    ]
    lev_vals = [
        lev["eur_t_truck_opex"], lev["eur_t_hp_capex"], lev["eur_t_hp_opex"],
        lev["eur_t_sort_capex"], lev["eur_t_sort_opex"], lev["eur_t_gasif_capex"],
        lev["eur_t_gasif_opex"], lev["eur_t_heat_recovery"], lev["eur_t_comp_capex"],
        lev["eur_t_comp_opex"], lev["eur_t_electrolyzer_capex"], lev["eur_t_electrolyzer_opex"],
    ]
    lev_colors = [
        "#66CCEE", "#EE6677", "#EE6677", "#4477AA", "#4477AA",
        "#228833", "#228833", "#1ABC9C", "#CCBB44", "#CCBB44", "#AA3377", "#AA3377",
    ]
    bl_x = range(len(lev_vals))
    ax_lev.bar(bl_x, lev_vals, color=lev_colors, edgecolor="black", linewidth=0.6)
    ax_lev.axhline(
        lev["eur_t_total"], color="gray", linestyle="--", linewidth=1.2,
        label=f"Total = {lev['eur_t_total']:.0f} EUR/t",
    )
    ax_lev.set_xticks(list(bl_x))
    ax_lev.set_xticklabels(lev_labels, fontsize=11)
    ax_lev.set_ylabel("Levelized cost (EUR/t methanol)", fontsize=13)
    ax_lev.set_title(f"Levelized cost breakdown ({annual_methanol_t_per_yr:,.0f} t methanol/a)", fontsize=14)
    ax_lev.legend(loc="upper right", fontsize=11)
    ax_lev.grid(axis="y", alpha=0.35)
    ax_lev.tick_params(axis="y", labelsize=11)

    fig.suptitle(f"Cost breakdown — {n_plants} plant(s) replaced by centralized gasifier", fontsize=15, y=1.01)
    fig.tight_layout()
    d = os.path.dirname(out_path)
    if d:
        os.makedirs(d, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    if debug:
        print("plot_replacement_cost_breakdown saved:", out_path)
    plt.close(fig)


def plot_replacement_energy_breakdown(
    n_plants: int,
    hub_out: Dict[str, Any],
    replacement: Dict[str, Any],
    plants_slice: pd.DataFrame,
    c: dict,
    out_path: str,
    debug: bool = False,
) -> None:
    """Two-panel energy balance (MW; thermal and electrical on one axis)."""
    eta_boiler = c["eta_boiler"]
    c_fuel = "#7B241C"
    c_heat_1 = "#C0392B"
    c_heat_2 = "#E74C3C"
    c_heat_3 = "#F5B7B1"
    c_pwr_1 = "#1A5276"
    c_pwr_2 = "#2E86C1"
    c_pwr_3 = "#85C1E9"
    c_loss_combustor = "#922B21"
    c_loss_gasifier = "#D35400"
    c_loss_electrolyzer = "#7D6608"
    c_loss_synthesis = "#566573"

    per_plant = replacement["per_plant"]
    names = [p["plant_name"] for p in per_plant]
    n = len(names)
    x_idx = np.arange(n)
    w = 0.13

    qfgc_by_name = {str(row["Name"]): float(row["Qfgc"]) for _, row in plants_slice.iterrows()}
    qlhv_by_name = {str(row["Name"]): float(row["Qlhv"]) for _, row in plants_slice.iterrows()}

    fuel_before = []
    boiler_loss_before = []
    for p in per_plant:
        qlhv = qlhv_by_name.get(p["plant_name"], p["fuel_mwth"])
        qfgc = qfgc_by_name.get(p["plant_name"], 0.0)
        fuel_before.append(qlhv + qfgc)
        boiler_loss_before.append((1 - eta_boiler) * qlhv)

    heat_before = [p["heat_out_mwth"] for p in per_plant]
    power_before = [p["power_out_mwel"] for p in per_plant]
    heat_after = [p["heat_deficit_mwth"] for p in per_plant]
    power_after = [p["power_deficit_mwel"] for p in per_plant]

    sum_hp_heat = sum(heat_after)
    sum_hp_elec = sum(p["hp_elec_mwel"] for p in per_plant)
    sum_site_pwr = sum(power_after)

    fig, (ax_dec, ax_sum) = plt.subplots(2, 1, figsize=(14, 11))

    ax_dec.bar(
        x_idx - 2.5 * w, boiler_loss_before, w,
        label=f"Boiler loss (before, {100 * (1 - eta_boiler):.0f}%)",
        color=c_heat_1, edgecolor="black", linewidth=0.4,
    )
    ax_dec.bar(x_idx - 1.5 * w, fuel_before, w, label="Fuel in (before, HHV)", color=c_fuel, edgecolor="black", linewidth=0.4)
    ax_dec.bar(x_idx - 0.5 * w, heat_before, w, label="Heat out (before)", color=c_heat_2, edgecolor="black", linewidth=0.4)
    ax_dec.bar(
        x_idx + 0.5 * w, heat_after, w, label="Heat out (after, HP)",
        color=c_heat_3, edgecolor="black", linewidth=0.4, hatch="///",
    )
    ax_dec.bar(x_idx + 1.5 * w, power_before, w, label="Power out (before)", color=c_pwr_2, edgecolor="black", linewidth=0.4)
    ax_dec.bar(
        x_idx + 2.5 * w, power_after, w, label="Power deficit (after, site)",
        color=c_pwr_3, edgecolor="black", linewidth=0.4, hatch="///",
    )
    ax_dec.set_ylabel("Capacity [MW]", fontsize=13)
    ax_dec.set_title("Replaced decentralized plants — before shutdown vs after heat pumps", fontsize=14)
    ax_dec.set_xticks(x_idx)
    ax_dec.set_xticklabels(names, rotation=35, ha="right", fontsize=9)
    ax_dec.grid(axis="y", alpha=0.35)
    ax_dec.tick_params(axis="y", labelsize=11)
    ax_dec.legend(loc="upper right", fontsize=9, ncol=2)

    hp_labels = ["Σ heat out\n(HP)", "Σ power deficit\n(HP)", "Σ power deficit\n(site)"]
    hp_vals = [sum_hp_heat, sum_hp_elec, sum_site_pwr]
    hub_labels = ["Hub fuel\ninput", "Hub power\ninput", "Methanol\noutput", "Heat losses\n(total)"]
    hub_vals = [
        hub_out.get("q_fuel_mwth", hub_out["input_fuel_mwth"]),
        hub_out.get("w_power_mwel", hub_out["power_input_mwel"]),
        hub_out.get("q_methanol_mwth", hub_out["methanol_out_mwth"]),
    ]
    loss_parts = [
        ("Combustor", hub_out.get("q_loss_combustor_mwth", 0.0)),
        ("Gasifier", hub_out.get("q_loss_gasifier_mwth", 0.0)),
        ("Electrolyzer", hub_out.get("q_loss_electrolyzer_mwth", 0.0)),
        ("Synthesis", hub_out.get("q_loss_synthesis_mwth", 0.0)),
    ]
    loss_colors = [c_loss_combustor, c_loss_gasifier, c_loss_electrolyzer, c_loss_synthesis]

    x_hp = np.array([0.0, 1.0, 2.0])
    x_hub = np.array([4.5, 5.5, 6.5])
    x_loss = 7.5
    bar_w = 0.75
    ax_sum.bar(x_hp, hp_vals, bar_w, color=[c_heat_2, c_pwr_2, c_pwr_3], edgecolor="black", linewidth=0.5, hatch="///")
    ax_sum.bar(
        x_hub,
        hub_vals,
        bar_w,
        color=[c_fuel, c_pwr_1, c_heat_1],
        edgecolor="black",
        linewidth=0.5,
    )
    loss_bottom = 0.0
    for (loss_label, loss_val), loss_color in zip(loss_parts, loss_colors):
        loss_val = max(0.0, float(loss_val))
        if loss_val <= 0:
            continue
        ax_sum.bar(
            x_loss,
            loss_val,
            bar_w,
            bottom=loss_bottom,
            color=loss_color,
            edgecolor="black",
            linewidth=0.4,
            label=loss_label,
        )
        loss_bottom += loss_val
    ax_sum.axvline(3.25, color="gray", linestyle=":", linewidth=1.0)
    ax_sum.set_xticks(list(x_hp) + list(x_hub) + [x_loss])
    ax_sum.set_xticklabels(hp_labels + hub_labels, fontsize=11)
    ax_sum.set_ylabel("Capacity [MW]", fontsize=13)
    ax_sum.set_title(
        f"Aggregated balances — η_total (hub) = {hub_out['eta_total']:.3f}  |  "
        f"{n_plants} plant(s) replaced",
        fontsize=13,
    )
    ax_sum.grid(axis="y", alpha=0.35)
    ax_sum.tick_params(axis="y", labelsize=11)
    ax_sum.legend(loc="upper right", fontsize=9, title="Hub heat losses", title_fontsize=9)

    if debug:
        print(
            "plot_replacement_energy_breakdown hub losses [MWth]:",
            {label: val for label, val in loss_parts},
            "total",
            hub_out.get("q_loss_total_mwth", loss_bottom),
        )

    fig.suptitle(
        f"Energy balance — {n_plants} plant(s) replaced (MW; MWth and MWel on one axis)",
        fontsize=14, y=1.01,
    )
    fig.tight_layout()
    d = os.path.dirname(out_path)
    if d:
        os.makedirs(d, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    if debug:
        print("plot_replacement_energy_breakdown saved:", out_path)
    plt.close(fig)


def plot_mitigation_cost_breakdown(
    plant_name: str,
    cost_details: Dict[str, Any],
    out_path: str,
    debug: bool = False,
) -> None:
    """Three-panel CCS cost figure for a single Mitigation plant (EUR/t CO₂)."""
    capex_k = cost_details["capex_overnight_kEUR"]
    opex_yr = cost_details["opex_annual_eur_yr"]
    annual_co2_kt_yr = cost_details["annual_co2_kt_yr"]

    fig, (ax_capex, ax_opex, ax_lev) = plt.subplots(3, 1, figsize=(12, 13))

    capex_items = [
        ("Capture", "capture"),
        ("Heat pump", "hp"),
        ("Loading", "loading"),
        ("Rail", "train"),
    ]
    capex_labels = [lbl for lbl, key in capex_items if capex_k.get(key, 0) > 0]
    capex_keys = [key for _, key in capex_items if capex_k.get(key, 0) > 0]
    capex_meur = [capex_k[k] * 1e-3 for k in capex_keys]
    cx = np.arange(len(capex_labels))
    ax_capex.bar(cx, capex_meur, color="#4477AA", edgecolor="black", linewidth=0.5)
    ax_capex.set_ylabel("Overnight CAPEX (MEUR)", fontsize=13)
    ax_capex.set_title(f"CCS overnight CAPEX — {plant_name}", fontsize=14)
    ax_capex.set_xticks(cx)
    ax_capex.set_xticklabels(capex_labels, fontsize=11)
    ax_capex.grid(axis="y", alpha=0.35)
    ax_capex.tick_params(axis="y", labelsize=11)

    opex_labels = ["Fixed (5%)", "Amine makeup", "Energy"]
    opex_keys = ["fix", "makeup", "energy"]
    opex_meur_a = [opex_yr[k] * 1e-6 for k in opex_keys]
    ox = np.arange(len(opex_labels))
    ax_opex.bar(ox, opex_meur_a, color="#228833", edgecolor="black", linewidth=0.5)
    ax_opex.set_ylabel("Annual OPEX (MEUR/a)", fontsize=13)
    ax_opex.set_title(f"CCS annual OPEX — {plant_name}", fontsize=14)
    ax_opex.set_xticks(ox)
    ax_opex.set_xticklabels(opex_labels, fontsize=11)
    ax_opex.grid(axis="y", alpha=0.35)
    ax_opex.tick_params(axis="y", labelsize=11)

    lev_map = [
        ("Capture\nCAPEX", "CAPEX_capture_lev"),
        ("HP\nCAPEX", "CAPEX_HP_lev"),
        ("Fixed\nOPEX", "OPEX_fix"),
        ("Makeup\nOPEX", "OPEX_makeup"),
        ("Energy\nOPEX", "OPEX_energy"),
        ("Loading", "loading_cost"),
        ("Truck", "truck_cost"),
        ("Pipeline", "pipeline_cost"),
        ("Rail", "rail_cost"),
        ("Shipping", "shipping_cost"),
        ("Storage", "storage_cost"),
    ]
    lev_labels = [lbl for lbl, key in lev_map if cost_details.get(key, 0) > 0]
    lev_vals = [cost_details[key] for _, key in lev_map if cost_details.get(key, 0) > 0]
    lev_colors = plt.cm.tab20(np.linspace(0, 0.85, max(len(lev_vals), 1)))
    lx = range(len(lev_vals))
    ax_lev.bar(lx, lev_vals, color=lev_colors, edgecolor="black", linewidth=0.6)
    ax_lev.axhline(
        cost_details["cost_CCS"], color="gray", linestyle="--", linewidth=1.2,
        label=f"Total = {cost_details['cost_CCS']:.0f} EUR/t",
    )
    ax_lev.set_xticks(list(lx))
    ax_lev.set_xticklabels(lev_labels, fontsize=10)
    ax_lev.set_ylabel("Levelized cost (EUR/t CO₂)", fontsize=13)
    ax_lev.set_title(f"Levelized cost breakdown ({annual_co2_kt_yr:,.0f} kt CO₂/a)", fontsize=14)
    ax_lev.legend(loc="upper right", fontsize=11)
    ax_lev.grid(axis="y", alpha=0.35)
    ax_lev.tick_params(axis="y", labelsize=11)

    fig.suptitle(f"Cost breakdown — Mitigation (CCS) @ {plant_name}", fontsize=15, y=1.01)
    fig.tight_layout()
    d = os.path.dirname(out_path)
    if d:
        os.makedirs(d, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    if debug:
        print("plot_mitigation_cost_breakdown saved:", out_path)
    plt.close(fig)


def plot_mitigation_energy_breakdown(
    plant_name: str,
    cost_details: Dict[str, Any],
    eta_boiler: float,
    out_path: str,
    debug: bool = False,
) -> None:
    """Two-panel CCS energy balance (MW; MWth and MWel on one axis)."""
    e = cost_details["energy"]
    c_fuel = "#7B241C"
    c_heat_1 = "#C0392B"
    c_heat_2 = "#E74C3C"
    c_heat_3 = "#F5B7B1"
    c_pwr_1 = "#1A5276"
    c_pwr_2 = "#2E86C1"
    c_pwr_3 = "#85C1E9"

    fig, (ax_site, ax_ccs) = plt.subplots(2, 1, figsize=(12, 10))

    site_labels = [
        f"Boiler loss\n(before, {100 * (1 - eta_boiler):.0f}%)",
        "Fuel in\n(before, HHV)",
        "Heat out\n(before)",
        "Power out\n(before)",
        "Heat from HEX\n(after)",
        "Heat from HP\n(after)",
        "Power\ndeficit (after)",
        "Power out\n(after, site)",
    ]
    site_vals = [
        e["q_boiler_loss_mwth"],
        e["q_fuel_plot_mwth"],
        e["q_heat_target_mwth"],
        e["p_chp_export_before_mwel"],
        e["qrec_hex_mwth"],
        e["qhp_out_mwth"],
        e["p_ccs_penalty_mwel"],
        max(e["p_chp_export_after_mwel"], 0.0),
    ]
    sx = np.arange(len(site_labels))
    site_colors = [c_heat_1, c_fuel, c_heat_2, c_pwr_2, c_heat_3, c_heat_3, c_pwr_3, c_pwr_1]
    hatch = [""] * 4 + ["///"] * 4
    ax_site.bar(sx, site_vals, color=site_colors, edgecolor="black", linewidth=0.5, hatch=hatch)
    ax_site.set_ylabel("Capacity [MW]", fontsize=13)
    ax_site.set_title(f"CHP site — before retrofit vs with CCS ({plant_name})", fontsize=14)
    ax_site.set_xticks(sx)
    ax_site.set_xticklabels(site_labels, fontsize=10)
    ax_site.grid(axis="y", alpha=0.35)
    ax_site.tick_params(axis="y", labelsize=11)

    ccs_labels = [
        "Capture\nreboiler", "Capture\nelec.", "Condition\nelec.", "Heat pump\nelec.",
    ]
    ccs_vals = [
        e["qreb_mwth"],
        e["pcapture_mwel"],
        e["pcondition_mwel"],
        e["whp_mwel"],
    ]
    cx = np.arange(len(ccs_labels))
    ccs_colors = plt.cm.Set2(np.linspace(0, 0.9, len(ccs_labels)))
    ax_ccs.bar(cx, ccs_vals, color=ccs_colors, edgecolor="black", linewidth=0.5)
    ax_ccs.set_ylabel("Capacity [MW]", fontsize=13)
    ax_ccs.set_title("CCS block — thermal and electrical loads", fontsize=14)
    ax_ccs.set_xticks(cx)
    ax_ccs.set_xticklabels(ccs_labels, fontsize=10)
    ax_ccs.grid(axis="y", alpha=0.35)
    ax_ccs.tick_params(axis="y", labelsize=11)

    fig.suptitle(
        f"Energy balance — Mitigation (CCS) @ {plant_name} (MW; MWth and MWel on one axis)",
        fontsize=14, y=1.01,
    )
    fig.tight_layout()
    d = os.path.dirname(out_path)
    if d:
        os.makedirs(d, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    if debug:
        print("plot_mitigation_energy_breakdown saved:", out_path)
    plt.close(fig)


def plot_recovery_cost_breakdown(
    plant_name: str,
    cost_details: Dict[str, Any],
    out_path: str,
    debug: bool = False,
) -> None:
    """Three-panel CCU cost figure for a single Recovery plant (EUR/t methanol)."""
    capex_k = cost_details["capex_overnight_kEUR"]
    opex_yr = cost_details["opex_annual_eur_yr"]
    annual_methanol_t_per_yr = cost_details["annual_methanol_t_yr"] * 1000.0

    fig, (ax_capex, ax_opex, ax_lev) = plt.subplots(3, 1, figsize=(12, 13))

    capex_labels = ["Capture", "Heat pump", "Comp. CO₂", "Comp. H₂", "H₂ (el.)", "Synthesis"]
    capex_keys = ["capture", "hp", "comp_co2", "comp_h2", "h2", "synthesis"]
    capex_meur = [capex_k[k] * 1e-3 for k in capex_keys]
    cx = np.arange(len(capex_labels))
    ax_capex.bar(cx, capex_meur, color="#4477AA", edgecolor="black", linewidth=0.5)
    ax_capex.set_ylabel("Overnight CAPEX (MEUR)", fontsize=13)
    ax_capex.set_title(f"CCU overnight CAPEX — {plant_name}", fontsize=14)
    ax_capex.set_xticks(cx)
    ax_capex.set_xticklabels(capex_labels, fontsize=11)
    ax_capex.grid(axis="y", alpha=0.35)
    ax_capex.tick_params(axis="y", labelsize=11)

    opex_labels = ["Fixed (5%)", "Amine makeup", "Energy"]
    opex_keys = ["fix", "makeup", "energy"]
    opex_meur_a = [opex_yr[k] * 1e-6 for k in opex_keys]
    ox = np.arange(len(opex_labels))
    ax_opex.bar(ox, opex_meur_a, color="#228833", edgecolor="black", linewidth=0.5)
    ax_opex.set_ylabel("Annual OPEX (MEUR/a)", fontsize=13)
    ax_opex.set_title(f"CCU annual OPEX — {plant_name}", fontsize=14)
    ax_opex.set_xticks(ox)
    ax_opex.set_xticklabels(opex_labels, fontsize=11)
    ax_opex.grid(axis="y", alpha=0.35)
    ax_opex.tick_params(axis="y", labelsize=11)

    lev_map = [
        ("Capture\nCAPEX", "CAPEX_capture_lev"),
        ("HP\nCAPEX", "CAPEX_HP_lev"),
        ("Comp. CO₂\nCAPEX", "CAPEX_comp_CO2_lev"),
        ("Comp. H₂\nCAPEX", "CAPEX_comp_H2_lev"),
        ("H₂\nCAPEX", "CAPEX_H2_lev"),
        ("Synth.\nCAPEX", "CAPEX_synthesis_lev"),
        ("Fixed\nOPEX", "OPEX_fix"),
        ("Makeup\nOPEX", "OPEX_makeup"),
        ("Energy\nOPEX", "OPEX_energy"),
    ]
    lev_labels = [lbl for lbl, key in lev_map if cost_details.get(key, 0) > 0]
    lev_vals = [cost_details[key] for _, key in lev_map if cost_details.get(key, 0) > 0]
    lev_colors = plt.cm.tab20(np.linspace(0, 0.85, len(lev_vals)))
    lx = range(len(lev_vals))
    ax_lev.bar(lx, lev_vals, color=lev_colors, edgecolor="black", linewidth=0.6)
    ax_lev.axhline(
        cost_details["cost_CCU"], color="gray", linestyle="--", linewidth=1.2,
        label=f"Total = {cost_details['cost_CCU']:.0f} EUR/t",
    )
    ax_lev.set_xticks(list(lx))
    ax_lev.set_xticklabels(lev_labels, fontsize=10)
    ax_lev.set_ylabel("Levelized cost (EUR/t methanol)", fontsize=13)
    ax_lev.set_title(f"Levelized cost breakdown ({annual_methanol_t_per_yr:,.0f} t methanol/a)", fontsize=14)
    ax_lev.legend(loc="upper right", fontsize=11)
    ax_lev.grid(axis="y", alpha=0.35)
    ax_lev.tick_params(axis="y", labelsize=11)

    fig.suptitle(f"Cost breakdown — Recovery (CCU) @ {plant_name}", fontsize=15, y=1.01)
    fig.tight_layout()
    d = os.path.dirname(out_path)
    if d:
        os.makedirs(d, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    if debug:
        print("plot_recovery_cost_breakdown saved:", out_path)
    plt.close(fig)


def plot_recovery_energy_breakdown(
    plant_name: str,
    cost_details: Dict[str, Any],
    eta_boiler: float,
    out_path: str,
    debug: bool = False,
) -> None:
    """Two-panel CCU energy balance (MW; MWth and MWel on one axis)."""
    e = cost_details["energy"]
    c_fuel = "#7B241C"
    c_heat_1 = "#C0392B"
    c_heat_2 = "#E74C3C"
    c_heat_3 = "#F5B7B1"
    c_pwr_1 = "#1A5276"
    c_pwr_2 = "#2E86C1"
    c_pwr_3 = "#85C1E9"

    fig, (ax_site, ax_ccu) = plt.subplots(2, 1, figsize=(12, 10))

    site_labels = [
        f"Boiler loss\n(before, {100 * (1 - eta_boiler):.0f}%)",
        "Fuel in\n(before, HHV)",
        "Heat out\n(before)",
        "Power out\n(before)",
        "Heat from HP\n(after)",
        "Power net\ndeficit (after)",
        "Power out\n(after, site)",
    ]
    site_vals = [
        e["q_boiler_loss_mwth"],
        e["q_fuel_plot_mwth"],
        e["q_heat_target_mwth"],
        e["p_chp_export_before_mwel"],
        e["qhp_out_mwth"],
        e["p_ccu_penalty_mwel"],
        max(e["p_chp_export_after_mwel"], 0.0),
    ]
    sx = np.arange(len(site_labels))
    site_colors = [c_heat_1, c_fuel, c_heat_2, c_pwr_2, c_heat_3, c_pwr_3, c_pwr_1]
    hatch = [""] * 4 + ["///"] * 3
    ax_site.bar(sx, site_vals, color=site_colors, edgecolor="black", linewidth=0.5, hatch=hatch)
    ax_site.set_ylabel("Capacity [MW]", fontsize=13)
    ax_site.set_title(f"CHP site — before retrofit vs with CCU ({plant_name})", fontsize=14)
    ax_site.set_xticks(sx)
    ax_site.set_xticklabels(site_labels, fontsize=10)
    ax_site.grid(axis="y", alpha=0.35)
    ax_site.tick_params(axis="y", labelsize=11)

    ccu_labels = [
        "Capture\nreboiler", "Capture\nelec.", "H₂\nelec.",
        "Comp. CO₂", "Comp. H₂", f"Recycle comp.\n(×{cost_details.get('recycle_ratio', 3):.0f})",
        "Heat pump\nelec.", "MeOH\noutput",
    ]
    ccu_vals = [
        e["qreb_mwth"],
        e["pcapture_mwel"],
        e["ph2_mwel"],
        e["wcomp_co2_mwel"],
        e["wcomp_h2_mwel"],
        e["wcomp_recycle_mwel"],
        e["whp_mwel"],
        e["qmethanol_out_mwth"],
    ]
    cx = np.arange(len(ccu_labels))
    ccu_colors = plt.cm.Set2(np.linspace(0, 0.9, len(ccu_labels)))
    ax_ccu.bar(cx, ccu_vals, color=ccu_colors, edgecolor="black", linewidth=0.5)
    ax_ccu.set_ylabel("Capacity [MW]", fontsize=13)
    ax_ccu.set_title("CCU block — thermal and electrical loads", fontsize=14)
    ax_ccu.set_xticks(cx)
    ax_ccu.set_xticklabels(ccu_labels, fontsize=10)
    ax_ccu.grid(axis="y", alpha=0.35)
    ax_ccu.tick_params(axis="y", labelsize=11)

    fig.suptitle(
        f"Energy balance — Recovery (CCU) @ {plant_name} (MW; MWth and MWel on one axis)",
        fontsize=14, y=1.01,
    )
    fig.tight_layout()
    d = os.path.dirname(out_path)
    if d:
        os.makedirs(d, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    if debug:
        print("plot_recovery_energy_breakdown saved:", out_path)
    plt.close(fig)


def WACCUS_EPR(
    # [C] Constants
    EPR_design="Recovery", # [Mitigation, Recovery, Replacement]
    plants_df=None, 
    shipping_costs=None,
    compression_costs=None,
    thermo_props=None,    
    SEK_to_EUR=0.091,
    profit=0.10,
    plot_results=False,

    CPI2015=314.21, # [SCB]
    CPI2025=417.96, # [SCB]
    capture_rate = 0.90,    # [-] 
    eta_is = 0.80,
    CEPCI_reference = 600,  # [-] [University of Manchester, 2025] default ref year for legacy CAPEX refs
    CEPCI_capture_reference = 900,  # [-] Sysav 2026 capture estimate already at CEPCI_2026
    CEPCI_HP_reference = 816,  # [-] CEPCI 2022 base year for heat-pump overnight CAPEX (Bergander)

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
    q_hex = 0.64,           # [MWth/MWreb] [Beiron, 2022] heat recovery from capture reboiler
    p_capture = 0.1,        # [MWh/tCO2] [Beiron, 2022]
    p_condition = 0.37,     # [MJ/kgCO2] [Kumar, 2023]
    COP = 3,                # [MWth/MWel]
    eta_electrolyzer = 0.699, # [MWH2/MWel] Table2.1 MSc Jacobsson & Palmgren (2025)
    heat_optimism = 0.15,    # [-] [0-1.0] [0-100%] optimistic assumption on heat recovery from electrolyzers
    opex_var_sorting_sek_per_t_waste = 200.0,  # [SEK/t waste] Brista basis
    opex_var_gasification_eur_per_mwh_fuel = 1.4,  # [EUR/MWh_fuel] Beiron 2026

    k = 0.67,                           # [-] [Stenström, 2025] assumed economy-of-scale factor
    dr = 0.075,                         # [-]
    t = 25,                             # [yr]
    opex_fix = 0.05,                     # [-] fixed OPEX as fraction of overnight CAPEX
    capex_ref_train_eur = 8610000,  # [EUR] fixed train @ 15 wagons × 60 t/wagon [Gunnarsson, 2025]
    capex_ref_synthesis_meur = 1.8749,  # [MEUR] synthesis CAPEX ∝ (m_methanol [t/d])^-0.315 [Danish Renewable Fuels]

    camine = 44,            # [SEK/tCO2] [Ramboll-Malmö, 2023]
    celc = 50,              # [EUR/MWh]
    cheat = 0.75,           # [% of elc]
    pmethanol = 650,        # [EUR/t] [MSc Omar & Widgren, 2025]

    shipping_case = "pessimist_1Mt", # The main transport uncertainty! Dictates 1Mt, 2Mt, or 3Mt costs.
    storage = "oygarden",   # ["oygarden", "kalundborg"]
    storage_cost = 20, # [EUR/tCO2] https://www.globalccsinstitute.com/wp-content/uploads/2025/12/Cost-of-CO2-Storage-1225.pdf
    transport_cost_factor = 1.0,  # [-] scales CCS/replacement transport stack

    CRC = 100,              # [EUR/tCO2]
    ETS = 80,               # [EUR/tCO2] Use this report: The EU-ETS Price Through 2030 and Beyond: A closer look at drivers, models and assumptions (https://www.ecologic.eu/19034)
    
    # [L] Levers
    EPR_products = True,
    EPR_fee = 200, # [EUR/tpl]

    # Replacement (centralized gasification) — process and cost references
    n_replacement_cases = 10,  # [-] cumulative cases: 1 … n largest plants by m_tot
    FLH_gasifier = 8000.0,  # [h/yr] hub full-load hours
    LHV_H2_MJ_PER_KMOL = 243.0,  # [MJ/kmolH2]
    LHV_CO_MJ_PER_KMOL = 286.0,  # [MJ/kmolCO]
    LHV_CH3OH = 21.1,  # [MJ/kg]
    recycle_ratio = 3.0,  # [-] recycle compression multiplier
    gasified_carbon_fraction = 0.70,  # [-] fraction of C to gasifier branch [Ecoplanta]
    frac_combustor_pl = 0.25,  # [-] share of combustor C that is plastic
    frac_energy = 0.60,  # [-] share of gasified fuel energy ending up in methanol [check Alberto Alamia]
    eta_boiler = 0.85,  # [-] boiler efficiency (plot-only loss bar at replaced sites)

    ADJUST_CEPCI = True,  # [-] if False, all CEPCI escalation factors are 1.0
    CEPCI_target = 900,  # [-] [University of Manchester, 2025] cost evaluation year
    CEPCI_sorting_ref = 900,  # [-] Tekniska Verken quote year
    CEPCI_opex_sorting_ref = 816,  # [-] Brista OPEX base year
    CEPCI_gasification_ref = 600,  # [-] ECOPLANTA CAPEX base
    CEPCI_opex_gasification_ref = 900,  # [-] Beiron OPEX year
    CEPCI_h2_ref = 900,  # [-] electrolyzer / compression CAPEX base

    # Reference CAPEX — specific (× installed MW; no capacity_ref)
    capex_ref_hp_keur_per_mwth = 860,  # [kEUR/MWth] heat pump [Bergander & Hellander, 2024]
    capex_ref_h2_keur_per_mwel = 550,  # [kEUR/MWel] electrolyzer Danish AEC100MW, or [2075kEUR if clean-hydrogen observatory]

    # Reference CAPEX + capacity — overnight cost scales as (capacity / capacity_ref)^k
    capex_ref_capture_keur = 3.7e9 * 0.091 / 1000,  # [kEUR] capture @ capacity_ref_capture_kt_per_yr
    capacity_ref_capture_kt_per_yr = 400.0,  # [ktCO2/yr] Sysav 2026 reference
    capex_ref_loading_sek = 63000000,  # [SEK] loading hub @ capacity_ref_loading_kt_per_yr [Koldioxid på tåg, 2024]
    capacity_ref_loading_kt_per_yr = 150.0,  # [ktCO2/yr]
    capex_ref_sorting_msek = 650.0,  # [MSEK] sorting @ capacity_ref_sorting_t_waste_per_yr [Tekniska Verken]
    capacity_ref_sorting_t_waste_per_yr = 200_000.0,  # [t waste/yr]
    capex_ref_gasification_meur = 749_729_639 * 1e-6,  # [MEUR] gasification @ capacity_ref_gasification_t_methanol_per_yr [ECOPLANTA]
    capacity_ref_gasification_t_methanol_per_yr = 237_000.0,  # [t methanol/yr]

):
    # Store parameters in dicts
    c = {
        "plants_df": plants_df,
        "shipping_costs": shipping_costs,
        "compression_costs": compression_costs,
        "thermo_props": thermo_props,
        "SEK_to_EUR": SEK_to_EUR,
        "profit": profit,

        "capex_ref_capture_keur": capex_ref_capture_keur,
        "capacity_ref_capture_kt_per_yr": capacity_ref_capture_kt_per_yr,
        "capex_ref_loading_sek": capex_ref_loading_sek,
        "capacity_ref_loading_kt_per_yr": capacity_ref_loading_kt_per_yr,
        "capture_rate": capture_rate,
        "eta_is": eta_is,
        "LHV_CH3OH": LHV_CH3OH,

        "n_replacement_cases": n_replacement_cases,
        "FLH_gasifier": FLH_gasifier,
        "LHV_H2_MJ_PER_KMOL": LHV_H2_MJ_PER_KMOL,
        "LHV_CO_MJ_PER_KMOL": LHV_CO_MJ_PER_KMOL,
        "recycle_ratio": recycle_ratio,
        "gasified_carbon_fraction": gasified_carbon_fraction,
        "frac_combustor_pl": frac_combustor_pl,
        "frac_energy": frac_energy,
        "eta_boiler": eta_boiler,
        "capex_ref_sorting_meur": capex_ref_sorting_msek * SEK_to_EUR,  # [MEUR]
        "capacity_ref_sorting_t_waste_per_yr": capacity_ref_sorting_t_waste_per_yr,

        "capex_ref_gasification_meur": capex_ref_gasification_meur,
        "capacity_ref_gasification_t_methanol_per_yr": capacity_ref_gasification_t_methanol_per_yr,

        "ADJUST_CEPCI": ADJUST_CEPCI,
        "CEPCI_target": CEPCI_target,
        "CEPCI_reference": CEPCI_reference,
        "CEPCI_capture_reference": CEPCI_capture_reference,
        "CEPCI_sorting_ref": CEPCI_sorting_ref,
        "CEPCI_opex_sorting_ref": CEPCI_opex_sorting_ref,
        "CEPCI_HP_reference": CEPCI_HP_reference,
        "CEPCI_gasification_ref": CEPCI_gasification_ref,
        "CEPCI_opex_gasification_ref": CEPCI_opex_gasification_ref,
        "CEPCI_h2_ref": CEPCI_h2_ref,
    }
    x = {
        "q_reb": q_reb,
        "q_hex": q_hex,
        # "q_electrolyzer": q_electrolyzer,
        "p_capture": p_capture,
        "p_condition": p_condition,
        "COP": COP,
        "eta_electrolyzer": eta_electrolyzer,
        "heat_optimism": heat_optimism,
        "opex_var_sorting_sek_per_t_waste": opex_var_sorting_sek_per_t_waste,
        "opex_var_gasification_eur_per_mwh_fuel": opex_var_gasification_eur_per_mwh_fuel,

        "k": k,
        "dr": dr,
        "t": t,
        "capex_ref_hp_keur_per_mwth": capex_ref_hp_keur_per_mwth,
        "capex_ref_h2_keur_per_mwel": capex_ref_h2_keur_per_mwel,
        "capex_ref_train_eur": capex_ref_train_eur,
        "capex_ref_synthesis_meur": capex_ref_synthesis_meur,
        "opex_fix": opex_fix,

        "camine": camine,
        "celc": celc,
        "cheat": cheat,
        "pmethanol": pmethanol,

        "shipping_case": shipping_case,
        "storage": storage,
        "storage_cost": storage_cost,
        "transport_cost_factor": transport_cost_factor,

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

    # (2) Simulate EPR subsidies per case
    if EPR_design == "Mitigation":
        bids = []
        for _, plant in plants_df.iterrows():
            strike_price, FCCS, BECCS, Ppenalty, Pnew_capacity, Qpenalty, cost_details = plan_CCS(plant, c, x, l)
            bids.append({"Name": plant["Name"], "Design": EPR_design, "strike_price": strike_price,
                            "FCCS": FCCS, "BECCS": BECCS, "Ppenalty": Ppenalty, "Pnew_capacity": Pnew_capacity,
                            "Qpenalty": Qpenalty, "cost_details": cost_details})

        if plot_results:
            sorted_costs = sorted(bids, key=lambda r: r["cost_details"]["cost_CCS"])
            for case in [sorted_costs[0], sorted_costs[-1]]:
                tag = str(case["Name"]).replace(" ", "_")
                details = case["cost_details"]
                plot_mitigation_cost_breakdown(
                    case["Name"],
                    details,
                    out_path=f"results/mitigation_cost_breakdown_{tag}.png",
                    debug=False,
                )
                plot_mitigation_energy_breakdown(
                    case["Name"],
                    details,
                    c["eta_boiler"],
                    out_path=f"results/mitigation_energy_breakdown_{tag}.png",
                    debug=False,
                )

    elif EPR_design == "Recovery":
        bids = []
        for _, plant in plants_df.iterrows():
            strike_price, FCCU, BCCU, Qmethanol, Ppenalty, Pnew_capacity, Qpenalty, cost_details = plan_CCU(plant, c, x, l)
            bids.append({"Name": plant["Name"], "Design": EPR_design, "strike_price": strike_price,
                            "FCCU": FCCU, "BCCU": BCCU, "Ppenalty": Ppenalty, "Pnew_capacity": Pnew_capacity,
                            "Qpenalty": Qpenalty, "Qmethanol": Qmethanol, "cost_details": cost_details})
        
        if plot_results:
            sorted_costs = sorted(bids, key=lambda r: r["cost_details"]["cost_CCU"])
            for case in [sorted_costs[0], sorted_costs[-1]]:
                tag = str(case["Name"]).replace(" ", "_")
                details = case["cost_details"]
                plot_recovery_cost_breakdown(
                    case["Name"],
                    details,
                    out_path=f"results/recovery_cost_breakdown_{tag}.png",
                    debug=False,
                )
                plot_recovery_energy_breakdown(
                    case["Name"],
                    details,
                    c["eta_boiler"],
                    out_path=f"results/recovery_energy_breakdown_{tag}.png",
                    debug=False,
                )

    elif EPR_design == "Replacement":
        plants = c["plants_df"].sort_values(by="m_tot", ascending=False).reset_index(drop=True)
        n_cases = c["n_replacement_cases"]
        if len(plants) < n_cases:
            raise ValueError(f"Need at least {n_cases} plants in plants_df; found {len(plants)}.")

        replacement_cases: List[Dict[str, Any]] = []

        for n_plants in range(1, n_cases + 1):
            subset = plants.iloc[:n_plants]
            fuel = aggregate_fuel(subset)
            gasifier_results = plan_gasifier(fuel, c, x, debug=False)
            replacement_results = plan_replacements(subset, c, x, debug=False)
            lev = levelize_all_replacement_costs(gasifier_results, replacement_results, x, debug=False)

            replacement_cases.append(
                {
                    "n_plants": n_plants,
                    "gasifier": gasifier_results,
                    "replacement": replacement_results,
                    "central_costs": _gasifier_central_costs(gasifier_results),
                    "levelized": lev,
                    "plants_slice": subset,
                }
            )

        if plot_results:
            for case in (replacement_cases[0], replacement_cases[-1]):
                n_p = case["n_plants"]
                tag = f"{n_p}_plant" if n_p == 1 else f"{n_p}_plants"
                plot_replacement_cost_breakdown(
                    n_p,
                    case["replacement"],
                    case["central_costs"],
                    case["levelized"],
                    case["gasifier"]["annual_methanol_t_yr"],
                    out_path=f"results/replacement_cost_breakdown_{tag}.png",
                    debug=False,
                )
                plot_replacement_energy_breakdown(
                    n_p,
                    case["gasifier"],
                    case["replacement"],
                    case["plants_slice"],
                    c,
                    out_path=f"results/replacement_energy_breakdown_{tag}.png",
                    debug=False,
                )

    else:
        raise ValueError(f"Unknown EPR_design: {EPR_design}")

    # (3) Distribute subsidies
    if EPR_design == "Mitigation":
        fco2_key, bco2_key = "FCCS", "BECCS"
        for bid in bids:
            cost_gap_raw = (bid["strike_price"] - x["ETS"]) * (bid[fco2_key] + bid[bco2_key]) * 1000  # [EUR/yr]
            bid["cost_gap_plant"] = max(0.0, cost_gap_raw)  # [EUR/yr] profitable vs ETS → no subsidy
        remaining_fund = available_subsidies
        for bid in sorted(bids, key=lambda b: b["strike_price"]):
            subsidy_requested = bid["cost_gap_plant"] * (1 + c["profit"])  # [EUR/yr]
            bid["Financed"] = subsidy_requested <= remaining_fund
            if bid["Financed"] and subsidy_requested > 0:
                remaining_fund -= subsidy_requested
            bid["subsidy_requested_eur_yr"] = subsidy_requested
            bid["Subsidized"] = bid["Financed"] and bid["cost_gap_plant"] > 0

    elif EPR_design == "Recovery":
        for bid in bids:
            methanol_mass_t_yr = bid["Qmethanol"] / (c["LHV_CH3OH"] / 3600) / 1000  # [t/yr]
            cost_gap_raw = (bid["strike_price"] - x["pmethanol"]) * methanol_mass_t_yr  # [EUR/yr]
            bid["cost_gap_plant"] = max(0.0, cost_gap_raw)  # [EUR/yr] profitable vs market → no subsidy
        remaining_fund = available_subsidies
        for bid in sorted(bids, key=lambda b: b["strike_price"]):
            subsidy_requested = bid["cost_gap_plant"] * (1 + c["profit"])  # [EUR/yr]
            bid["Financed"] = subsidy_requested <= remaining_fund
            if bid["Financed"] and subsidy_requested > 0:
                remaining_fund -= subsidy_requested
            bid["subsidy_requested_eur_yr"] = subsidy_requested
            bid["Subsidized"] = bid["Financed"] and bid["cost_gap_plant"] > 0

    elif EPR_design == "Replacement":
        for case in replacement_cases:
            annual_methanol_t_yr = case["gasifier"]["annual_methanol_t_yr"]  # [t/a]
            cost_gap_raw = (case["levelized"]["eur_t_total"] - x["pmethanol"]) * annual_methanol_t_yr  # [EUR/yr]
            cost_gap_replacement = max(0.0, cost_gap_raw)  # [EUR/yr] profitable vs market → no subsidy
            subsidy_requested = cost_gap_replacement * (1 + c["profit"])  # [EUR/yr] total for this case
            case["cost_gap_replacement"] = cost_gap_replacement
            case["subsidy_requested_eur_yr"] = subsidy_requested
            case["Affordable"] = subsidy_requested <= available_subsidies

        affordable_cases = [rc for rc in replacement_cases if rc["Affordable"]]
        if affordable_cases:
            selected_case = max(
                affordable_cases,
                key=lambda rc: rc["gasifier"]["annual_methanol_t_yr"],
            )
            selected_n_plants = selected_case["n_plants"]
        else:
            selected_case = None
            selected_n_plants = None

        for case in replacement_cases:
            case["Financed"] = selected_n_plants is not None and case["n_plants"] == selected_n_plants
            case["Subsidized"] = case["Financed"] and case["cost_gap_replacement"] > 0
        remaining_fund = available_subsidies - sum(case["subsidy_requested_eur_yr"] for case in replacement_cases if case["Subsidized"])

    results = {"EPR_design": EPR_design}
    if EPR_design in ("Mitigation", "Recovery"):
        results["bids"] = bids
    elif EPR_design == "Replacement":
        results["replacement_cases"] = replacement_cases

    # Mass KPIs (energy and economics later)
    KPI1 = 0.0  # [n] CHP plants financed (CCS, CCU retrofit, or replacement)
    KPI2 = 0.0  # [ktCO2f/yr] fossil CO2 captured and STORED  — financed CCS plants
    KPI3 = 0.0  # [ktCO2b/yr] biogenic CO2 captured and STORED— financed CCS plants
    KPI4 = 0.0  # [ktMeOHf/yr] fossil methanol — financed / selected case
    KPI5 = 0.0  # [ktMeOHb/yr] biogenic methanol — financed / selected case
    KPI6 = 0.0  # [ktC/yr] carbon treated (stored CCS, recovered CCU, or gasified MeOH)
    KPI7 = 0.0  # [ktCO2f/yr] residual fossil CO2 — full fleet
    KPI8 = 0.0  # [ktCO2f/yr] fossil CO2 at hub combustor (Replacement only)
    KPI9 = 0.0  # [ktCO2b/yr] biogenic CO2 at hub combustor (Replacement only)

    KPI10 = 0.0  # Mitigation/Recovery: new equipment capacity [MWe]; Replacement: hub + site electrical demand [MWel]
    KPI11 = 0.0  # [TWh/yr] new power required

    KPI12 = plastic_supply / 1e6  # [Mtpl/yr] targeted plastic supply
    KPI13 = EPR_fee  # [EUR/tpl] EPR fee
    KPI14 = available_subsidies / 1e6 # [MEUR/yr] available subsidies
    KPI15 = remaining_fund / 1e6 # [MEUR/yr] remaining subsidies
    KPI16 = granulate_inc * 100 # [%] granulate price increase
    KPI17 = products_inc * 100 # [%] products price increase
    replacement_cost = float("nan")  # [EUR/t MeOH] subsidized Replacement only (cost gap > 0); NaN if profitable vs pmethanol
    replacement_cost_financed = float("nan")  # [EUR/t MeOH] financed Replacement hub; NaN if no affordable case

    plants_lookup = c["plants_df"].set_index("Name")

    if EPR_design == "Mitigation":
        for bid in bids:
            plant = plants_lookup.loc[bid["Name"]]
            fossil_co2_kt_yr = plant["Fossil"]  # [ktCO2f/yr]
            if not bid["Financed"]:
                KPI7 += fossil_co2_kt_yr  # [ktCO2f/yr] unabated CHP
                continue
            KPI1 += 1
            KPI2 += bid["FCCS"] 
            KPI3 += bid["BECCS"] 
            KPI7 += fossil_co2_kt_yr * (1.0 - c["capture_rate"])  # [ktCO2f/yr] uncaptured
            KPI10 += max(0.0, bid["Pnew_capacity"])
            KPI11 += bid["Ppenalty"] / 1e6  # [TWh/yr]

    elif EPR_design == "Recovery":
        for bid in bids:
            plant = plants_lookup.loc[bid["Name"]]
            fossil_co2_kt_yr = plant["Fossil"]  # [ktCO2f/yr]
            if not bid["Financed"]:
                KPI7 += fossil_co2_kt_yr  # [ktCO2f/yr] unabated CHP
                continue
            KPI1 += 1
            KPI4 += bid["FCCU"] * 32.0 / 44.0 # [ktMeOHf/yr] 
            KPI5 += bid["BCCU"] * 32.0 / 44.0 # [ktMeOHb/yr] 
            KPI7 += fossil_co2_kt_yr * (1.0 - c["capture_rate"])  # [ktCO2f/yr] uncaptured
            KPI10 += max(0.0, bid["Pnew_capacity"])
            KPI11 += bid["Ppenalty"] / 1e6  # [TWh/yr]

    elif EPR_design == "Replacement":
        plants_df_kpi = c["plants_df"]
        KPI7 = plants_df_kpi["Fossil"].sum()  # [ktCO2f/yr] all CHP plants (baseline)
        selected_case = next((rc for rc in replacement_cases if rc["Financed"]), None)
        if selected_case is not None:
            KPI1 = float(selected_case["n_plants"])
            replaced = selected_case["plants_slice"]
            KPI7 -= replaced["Fossil"].sum()  # replaced sites shut down
            gasifier = selected_case["gasifier"]
            annual_methanol_t_yr = gasifier["annual_methanol_t_yr"]
            KPI4 = annual_methanol_t_yr * (1.0 - gasifier["frac_bio_gasifier"]) / 1000.0 # [ktMeOHf/yr]
            KPI5 = annual_methanol_t_yr * gasifier["frac_bio_gasifier"] / 1000.0 # [ktMeOHb/yr]
            combustor_co2_kt_yr = gasifier["c_combusted_t_yr"] * 44.0 / 12.0 / 1000.0  # [ktCO2/yr]
            KPI7 += combustor_co2_kt_yr * (1.0 - gasifier["frac_bio_combustor"])  # [ktCO2f/yr] hub combustor
            KPI8 = combustor_co2_kt_yr * (1.0 - gasifier["frac_bio_combustor"])  # [ktCO2f/yr] hub combustor
            KPI9 = combustor_co2_kt_yr * gasifier["frac_bio_combustor"]  # [ktCO2b/yr] hub combustor
            replacement = selected_case["replacement"]
            KPI10 = gasifier["power_input_mwel"] + replacement["p_site_deficit_mwel"]  # [MWel] hub + site deficit
            KPI11 = (gasifier["p_energy_mwh_yr"] + replacement["p_energy_mwh_yr"]) / 1e6  # [TWh/yr]
            replacement_cost_financed = float(selected_case["levelized"]["eur_t_total"])

        subsidized_case = next((rc for rc in replacement_cases if rc.get("Subsidized")), None)
        if subsidized_case is not None:
            replacement_cost = float(subsidized_case["levelized"]["eur_t_total"])

    # Carbon treated from stored CO2 (KPI2–3) and methanol (KPI4–5) [ktC/yr]
    KPI6 = (
        KPI2 * 12.0 / 44.0
        + KPI3 * 12.0 / 44.0
        + KPI4 * 12.0 / 32.0
        + KPI5 * 12.0 / 32.0
    )

    results["KPI1"] = KPI1
    results["KPI2"] = KPI2
    results["KPI3"] = KPI3
    results["KPI4"] = KPI4
    results["KPI5"] = KPI5
    results["KPI6"] = KPI6
    results["KPI7"] = KPI7
    results["KPI8"] = KPI8
    results["KPI9"] = KPI9
    results["KPI10"] = KPI10
    results["KPI11"] = KPI11
    results["KPI12"] = KPI12
    results["KPI13"] = KPI13
    results["KPI14"] = KPI14
    results["KPI15"] = KPI15
    results["KPI16"] = KPI16
    results["KPI17"] = KPI17
    results["replacement_cost"] = replacement_cost
    results["replacement_cost_financed"] = replacement_cost_financed
    return results

if __name__ == "__main__":

    # Read data
    plants_df = pd.read_csv('data/plants_clean.csv')
    shipping_costs = pd.read_csv('data/shipping_costs.csv')
    compression_costs = pd.read_csv('data/compression_costs.csv')
    thermo_props = get_CoolProp()

    # Adjust dataframes: add 0.5Mt shipping costs, convert SEK to EUR
    SEK_to_EUR = 0.091
    shipping_costs = shipping_adjustment(shipping_costs, scaling=0.67, debug=False)
    cost_columns = [col for col in shipping_costs.columns if col != 'distance']
    shipping_costs[cost_columns] = shipping_costs[cost_columns] * SEK_to_EUR
    # Run the model
    results = WACCUS_EPR(
        EPR_design="Replacement", # Mitigation, Recovery, Replacement 
        plants_df=plants_df, 
        shipping_costs=shipping_costs,
        compression_costs=compression_costs,
        thermo_props=thermo_props,    
        SEK_to_EUR=SEK_to_EUR,
        profit=0.10,
        plot_results=True,
    )

    print("\n----------------------------------------These are the simulation results:----------------------------------------")
    print(
        f"\nKPIs: plants financed={results['KPI1']:.0f}, "
        f"\nCO2f stored={results['KPI2']:,.0f} kt/a, CO2b stored={results['KPI3']:,.0f} kt/a, "
        f"MeOHf={results['KPI4']:,.0f} kt/a, MeOHb={results['KPI5']:,.0f} kt/a, "
        f"carbon treated={results['KPI6']:,.0f} ktC/a, "
        f"residual CO2f={results['KPI7']:,.0f} kt/a, "
        f"hub combustor CO2f={results['KPI8']:,.0f} kt/a, hub combustor CO2b={results['KPI9']:,.0f} kt/a, "
        f"\nnew power capacity={results['KPI10']:,.1f} MWel, new power={results['KPI11']:,.1f} TWh/yr"
        f"\ntargeted plastic supply={results['KPI12']:,.2f} Mtpl/yr, EPR fee={results['KPI13']:,.0f} EUR/tpl, "
        f"available subsidies={results['KPI14']:,.0f} MEUR/yr,"
        f"remaining subsidies={results['KPI15']:,.0f} MEUR/yr,"
        f"\ngranulate price increase={results['KPI16']:,.0f} %, "
        f"products price increase={results['KPI17']:,.0f} %"
    )
    plt.show()
