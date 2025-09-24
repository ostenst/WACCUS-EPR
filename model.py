# This is the model.py file
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

def levelize_kEUR(CAPEX, annual_CO2, x):
    """
    Levelize CAPEX in [kEUR] to [EUR/tCO2].

    """
    CRF = x["dr"] * (1 + x["dr"])**x["t"] / ((1 + x["dr"])**x["t"] - 1)
    CAPEX_annual = CAPEX * CRF  # [kEUR/yr]
    CAPEX_lev = CAPEX_annual / annual_CO2 * 1000  # [EUR/tCO2]
    return CAPEX_lev

def truck_cost(annual_CO2, distance, trucks_df):
    """Get interpolated truck cost for given mass and distance using 2D interpolation"""

    masses = trucks_df['mass'].values
    distances = trucks_df['km'].values
    costs = trucks_df['EUR/ton'].values
    
    # Create grid points for interpolation
    points = np.column_stack((masses, distances))
    
    # Check if point is within bounds, if not use extrapolation
    mass_min, mass_max = masses.min(), masses.max()
    dist_min, dist_max = distances.min(), distances.max()
    
    if annual_CO2 < mass_min or annual_CO2 > mass_max or distance < dist_min or distance > dist_max:
        # Use nearest neighbor for extrapolation
        truck_cost = griddata(points, costs, (annual_CO2, distance), method='nearest')
    else:
        # Use linear interpolation
        truck_cost = griddata(points, costs, (annual_CO2, distance), method='linear')
    return truck_cost

def loading_cost(annual_CO2, x):
    """Estimate CAPEX of loading infrastructure based on a 150 ktCO2/yr reference"""
    loading_cost = x["CAPEXref_loading"] * (annual_CO2/(150*10**3)) ** x["k"] * x["CEPCI"] /1000 # [kEUR]
    loading_cost = levelize_kEUR(loading_cost, annual_CO2, x) # [EUR/tCO2]
    return loading_cost

def pipeline_cost(annual_CO2, pipeline_length, x):
    """ Pipeline length in [m], returns levelized CAPEX in [EUR/tCO2]"""
    mCO2 = annual_CO2*1000/x["FLH"]/3600    # [kgCO2/s]
    nCO2 = mCO2/44                          # [kmolCO2/s]
    VCO2 = nCO2*22.4                        # [m3CO2/s] assuming ideal gas at standard conditions
    vCO2 = 16                               # [m/s] [Tharun, 2025] check paper "Enhancing early ..." and Appendix
    Mcomp = 1.2 # [-] margin
    Cfcomp = 568 # [EUR/m2]
    CAPEX_pipeline = (2*np.pi*((VCO2/vCO2)/np.pi)**0.5 * (pipeline_length*Mcomp)) * Cfcomp # [EUR2015] [Tharun, 2025]
    CAPEX_pipeline = CAPEX_pipeline * x["CEPCI"] / 1000 # [kEUR]
    CAPEXlev_pipeline = levelize_kEUR(CAPEX_pipeline, annual_CO2, x) # [EUR/tCO2]
    return CAPEXlev_pipeline

def plan_CCS(plant, c, x, l):
    # burn fuel and capture/condition CO2
    mfuel = plant["Qwaste"] / (x["LHVf"]/3600) /3600    # [kgf/s]
    mCO2 = mfuel* x["Ccontent"] * 44/12                 # [kgCO2/s]
    mcaptured = mCO2 * x["capture_rate"]                # [kgCO2/s]
    Qreb = mcaptured * x["q_reb"]                       # [MW]
    Pcapture = x["p_capture"] * mcaptured/1000*3600     # [MW] 
    Pcondition = x["p_condition"] * mcaptured           # [MW]  

    # penalize CHP and recover heat up to 100 % of original DH - use whatever power is available for HP
    P = plant["P"] * (1 - Qreb/plant["Qwaste"])         # assuming live steam is used for reboiler
    P = P - Pcapture - Pcondition
    Qdh = plant["Qdh"] * (1 - Qreb/plant["Qwaste"])

    Qhex = 0.64 * Qreb                                  # [MW] [Beiron, 2022]
    Qdiff = plant["Qdh"] - (Qdh + Qhex)
    if Qdiff < 0:
        raise ValueError
    else:
        Whp = Qdiff / x["COP"]
        if Whp > P: 
            Whp = P
    Qdh = Qdh + Qhex + Whp*x["COP"]
    P -= Whp
    Ppenalty = (plant["P"] - P) * x["FLH"]        # [MWh/yr]
    Qpenalty = (plant["Qdh"] - Qdh) * x["FLH"]    # [MWh/yr]

    # Estimate CAPEX and on-site OPEX
    annual_CO2 = mcaptured/1000*3600 * x["FLH"]  # [tCO2/yr]
    CAPEX_capture = x["CAPEXref_capture"] * (annual_CO2/(400*10**3)) ** x["k"] * x["CEPCI"] # [kEUR]
    CAPEXlev_capture = levelize_kEUR(CAPEX_capture, annual_CO2, x)                  # [EUR/tCO2]

    OPEXfix = (CAPEX_capture*1000 * x["OPEXfix"]) / annual_CO2                      # [EUR/tCO2] 
    OPEXmakeup = x["camine"]                                                        # [EUR/tCO2]
    OPEXenergy = (Ppenalty*x["celc"] + Qpenalty*x["celc"]*x["cheat"]) / annual_CO2  # [EUR/tCO2]
    OPEX = OPEXfix + OPEXmakeup + OPEXenergy   

    # Calculate transport costs
    if plant['Truck_distance'] is not None and not pd.isna(plant['Truck_distance']):
        cost_loading = loading_cost(annual_CO2, x)                                  # [EUR/tCO2]
        cost_truck = truck_cost(annual_CO2, plant['Truck_distance'], c["truck_df"]) # [EUR/tCO2]

    if plant['Pipeline_distance'] is not None and not pd.isna(plant['Pipeline_distance']):
        cost_pipeline = pipeline_cost(annual_CO2, plant['Pipeline_distance'], x)    # [EUR/tCO2]

    # NEXT: IMPLEMENT TRAIN AND SHIP COSTS


    # # Construct a reversed auction bid
    # CAC = OPEX + levelized_CAPEX + transport_cost                        # [EUR/t]

    # fossil = plant["Fossil"] / plant["Total"]                                 # [tfossil/t] share of fossil CO2
    # biogenic = 1 - fossil                                                     # [tbiogenic/t] share of biogenic CO2
    # incentives = fossil * x["ETS"] + biogenic * x["CRC"]                      # [EUR/t]
    # bid = CAC - incentives          

    # # Store detailed cost data
    # cost_details = {
    #     'OPEXfix': OPEXfix,
    #     'OPEXmakeup': OPEXmakeup,
    #     'OPEXenergy': OPEXenergy,
    #     'OPEX': OPEX,
    #     'levelized_CAPEX': levelized_CAPEX,
    #     'transport_cost': transport_cost,
    #     'CAC': CAC,
    #     'fossil_incentive': fossil * x['ETS'],
    #     'biogenic_incentive': biogenic * x['CRC'],
    #     'incentives': incentives,
    #     'bid': bid
    # }

    # FCCS = mcaptured*10**-6*3600 * x["FLH"] * fossil                           # [ktCO2/yr]
    # BECCS = mcaptured*10**-6*3600 * x["FLH"] * biogenic                        # [ktCO2/yr]

    bid, FCCS, BECCS, Ppenalty, Qpenalty, cost_details = [1,2,3,4,5,6]
    return bid, FCCS, BECCS, Ppenalty, Qpenalty, cost_details # [MWh/yr]

def WACCUS_EPR( 
    # constants
    question="granulates",
    CCUS="CCS", 
    plants_df=None, 
    shipping_df=None,
    truck_df=None,          
    compression_df=None,
    thermo_props=None,    
    SEK_to_EUR=0.091,

    # uncertainties
    mKN39 = 884393,         # [t/a] plastic products mappable under KN39 [IVL] high uncertainty + combine with policy lever uncertainty
    pKN39 = 46000,          # [SEK/tpl] [IVL]
    mgranulates = 1258597,  # [t/a] granulates [IVL] low uncertainty
    pgranulates = 13000,    # [SEK/tpl] [IVL]
    mbag = 5*10**-6,        # [t/bag] check plastic_mapping calculation
    pbag = 4,               # [SEK/bag]
    cfraction = 0.85,       # [-] cfraction of carbon in plastic [Isabel]
    circulated = 0.10,      # [-] fraction of granulates circulated, exempt from tax NOTE: no "market response" to tax

    FLH = 8000,
    capture_rate = 0.90,    # [-] 
    Ccontent = 0.298,       # [kgC/kgf]
    carbon_change = 0.10,   # [-] [-0.10,0.10] [Malder, 2023] fraction of biogenic carbon
    LHVf = 11,              # [MJ/kgf] [Hammar]
    LHVmethanol = 19.8,     # [MJ/kg] [Formelsamling]
    makeup = 0.584/1000,    # [m3/tCO2] [Kumar, 2023]

    q_reb = 3.5,            # [MJ/kgCO2]
    p_capture = 0.1,        # [MWh/tCO2] [Beiron, 2022]
    p_condition = 0.37,     # [MJ/kgCO2] [Kumar, 2023]
    n_is = 0.80,            # [-]
    n_electrolyzer = 0.699, # [MWH2/MWel] Table2.1 MSc Jacobsson & Palmgren (2025)
    q_electrolyzer = 0.154, # [MWth/MWel] [AEL tech, Fig2.1 MSc Jacobsson & Palmgren, 2025] OR [Danish Renwable Fuels 100MW AEC]
    q_synthesis = 0.087,    # [MWsteam/MWH2] about 0.08/(1-0.08)*QH2 [Danish Renewable Fuels Fig3, section 5.2 Methanol from Hydrogen and Carbon Dioxide]
    n_synthesis = 0.78,     # [MWmethanol/MWH2+steam]
    q_distill = 0.20,       # [MWth/MWH2+steam]
    COP = 3,                # [MWth/MWel]
    heat_optimism = 0.70,   # [0,1] assumed % of waste heat that can be recovered to DH

    CAPEXref_capture = 3550*0.09*1000, # [MNOK]->[kEUR] @400 ktCO2/yr [Gassnova, Demonstrasjon av Fullskala CO2-Håndtering - Rapport for Avsluttet Forprosjekt]
    CAPEXref_H2 = 550,                  # [kEUR/MWe] [Danish Agency Excel Renewable Fuels AEC100MW]
    CAPEXref_synthesis = 1.8749,        # [MEUR] [Danish Renewable Fuels PDF has a power function of CAPEX_synthesis. Fig4, p.186.] 
    CAPEXref_loading = 63000000,          # [SEK*] @150 ktCO2/yr excluding railway track [Koldioxid på tåg, 2024]
    CAPEXref_train = 86140000,          # [EUR*] an oversized train @15 wagons, cost = 4.98 *10**6 + 242*15 *10**3 [MSc Gunnarsson, 2025]
    k = 0.67,                           # [-] [Stenström, 2025] assumed economy-of-scale factor
    CEPCI = 900,                        # [-] [University of Manchester, 2025] applies to reference CAPEX values
    
    OPEXfix = 0.02,         # [-] % of base CAPEX, calculated from [Ramboll-Malmö, 2023]
    dr = 0.075,             # [-]
    t = 25,                 # [yr]
    camine = 44,            # [SEK/tCO2] [Ramboll-Malmö, 2023]
    celc = 60,              # [EUR/MWh]
    cheat = 0.75,           # [% of elc]
    CRC = 100,              # [EUR/tCO2]
    ETS = 80,               # [EUR/tCO2]
    pmethanol = 625,        # [EUR/t] [MSc Omar & Widgren, 2025]

    ship_uncertain = 0.10,  # [-] [-0.15,0.15]
    truck_uncertain = 0.10, # [-] [-0.15,0.15]
    train_uncertain = 0.10, # [-] [-0.15,0.15]
    stockholm = 1,          # [Mt/yr] [1,2,3] 
    malmo = 0.5,            # [Mt/yr] [0.5,1,2] 
    gothenburg = 0.5,       # [Mt/yr] [0.5,1,2] 
    storage = "oygarden",   # ["oygarden", "kalundborg"]

    # levers
    tax = 100,          # [EUR/tCO2] [50, 400] NOTE: explore discrete ranges => easier to visualize later
    recyclable = 0.15,  # [-] fraction of products possible to recycle mechanically (exempt from tax), determined by policy criteria
):
    # Store constants, uncertainties (converted to EUR or CEPCI), and levers
    c = {
        "shipping_df": shipping_df,
        "truck_df": truck_df,
        "compression_df": compression_df,
        "thermo_props": thermo_props,
    }
    x = {
        "mKN39": mKN39,
        "pKN39": pKN39 * SEK_to_EUR,                # [EUR/tpl]
        "mgranulates": mgranulates,
        "pgranulates": pgranulates * SEK_to_EUR,    # [EUR/tpl]
        "mbag": mbag,
        "pbag": pbag * SEK_to_EUR,                  # [EUR/bag]
        "cfraction": cfraction,
        "circulated": circulated,

        "FLH": FLH,
        "capture_rate": capture_rate,
        "Ccontent": Ccontent,
        "carbon_change": carbon_change,
        "LHVf": LHVf,
        "LHVmethanol": LHVmethanol,
        "makeup": makeup,

        "q_reb": q_reb,
        "p_capture": p_capture,
        "p_condition": p_condition,
        "n_is": n_is,
        "n_electrolyzer": n_electrolyzer,
        "q_electrolyzer": q_electrolyzer,
        "q_synthesis": q_synthesis,
        "n_synthesis": n_synthesis,
        "q_distill": q_distill,
        "COP": COP,
        "heat_optimism": heat_optimism,

        "CAPEXref_capture": CAPEXref_capture,
        "CAPEXref_H2": CAPEXref_H2,
        "CAPEXref_synthesis": CAPEXref_synthesis,
        "CAPEXref_loading": CAPEXref_loading * SEK_to_EUR, # [EUR @150ktCO2/yr]
        "CAPEXref_train": CAPEXref_train, 
        "k": k,
        "CEPCI": CEPCI / 600,                              # assuming 600 as the ref year for all CAPEX

        "OPEXfix": OPEXfix,
        "dr": dr,
        "t": t,
        "camine": camine * SEK_to_EUR, # [EUR/tCO2]
        "celc": celc,
        "cheat": cheat,
        "CRC": CRC,
        "ETS": ETS,
        "pmethanol": pmethanol,

        "ship_uncertain": ship_uncertain,
        "truck_uncertain": truck_uncertain,
        "train_uncertain": train_uncertain,
        "stockholm": stockholm,
        "malmo": malmo,
        "gothenburg": gothenburg,
        "storage": storage,
    }
    l = {
        "tax": tax,
        "recyclable": recyclable,
    }

    # Tax plastic products
    if question == "granulates":
        mass_taxed = mgranulates * (1 - circulated) * cfraction # [tC/yr]
        granulates_inc = tax * (cfraction*3.66) / (pgranulates*SEK_to_EUR) # [-]
        products_inc = tax * (cfraction*3.66) / (pKN39*SEK_to_EUR) 
        bag_inc = tax * (cfraction*3.66 * mbag) / (pbag*SEK_to_EUR) 
    elif question == "products":
        mass_taxed = mKN39 * (1 - recyclable) * cfraction # [tC/yr]
        granulates_inc = 0 # [-]
        products_inc = tax * (cfraction*3.66) / (pKN39*SEK_to_EUR) 
        bag_inc = tax * (cfraction*3.66 * mbag) / (pbag*SEK_to_EUR)
    elif question == "both":
        mass_taxed = mgranulates * (1 - circulated) * cfraction + mKN39 * (1 - recyclable) * cfraction # [tC/yr]
        granulates_inc = tax * (cfraction*3.66) / (pgranulates*SEK_to_EUR) # EUR/tCO2 * (tC/tpl*tCO2/tC) / (EUR/tpl) = EUR/tCO2 * (tCO2/tpl) / (EUR/tpl)
        products_inc = 2 * tax * (cfraction*3.66) / (pKN39*SEK_to_EUR) # [-] Motivate why this is not double taxation!
        bag_inc = 2 * tax * (cfraction*3.66 * mbag) / (pbag*SEK_to_EUR) # EUR/tCO2 * (tCO2/bag) / (EUR/bag)
    mass_CO2 = mass_taxed * 3.66        # [tCO2/yr]
    mass_taxed = mass_taxed / cfraction # [tpl/yr]
    fund = mass_CO2 * tax * 10**-6      # [MEUR/yr]

    # Create CCUS bids
    bids = []
    for _, plant in plants_df.iterrows():
        if CCUS == "CCS":
            bid, FCCS, BECCS, Ppenalty, Qpenalty, cost_details = plan_CCS(plant, c, x, l)
            bids.append({
                'type': 'CCS',
                'name': plant['Name'],
                'bid': bid,
                'FCCS': FCCS,
                'BECCS': BECCS,
                'Ppenalty': Ppenalty,
                'Qpenalty': Qpenalty,
                'cost_details': cost_details
            })

        # if CCUS == "CCU":
        #     FCCU, BCCU, bid, Ppenalty, Qpenalty, Qmethanol = plan_CCU(plant, x)
        #     bids.append({
        #         'type': 'CCU',
        #         'name': plant['Name'],
        #         'bid': bid,
        #         'FCCU': FCCU,
        #         'BCCU': BCCU,
        #         'Ppenalty': Ppenalty,
        #         'Qpenalty': Qpenalty,
        #         'Qmethanol': Qmethanol
        #     })
    bids.sort(key=lambda x: x['bid'])


    output = {
        # mass_taxed_granulates :
        # mass_taxed_products :

        # 'granulates_inc': granulates_inc,
        # 'products_inc': products_inc,
        # 'bag_inc': bag_inc,
        # 'total_FCCS': total_FCCS,
        # 'total_BECCS': total_BECCS,
        # 'total_FCCU': total_FCCU,
        # 'total_BCCU': total_BCCU,
        # 'total_Ppenalty': total_Ppenalty,
        # 'total_Qpenalty': total_Qpenalty,
        # 'total_Qmethanol': total_Qmethanol,
        # 'remaining_fund': remaining_fund,
        # 'bid_data': bids,
    }
    return output

if __name__ == "__main__":

    # Read data
    plants_df = pd.read_csv('data/plants.csv')
    plant_distances = pd.read_csv('data/plant_distances.csv')
    shipping_df = pd.read_csv('data/shipping_costs.csv')
    truck_df = pd.read_csv('data/truck_costs.csv')
    compression_df = pd.read_csv('data/compression_costs.csv')
    thermo_props = get_CoolProp()

    # Adjust dataframes: add plant distances, add 0.5Mt shipping costs, convert SEK to EUR
    SEK_to_EUR = 0.091
    plants_df = plants_df.merge(plant_distances, on='Name', how='left')
    shipping_df = shipping_adjustment(shipping_df, scaling=0.67, debug=False)
    
    cost_columns = [col for col in shipping_df.columns if col != 'distance']
    shipping_df[cost_columns] = shipping_df[cost_columns] * SEK_to_EUR
    truck_df["EUR/ton"] = truck_df["SEK/ton"] * SEK_to_EUR
    truck_df = truck_df.drop(columns=["SEK/ton"])

    # Run the model
    output = WACCUS_EPR(
        question="granulates",
        CCUS="CCS", 
        plants_df=plants_df, 
        shipping_df=shipping_df,
        truck_df=truck_df,
        compression_df=compression_df,
        thermo_props=thermo_props,    
        SEK_to_EUR=SEK_to_EUR,
    )
    
