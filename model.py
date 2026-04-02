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

    Qhex = c["q_hex"] * Qreb           # [MW] 
    Qdiff = Qdh_old - (Qdh + Qhex)     # [MW] 
    if Qdiff < 0:
        raise ValueError
    Whp = Qdiff / x["COP"]             # [MW] may exceed P (grid purchase)
    Qdh = Qdh + Qhex + Whp*x["COP"]    # restores Qdh to Qdh_old
    P -= Whp                           # negative P means grid power needed
    Ppenalty = (P_old - P) * FLH        # [MWh/yr] if negative P, the Ppenalty is inflated
    Qpenalty = (Qdh_old - Qdh) * FLH    # [MWh/yr]

    # Estimate CAPEX and on-site OPEX

    strike_price = 0
    FCCS = 0
    BECCS = 0
    Ppenalty = 0
    Qpenalty = 0
    cost_details = 0
    return strike_price, FCCS, BECCS, Ppenalty, Qpenalty, cost_details

def WACCUS_EPR(
    # [C] Constants
    EPR_design="Mitigation", # [Mitigation, Recovery, Circularity]
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

    capture_rate = 0.90,    # [-] 
    q_hex = 0.64,           # [MWth/MWreb] [Beiron, 2022] assumed heat exhange from capture plant

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
        "capture_rate": capture_rate,
        "q_hex": q_hex,
    }
    x = {
        "q_reb": q_reb,
        "p_capture": p_capture,
        "p_condition": p_condition,
        "COP": COP,
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
        for _, plant in plants_df.iterrows():
            strike_price, FCCS, BECCS, Ppenalty, Qpenalty, cost_details = plan_CCS(plant, c, x, l)

    elif EPR_design == "Recovery":
        print("Recovery not implemented yet")

    elif EPR_design == "Circularity":
        print("Circularity not implemented yet")
    
    # (3) Distribute subsidies and calculate KPIs

    output = 0
    return output

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
    output = WACCUS_EPR(
        EPR_design="Mitigation", # Mitigation, Recovery, Circularity 
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
    plt.show()
