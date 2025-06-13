# This is the model.py file
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import searoute as sr
import CoolProp.CoolProp as CP


def cost_transport():
    """
    Creates linear regression models for transport costs based on distance.
    Converts costs from SEK to EUR internally.
    
    Returns:
        dict: Dictionary containing linear regression models for different scenarios (all costs in EUR):
            Keys: 'optimist_1Mt', 'pessimist_1Mt', 'optimist_2Mt', 'pessimist_2Mt', 'optimist_3Mt', 'pessimist_3Mt'
    """
    # Read transport costs data
    df = pd.read_csv("data/transport_costs.csv")
    
    # Convert costs from SEK to EUR
    cost_columns = ['optimist_1Mt', 'pessimist_1Mt', 'optimist_2Mt', 
                   'pessimist_2Mt', 'optimist_3Mt', 'pessimist_3Mt']
    df[cost_columns] = df[cost_columns] * SEK_TO_EUR
    
    # Create linear regression models for each scenario
    transport_costs = {}
    r2_scores = {}
    
    for scenario in cost_columns:
        # Prepare data for regression
        X = df['distance'].values.reshape(-1, 1)
        y = df[scenario].values
        
        # Create and fit the model
        model = LinearRegression()
        model.fit(X, y)
        
        # Store the model
        transport_costs[scenario] = model
        
        # Calculate R² score
        r2_scores[scenario] = model.score(X, y)
    
    return transport_costs, r2_scores, df

def precalculate_sea_distances():
    """
    Read pre-calculated sea distances between hubs and destinations from CSV.
    
    Returns:
        dict: Dictionary with keys in format 'hub_destination' (e.g., 'stockholm_oygarden')
             and values as distances in kilometers
    """
    # Read pre-calculated distances
    df = pd.read_csv("data/hubs_destinations.csv")
    
    # Initialize dictionary to store distances
    distances = {}
    
    # Create distance entries for each hub-destination pair
    for _, row in df.iterrows():
        # Add Oygarden destination
        key_oygarden = f"{row['hub']}_oygarden"
        distances[key_oygarden] = row['destination_oygarden']
        
        # Add Kalundborg destination
        key_kalundborg = f"{row['hub']}_kalundborg"
        distances[key_kalundborg] = row['destination_kalundborg']
    
    return distances  # [km]

def sea_distance(origin, destination, distances_dict):
    """
    Get pre-calculated sea distance between hub and destination.
    
    Args:
        origin (str): Name of the origin hub
        destination (str): Name of the destination
        distances_dict (dict): Dictionary of pre-calculated distances
    
    Returns:
        float: Distance in kilometers
    """
    key = f"{origin.lower()}_{destination.lower()}"
    return distances_dict[key]

def estimate_CAPEX(mcaptured, x):
    """
    Estimate CAPEX for CO2 capture using power law model.
    
    Args:
        mcaptured: CO2 capture rate [kg/s]
        x: dictionary containing parameters
        
    Returns:
        CAPEX: Total CAPEX [kEUR]
        annualized_CAPEX: Annualized CAPEX [kEUR/yr]
        levelized_CAPEX: Levelized CAPEX [EUR/t]
    """
    # Convert from [kg/s] to [kt/yr]
    mannual = mcaptured/1000*3600 /1000 * x["FLH"]

    # Calculate CAPEX using power law model
    CAPEX = x["CAPEX_ref"] * (mannual / x["captured_ref"]) ** x["k"]  # [kEUR]
    CAPEX *= 1 + x["FOAK"]
    
    # Calculate annualized and levelized CAPEX
    CRF = (x["dr"] * (1 + x["dr"]) ** x["t"]) / ((1 + x["dr"]) ** x["t"] - 1)
    annualized_CAPEX = CAPEX * CRF  # [kEUR/yr]
    levelized_CAPEX = annualized_CAPEX / (mcaptured/1000*3600 * x["FLH"])  # [(kEUR/yr)/(t/yr)] -> [kEUR/t]
    
    return CAPEX, annualized_CAPEX, levelized_CAPEX

def compression_energy(mcaptured, T1, P1, gas_type='CO2', n_stages=4, pressure_ratio=3.0, Tdiff=30, thermo_props=None, etais=0.8, printing=False):
    """
    Calculate compression energy and cooling requirements for multi-stage compression.
    
    Args:
        mcaptured: Mass flow rate [kg/s]
        T1: Initial temperature [K]
        P1: Initial pressure [bar]
        gas_type: Type of gas ('CO2' or 'H2')
        n_stages: Number of compression stages
        pressure_ratio: Pressure ratio per stage
        Tdiff: Temperature difference for intercooling [K]
        thermo_props: Dictionary of thermodynamic properties
        etais: Isentropic efficiency [-]
        printing: Whether to print the results
    
    Returns:
        tuple: (Wcomp_list, Qcool_list, P_list, T_list)
            Wcomp_list: List of compression work for each stage [MW]
            Qcool_list: List of cooling requirements for each stage [MW]
            P_list: List of pressures at each stage [bar]
            T_list: List of temperatures at each stage [K]
    """
    Wcomp_list = []
    Qcool_list = []
    P_list = [P1]
    T_list = [T1]
    
    T = T1
    P = P1
    
    for stage in range(n_stages):
        # Get properties at current temperature
        kappa = get_property_at_temp(thermo_props, gas_type, T, 'kappa')
        cp_in = get_property_at_temp(thermo_props, gas_type, T, 'cp')
        
        # Calculate next pressure
        P_next = P * pressure_ratio
        P_list.append(P_next)
        
        # Calculate isentropic and actual temperatures
        T_isentropic = T * (P_next/P)**((kappa-1)/kappa)
        T_actual = T + (T_isentropic - T)/etais
        T_list.append(T_actual)
        
        # Get properties at actual temperature
        cp_out = get_property_at_temp(thermo_props, gas_type, T_actual, 'cp')
        
        # Calculate work and cooling
        Wcomp = mcaptured * (cp_in + cp_out)/2 * (T_actual - T)  # [kJ/s]
        
        # Calculate cooling only if not the last stage
        Qcool = 0 if stage == n_stages - 1 else mcaptured * cp_out * (T_actual - (T + Tdiff))    # [kJ/s]
        
        # Store results
        Wcomp_list.append(Wcomp/1000)  # Convert to MW
        Qcool_list.append(Qcool/1000)  # Convert to MW
        
        # Update temperature for next stage
        T = T + Tdiff
        P = P_next
    
    if printing:
        # Print summary table
        print(f"\n{gas_type} Compression Summary:")
        print("Stage | Pressure [bar] | Temperature [°C] | Work [MW] | Cooling [MW]")
        print("------|---------------|------------------|-----------|-------------")
        for i in range(len(Wcomp_list)):
            print(f"{i+1:5d} | {P_list[i]:13.1f} | {T_list[i]-273.15:16.1f} | {Wcomp_list[i]:9.1f} | {Qcool_list[i]:11.1f}")
        print(f"Final | {P_list[-1]:13.1f} | {T_list[-1]-273.15:16.1f} | {'-':9s} | {'-':11s}")
        print(f"\nTotal compression work: {sum(Wcomp_list):.1f} MW")
        print(f"Total cooling required: {sum(Qcool_list):.1f} MW")
    
    return Wcomp_list, Qcool_list, P_list, T_list

def compression_cost(Wcomp_list, gas_type='CO2', printing=False):
    """
    Calculate the cost of compression stages using coefficients from Deng's paper.
    
    Args:
        Wcomp_list: List of compression work for each stage [MW]
        gas_type: Type of gas ('CO2' or 'H2')
    
    Returns:
        tuple: (total_cost, stage_costs)
            total_cost: Total cost of compression [EUR]
            stage_costs: List of costs for each stage [EUR]
    """
    # Read coefficients from CSV
    df = pd.read_csv("data/compression_costs.csv", index_col=0)
    
    # Initialize lists
    stage_costs = []
    
    # Calculate cost for each stage
    for i, Wstage in enumerate(Wcomp_list):
        # Convert MW to kW
        Wstage_kW = Wstage * 1000

        # Get coefficients for this stage
        a = df.loc['Coefficient a', f'Stage {i+1}']
        b = df.loc['Coefficient b', f'Stage {i+1}']
        c = df.loc['Coefficient c', f'Stage {i+1}']
        
        # Calculate cost using these equations [Deng, 2019]:
        if i == 3:  # 4th stage (0-based indexing) has different equation
            cost = a + b * Wstage_kW + c * Wstage_kW**0.5
        else:
            cost = a + b * Wstage_kW**1.5 + c * Wstage_kW**2
        stage_costs.append(cost)
    
    total_cost = sum(stage_costs)
    
    # Print results
    if printing:
        print(f"\n{gas_type} Compression Costs:")
        print("Stage | Work [MW] | Cost [EUR]")
        print("------|-----------|------------")
        for i, (Wstage, cost) in enumerate(zip(Wcomp_list, stage_costs)):
            print(f"{i+1:5d} | {Wstage:9.1f} | {cost:10.0f}")
        print(f"Total | {sum(Wcomp_list):9.1f} | {total_cost:10.0f}")
        
    return total_cost, stage_costs

def plan_CCU(plant, x):
    # burn fuel
    mfuel = plant["Qwaste"] / (x["LHV"]/3600) /3600  # [kgf/s]
    mCO2 = mfuel* x["Ccontent"] * 44/12              # [kgCO2/s]
    Vfluegas = x["vfluegas"] * mfuel                 # [Nm3/s]

    # capture and compress CO2
    mcaptured = mCO2 * 0.90                          # [kgCO2/s]
    Qreb = mcaptured * x["qreb"]                     # [MW]
    Pcapture = 0.1 * mcaptured/1000*3600             # [MW] [Beiron, 2022]
    Qhex = 0.64 * Qreb                               # [MW] [Beiron, 2022]
    
    T1 = 40 + 273.15  # Initial temperature [K] [Deng, 2019]
    P1 = 1.0         # Initial pressure [bar]
    Wcomp_list, Qcool_list, P_list, T_list = compression_energy(
        mcaptured=mcaptured,
        T1=T1,
        P1=P1,
        gas_type='CO2',
        n_stages=4,
        pressure_ratio=3.0,
        Tdiff=30,
        thermo_props=x["thermo_props"],
        etais=x["etais"],
        printing=False
    )
    Wcomp_CO2 = sum(Wcomp_list)  # [MW]
    cost_CO2, _ = compression_cost(Wcomp_list, 'CO2', printing=False)
    Qcool_CO2 = sum(Qcool_list)  # [MW]

    print("Could recover the Qcool from compression using HEXs")
    print("Do a recovery strategy across all equipment later! Include HEX costs for both CCU/CCS")

    # produce H2 from AEL electrolyzer
    Hi = 241.82             # [kJ/molH2] [Formelsamling]
    nCO2 = mcaptured/44     # [kmol/s]
    nH2 = nCO2 * 3          # [kmol/s]
    QH2 = nH2*1000 * Hi     # [kW] 
    PH2 = QH2/0.699         # [kW] 
    Qrec_H2 = 0.223 * PH2   # [kW] [AEL tech, Fig2.1 MSc Jacobsson & Palmgren, 2025] OR [Danish Renwable Fuels 100MW AEC]

    # compress the H2
    mH2 = nH2 * 2           # [kg/s] 
    T1 = 75 + 273.15        # [AEL tech, Fig2.1 MSc Jacobsson & Palmgren, 2025]
    P1 = 20                 # Initial pressure [bar], assumed based on [Danish]
    Wcomp_list, Qcool_list, P_list, T_list = compression_energy(
        mcaptured=mH2,
        T1=T1,
        P1=P1,
        gas_type='H2',
        n_stages=2,
        pressure_ratio=2.0,
        Tdiff=10,
        thermo_props=x["thermo_props"],
        etais=x["etais"],
        printing=False
    )
    Wcomp_H2 = sum(Wcomp_list)  # [MW]
    cost_H2, _ = compression_cost(Wcomp_list, 'H2', printing=False)
    Qcool_H2 = sum(Qcool_list)  # [MW]

    # produce methanol and extra DH
    Qmethanol_raw = QH2/1000 # [MW] assumed no steam need for synthesis...
    Qmethanol = 0.78*Qmethanol_raw # [MW] [Danish Renewable Fuels Fig3]
    Qdistill = 0.20*Qmethanol_raw # [MW] 

    # penalize CHP and recover Qdh
    P = plant["P"] * (1 - Qreb/plant["Qwaste"])      # assuming live steam is used for reboiler
    P = P - Pcapture - PH2/1000 - Wcomp_CO2 - Wcomp_H2

    Qdh = plant["Qdh"] * (1 - Qreb/plant["Qwaste"])
    Qdh = Qdh + Qhex + Qcool_CO2 + Qrec_H2/1000 + Qcool_H2 + Qdistill

    Ppenalty = (plant["P"] - P) * x["FLH"]           # [MWh/yr] probably very positive
    Qpenalty = (plant["Qdh"] - Qdh) * x["FLH"]       # [MWh/yr] probably negative
    Qmethanol = Qmethanol * x["FLH"]                 # [MWh/yr] positive
    print(Ppenalty, Qpenalty, Qmethanol)

    # Estimate CAPEX of capture plant
    CAPEX, annualized_CAPEX, levelized_CAPEX = estimate_CAPEX(mcaptured, x)  # [kEUR, kEUR/yr, kEUR/tCO2] NOTE: Includes compression/liq CAPEX...
    CAPEX_H2 = 550/1000 * PH2 # [kEUR] [Danish]
    OPEX_H2 = 0.04 * CAPEX_H2 # [kEUR/yr] 
    CAPEX_synthesis = 0.7*1000 * Qmethanol/x["FLH"] # [kEUR] [Beiron, Grahn]
    OPEX_synthesis = 0.05 * CAPEX_synthesis # [kEUR/yr] NOTE: No CAPEX for destillation section here
    print("TODO: LEVELIZE ALL CAPEX AND OPEX BASED ON THE TONS OF CO2 NEUTRALIZED - THAT IS THE BID")

    CCU = 1
    bid = 2
    Ppenalty = 3
    Qpenalty = 4
    return CCU, bid, Ppenalty, Qpenalty

def plan_CCS(plant, x, transport_costs, sea_distances):
    # burn fuel
    mfuel = plant["Qwaste"] / (x["LHV"]/3600) /3600  # [kgf/s]
    mCO2 = mfuel* x["Ccontent"] * 44/12              # [kgCO2/s]
    Vfluegas = x["vfluegas"] * mfuel                 # [Nm3/s]

    # capture and condition CO2
    mcaptured = mCO2 * 0.90                          # [kgCO2/s]
    Qreb = mcaptured * x["qreb"]                     # [MW]
    Pcapture = 0.1 * mcaptured/1000*3600             # [MW] [Beiron, 2022]
    Pcondition = 0.37 * mcaptured                    # [MW] [Kumar, 2023] incl. CO2 conditioning

    # penalize CHP
    P = plant["P"] * (1 - Qreb/plant["Qwaste"])      # assuming live steam is used for reboiler
    P = P - Pcapture - Pcondition
    Qdh = plant["Qdh"] * (1 - Qreb/plant["Qwaste"])

    # recover heat up to 100 % of original DH - use whatever power is available for HP
    Qhex = 0.64 * Qreb                               # [MW] [Beiron, 2022]
    Qdiff = plant["Qdh"] - (Qdh + Qhex)
    if Qdiff < 0:
        raise ValueError
    else:
        Whp = Qdiff / x["COP"]
        if Whp > P: 
            Whp = P
    Qdh = Qdh + Qhex + Whp*x["COP"]
    P -= Whp
    Ppenalty = (plant["P"] - P) * x["FLH"]           # [MWh/yr]
    Qpenalty = (plant["Qdh"] - Qdh) * x["FLH"]       # [MWh/yr]

    # Estimate CAPEX
    CAPEX, annualized_CAPEX, levelized_CAPEX = estimate_CAPEX(mcaptured, x)  # [kEUR, kEUR/yr, kEUR/t]

    # Calculate transport cost using sea distance
    distance = sea_distance(plant["hub"], x["destination"], sea_distances)  # [km]
    print("<<< Missing Truck/Rail/Harbor costs >>>")
    print(plant["hub"], x["destination"])
    
    hub_value = x[plant["hub"].lower()]              # Get the value (1, 2, or 3) for this hub
    scenario = f"{hub_value}Mt"                      # Convert to scenario name (1Mt, 2Mt, or 3Mt)
    
    scenario_type = "optimist" if x["optimism"] else "pessimist"
    scenario_key = f"{scenario_type}_{scenario}"
    print(scenario_key)
    
    transport_cost = transport_costs[scenario_key].predict([[distance]])[0]  # [EUR/t]
    print(f"Transport cost at {distance} km: {transport_cost:.2f} EUR/t")

    # Estimate OPEX
    OPEXfix = (CAPEX*1000 * x["OPEXfix"]) / (mcaptured/1000*3600 * x["FLH"])  # [EUR/t]
    OPEXmakeup = x["makeup"] * x["camine"]                                    # [EUR/t]
    OPEXenergy = (Ppenalty*x["celc"] + Qpenalty*x["celc"]*x["cheat"]) / (mcaptured/1000*3600 * x["FLH"])  # [EUR/t]
    OPEX = OPEXfix + OPEXmakeup + OPEXenergy                                  # [EUR/t]

    # Construct a reversed auction bid
    CAC = levelized_CAPEX*1000 + OPEX + transport_cost                        # [EUR/t]
    fossil = plant["Fossil"] / plant["Total"]                                 # [tfossil/t] share of fossil CO2
    biogenic = 1 - fossil                                                     # [tbiogenic/t] share of biogenic CO2
    incentives = fossil * x["ETS"] + biogenic * x["CRC"]                      # [EUR/t]
  
    bid = CAC - incentives                                                     # [EUR/t]

    FCCS = mcaptured*10**-6*3600 * x["FLH"] * fossil                           # [ktCO2/yr]
    BECCS = mcaptured*10**-6*3600 * x["FLH"] * biogenic                        # [ktCO2/yr]

    return FCCS, BECCS, bid, Ppenalty, Qpenalty

def plot_transport_costs(transport_costs, r2_scores, df, show_plot=False, max_distance=3000):
    """
    Plot transport costs with linear regression fits.
    
    Args:
        transport_costs (dict): Dictionary of linear regression models
        r2_scores (dict): Dictionary of R² scores for each model
        df (pd.DataFrame): Original data
        show_plot (bool): Whether to display the plot
        max_distance (float): Maximum distance to show in the plot [km]
    
    Returns:
        tuple: (fig, ax) matplotlib figure and axis objects
    """
    # Create distance range for x-axis
    distances = np.linspace(0, max_distance, 100)
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot each scenario
    scenarios = ['1Mt', '2Mt', '3Mt']
    colors = ['blue', 'green', 'red']
    
    for i, scenario in enumerate(scenarios):
        # Plot data points
        ax.scatter(df['distance'], df[f'optimist_{scenario}'], 
                  color=colors[i], marker='o', alpha=0.3, label=f'{scenario} - Optimistic (data)')
        ax.scatter(df['distance'], df[f'pessimist_{scenario}'], 
                  color=colors[i], marker='s', alpha=0.3, label=f'{scenario} - Pessimistic (data)')
        
        # Plot regression lines
        ax.plot(distances, transport_costs[f'optimist_{scenario}'].predict(distances.reshape(-1, 1)), 
               color=colors[i], linestyle='-', 
               label=f'{scenario} - Optimistic (R²={r2_scores[f"optimist_{scenario}"]:.3f})')
        ax.plot(distances, transport_costs[f'pessimist_{scenario}'].predict(distances.reshape(-1, 1)), 
               color=colors[i], linestyle='--', 
               label=f'{scenario} - Pessimistic (R²={r2_scores[f"pessimist_{scenario}"]:.3f})')
    
    ax.set_xlabel('Distance [km]')
    ax.set_ylabel('Transport Cost [EUR/t]')
    ax.set_title(f'Transport Costs vs Distance for Different Scenarios\nwith Linear Regression Fits (extended to {max_distance} km)')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Add a note about the regression line extension
    fig.text(0.02, 0.02, 'Note: Regression lines extend beyond the data range and can be used for predictions at any distance', 
             fontsize=8, style='italic')
    
    plt.tight_layout()
    
    if show_plot:
        plt.show()
    
    return fig, ax

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points on the earth (specified in decimal degrees)
    
    Args:
        lat1, lon1: Latitude and longitude of first point
        lat2, lon2: Latitude and longitude of second point
    
    Returns:
        float: Distance in kilometers
    """
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371  # Radius of earth in kilometers
    
    return c * r

def assign_hub(plants_df):
    """
    Assign each plant to its nearest hub based on geographical coordinates.
    
    Args:
        plants_df (pd.DataFrame): DataFrame containing plant information with 'Latitude' and 'Longitude' columns
    
    Returns:
        pd.DataFrame: Updated plants DataFrame with new 'hub' column
    """
    # Read hubs data
    hubs_df = pd.read_csv("data/hubs.csv")
    
    # Create a copy of the plants DataFrame to avoid modifying the original
    plants = plants_df.copy()
    
    # Initialize hub column
    plants['hub'] = None
    plants['distance_to_hub'] = np.inf
    
    # For each plant, find the nearest hub
    for idx, plant in plants.iterrows():
        min_distance = np.inf
        nearest_hub = None
        
        for _, hub in hubs_df.iterrows():
            distance = haversine_distance(
                plant['Latitude'], plant['Longitude'],
                hub['lat'], hub['lon']
            )
            
            if distance < min_distance:
                min_distance = distance
                nearest_hub = hub['hub']
        
        plants.at[idx, 'hub'] = nearest_hub
        plants.at[idx, 'distance_to_hub'] = min_distance
    
    return plants

def get_thermo_properties():
    """
    Creates lookup tables for thermodynamic properties of CO2 and H2.
    
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

def get_property_at_temp(thermo_props, gas, T, property_name):
    """
    Get thermodynamic property at a specific temperature using interpolation.
    
    Args:
        thermo_props: Dictionary of thermodynamic properties
        gas: 'CO2' or 'H2'
        T: Temperature in Kelvin
        property_name: 'kappa', 'cp', or 'cv'
    
    Returns:
        float: Interpolated property value
    """
    # Ensure temperature is within bounds
    T = np.clip(T, thermo_props[gas]['temperatures'][0], thermo_props[gas]['temperatures'][-1])
    
    # Use numpy's interpolation
    return np.interp(T, thermo_props[gas]['temperatures'], thermo_props[gas][property_name])

def WACCUS_EPR( 
    # uncertainties
    mKN39 = 884393,     # [t/a] plastic products mappable under KN39 [IVL]
    pKN39 = 46000,      # [SEK/tpl] [IVL]
    recyclable = 0.15,  # [-] fraction of simple products possible to recycle mechanically

    LHV = 11,           # [MJ/kgf] [Hammar]
    vfluegas = 4.7,     # [Nm3/kgf]
    Ccontent = 0.298,   # [kgC/kgf]
    fossil = 0.40,      # [-] NOTE: assumption not needed: emissions data is available and used
    qreb = 3.5,         # [MJ/kgCO2]
    COP = 3,

    FLH = 8000,
    dr = 0.075,
    t = 25,
    FOAK = 0.45/2,      # [-] [Beiron, 2024] applies to CO2 capture and conditioning 
    OPEXfix = 0.05,     # [-] [Beiron, 2024] % of CAPEX 
    camine = 2000,      # [EUR/m3] [Beiron, 2024]
    celc = 60,          # [EUR/MWh]
    cheat = 0.80,       # [% of elc]
    CRC = 100,          # [EUR/tCO2]
    ETS = 80,           # [EUR/tCO2]

    lulea = 1,          # [1,2,3] [Mt/yr]
    sundsvall = 1,      # [1,2,3] [Mt/yr]
    stockholm = 1,      # [1,2,3] [Mt/yr]
    malmo = 1,          # [1,2,3] [Mt/yr]
    gothenburg = 1,     # [1,2,3] [Mt/yr]
    optimism = False,   # [True, False]
    destination = "oygarden",  # ["oygarden", "kalundborg"]

    # levers 
    tax = [800,1600,2400,3200],
    groups = [0,1,2,3],

    # constants
    plants = None,
    case = ["CCUS"],    # ["CCUS", "CCU", "CCS"] conduct analysis separately for the three cases
    k = 0.6857,         # [-] [Stenström, 2025]
    CAPEX_ref = 3715 * 87,  # [MNOK] -> [kEUR] NOTE: can pick other CELSIO CAPEX from source:[Gassnova, Demonstrasjon av Fullskala CO2-Håndtering - Rapport for Avsluttet Forprosjekt]
    captured_ref = 400,     # [ktCO2/yr]
    transport_costs = None,  # Dictionary of transport cost interpolation functions
    sea_distances = None,    # Dictionary of pre-calculated sea distances
    thermo_props = None,     # Dictionary of thermodynamic properties

    # Global assumptions
    SEK_TO_EUR = 0.091,     # [EUR/SEK] Exchange rate
    makeup = 0.584/1000,    # [m3/tCO2] [Kumar, 2023]
    etais = 0.80,           # [-]
):
    x = {
        "mKN39": mKN39,
        "recyclable": recyclable,
        "pKN39": pKN39,
        
        "LHV": LHV,
        "vfluegas": vfluegas,
        "Ccontent": Ccontent,
        "fossil": fossil,
        "qreb": qreb,
        "COP": COP,

        "FLH": FLH,
        "dr": dr,
        "t": t,
        "FOAK": FOAK,
        "OPEXfix": OPEXfix,
        "camine": camine,
        "celc": celc,
        "cheat": cheat,
        "CRC": CRC,
        "ETS": ETS,

        "tax": tax,
        "groups": groups,
        "k": k,
        "CAPEX_ref": CAPEX_ref,
        "captured_ref": captured_ref,

        "lulea": lulea,
        "sundsvall": sundsvall,
        "stockholm": stockholm,
        "malmo": malmo,
        "gothenburg": gothenburg,
        "optimism": optimism,
        "destination": destination,
        "makeup": makeup,
        "etais": etais,
        "thermo_props": thermo_props,  # Add thermodynamic properties to x dictionary
    }

    # RQ1: tax revenues

    # RQ2: subsidy costs
    if case == "CCUS":
        CCU_names = ["Handeloverket"]
        CCS_names = ["Renova","SAKAB","Filbornaverket","Garstadverket","Sjolunda","Handeloverket","Bristaverket","Bolanderna"]
        CCS_names = []
    elif case == "CCU":
        CCU_names = ["All"]
        CCS_names = None
    elif case == "CCS":
        CCU_names = None
        CCS_names = ["All"]

    for _, plant in plants.iterrows():
        if plant["Name"] in CCS_names:
            FCCS, BECCS, bid, Ppenalty, Qpenalty = plan_CCS(plant, x, transport_costs, sea_distances)
            print(plant["Name"], " ----> ",FCCS, BECCS, bid, Ppenalty, Qpenalty)

        if plant["Name"] in CCU_names:
            CCU, bid, Ppenalty, Qpenalty = plan_CCU(plant, x)

    # RQ2: reversed auction simulation
    # NOTE: need to add a check for bid < 0

    # RQ3: product cost increases
    # KN39, cheese, syringe, panel, cable, tire, pedal 

    output = 1
    return output

if __name__ == "__main__":
    # finding k exponent of CAPEX=CAPEX_ref⋅(mCO2/mCO2_ref)^k by linear regression: y = k * x
    df = pd.read_csv("data/capture_costs.csv")
    captured_ref = df["captured"].mean()
    CAPEX_ref = df["CAPEX"].mean()

    x = np.log(df["captured"] / captured_ref)
    y = np.log(df["CAPEX"] / CAPEX_ref)
    model = LinearRegression().fit(x.values.reshape(-1, 1), y.values)
    k = model.coef_[0]
    
    # Get sea transport cost functions and sea distances
    SEK_TO_EUR = 0.091  # [EUR/SEK] Exchange rate
    transport_costs, r2_scores, df = cost_transport()
    sea_distances = precalculate_sea_distances()
    thermo_props = get_thermo_properties()  # Get thermodynamic properties
    fig, ax = plot_transport_costs(transport_costs, r2_scores, df, show_plot=False)
    
    # reading plant data and assign these to transport hubs
    plants = pd.read_csv("data/plants.csv")
    plants = assign_hub(plants)
    print("\nPlant to Hub Assignments:")
    print(plants[['Name', 'hub', 'distance_to_hub']].to_string())
    
    # Run the model
    output = WACCUS_EPR(
        plants=plants, 
        k=k, 
        case="CCUS", 
        transport_costs=transport_costs,
        sea_distances=sea_distances,  # Pass pre-calculated distances dict
        thermo_props=thermo_props     # Pass thermodynamic properties
    )

