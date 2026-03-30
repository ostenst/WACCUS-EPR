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

def truck_cost(annual_CO2, distance, trucks_df, x):
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
    return truck_cost * (1 + x["truck_uncertain"]) # [EUR/tCO2]

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
    Mcomp = 1.2                             # [-] margin
    Cfcomp = 568                            # [EUR/m2]
    CAPEX_pipeline = (2*np.pi*((VCO2/vCO2)/np.pi)**0.5 * (pipeline_length*Mcomp)) * Cfcomp # [EUR2015] [Tharun, 2025]
    CAPEX_pipeline = CAPEX_pipeline * x["CEPCI"] / 1000 # [kEUR]
    CAPEXlev_pipeline = levelize_kEUR(CAPEX_pipeline, annual_CO2, x) # [EUR/tCO2]
    return CAPEXlev_pipeline

def train_cost(annual_CO2, rail_distance, x):
    """Rail distance in [km], returns levelized CAPEX in [EUR/tCO2]"""
    speed = 60                                                      # [km/h]
    unload_time = 5                                                 # [h]
    cycle_time = (rail_distance/speed + unload_time) * 2            # [h]
    train_capacity = 15 * 60                                        # [tCO2/train] @15 wagons
    train_capacity /= cycle_time                                    # [tCO2/h] 

    CAPEX_train = x["CAPEXref_train"]                               # [EUR]
    CAPEXlev_train = levelize_kEUR(CAPEX_train/1000, annual_CO2, x) # [EUR/tCO2]
    OPEX_train = x["OPEXfix"]*CAPEX_train + 0.0269*(train_capacity*(rail_distance*2*365)) # [EUR/yr] 1 roundtrip per day is more than enough!
    OPEX_train = OPEX_train / annual_CO2                            # [EUR/tCO2]

    return (CAPEXlev_train + OPEX_train) * (1 + x["train_uncertain"]) # [EUR/tCO2]

def ship_cost(annual_CO2, city, ship_distance, shipping_df, x, optimism="optimist"):
    """Ship distance in [km], returns levelized CAPEX in [EUR/tCO2]"""
    if city == "Stockholm":
        mt_capacity = x["stockholm"]
    elif city == "Malmo":
        mt_capacity = x["malmo"]
    elif city == "Goteborg":
        mt_capacity = x["gothenburg"]
    else:
        mt_capacity = 0.5

    column_name = f"{optimism}_{mt_capacity}Mt" # Determines what shipping capacity is used
    
    # Sort by distance for proper interpolation
    sorted_df = shipping_df.sort_values('distance')
    distances = sorted_df['distance'].values
    costs = sorted_df[column_name].values
    result = np.interp(ship_distance, distances, costs)

    shipping_cost = result * (1 + x["ship_uncertain"]) # [EUR/tCO2]
    return shipping_cost

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

def compression_energy(mcaptured, T1, P1, gas_type='CO2', n_stages=4, pressure_ratio=3.0, Tdiff=30, thermo_props=None, n_is=0.8, printing=False):
    """
    Calculate compression energy and cooling requirements for multi-stage compression.
    # Setting target p and T according to [Beiron, 2025 (unpublished)]
    Args:
        mcaptured: Mass flow rate [kg/s]
        T1: Initial temperature [K]
        P1: Initial pressure [bar]
        gas_type: Type of gas ('CO2' or 'H2')
        n_stages: Number of compression stages
        pressure_ratio: Pressure ratio per stage
        Tdiff: Temperature difference for intercooling [K]
        thermo_props: Dictionary of thermodynamic properties
        n_is: Isentropic efficiency [-]
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
        T_actual = T + (T_isentropic - T)/n_is
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
        T = T + Tdiff # NOTE: We neglect that temperatures wouldn't increase, but needs intercooling, and then final heating do synthesis temperature.
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

def plan_CCS(plant, c, x, l):

    # Burn fuel and capture/condition CO2
    mfuel = plant["Qwaste"] / (x["LHVf"]/3600) /3600    # [kgf/s]
    mCO2 = mfuel* x["Ccontent"] * 44/12                 # [kgCO2/s]
    mcaptured = mCO2 * x["capture_rate"]                # [kgCO2/s]
    Qreb = mcaptured * x["q_reb"]                       # [MW]
    Pcapture = x["p_capture"] * mcaptured/1000*3600     # [MW] 
    Pcondition = x["p_condition"] * mcaptured           # [MW]  

    # Penalize CHP and recover heat up to 100 % of original DH - use whatever power is available for HP
    P = plant["P"] * (1 - Qreb/plant["Qwaste"])         # assuming live steam is used for reboiler
    P = P - Pcapture - Pcondition
    Qdh = plant["Qdh"] * (1 - Qreb/plant["Qwaste"])

    Qhex = 0.64 * Qreb                                  # [MW] [Beiron, 2022]
    Qdiff = plant["Qdh"] - (Qdh + Qhex)
    if Qdiff < 0:
        raise ValueError
    else:
        Whp = Qdiff / x["COP"]  
        if Whp > P and P > 0:   
            Whp = P
        elif P < 0:             
            Whp = 0             
    Qdh = Qdh + Qhex + Whp*x["COP"]               # Qfgc is not impacted!
    P -= Whp
    Ppenalty = (plant["P"] - P) * x["FLH"]        # [MWh/yr]
    Qpenalty = (plant["Qdh"] - Qdh) * x["FLH"]    # [MWh/yr]

    # Estimate CAPEX and on-site OPEX
    annual_CO2 = mcaptured/1000*3600 * x["FLH"]   # [tCO2/yr]
    CAPEX_capture = x["CAPEXref_capture"] * (annual_CO2/(400*10**3)) ** x["k"] * x["CEPCI"] # [kEUR]
    CAPEXlev_capture = levelize_kEUR(CAPEX_capture, annual_CO2, x)                  # [EUR/tCO2]
    CAPEX_HP = x["CAPEXref_HP"] * Whp*x["COP"] *1000 * x["CEPCI"]                   # [kEUR] neglect HEX costs
    CAPEXlev_HP = levelize_kEUR(CAPEX_HP, annual_CO2, x)                            # [EUR/tCO2]

    OPEXfix = (CAPEX_capture*1000 * x["OPEXfix"]) / annual_CO2                      # [EUR/tCO2] 
    OPEXmakeup = x["camine"]                                                        # [EUR/tCO2]
    OPEXenergy = (Ppenalty*x["celc"] + Qpenalty*x["celc"]*x["cheat"]) / annual_CO2  # [EUR/tCO2]
    OPEX = OPEXfix + OPEXmakeup + OPEXenergy   

    # Calculate transport and storage costs
    costs_transport = []
    if plant['Truck_distance'] is not None and not pd.isna(plant['Truck_distance']):
        cost_loading = loading_cost(annual_CO2, x)                                      # [EUR/tCO2]
        cost_truck = truck_cost(annual_CO2, plant['Truck_distance'], c["truck_df"], x)  # [EUR/tCO2]
        costs_transport.append(cost_loading)
        costs_transport.append(cost_truck)

    if plant['Pipeline_distance'] is not None and not pd.isna(plant['Pipeline_distance']):
        cost_pipeline = pipeline_cost(annual_CO2, plant['Pipeline_distance'], x)        # [EUR/tCO2]
        costs_transport.append(cost_pipeline)

    if plant['Rail_distance'] is not None and not pd.isna(plant['Rail_distance']):
        cost_loading = loading_cost(annual_CO2, x)                                      # [EUR/tCO2]
        cost_rail = train_cost(annual_CO2, plant['Rail_distance'], x)                   # [EUR/tCO2]
        costs_transport.append(cost_loading)
        costs_transport.append(cost_rail)

    if plant['Oygarden_distance'] is not None and not pd.isna(plant['Oygarden_distance']):
        if x["storage"] == "oygarden":
            distance = plant['Oygarden_distance']
        if x["storage"] == "kalundborg":
            distance = plant['Kalundborg_distance']
        cost_ship = ship_cost(annual_CO2, plant["City"], distance, c["shipping_df"], x) # [EUR/tCO2]
        costs_transport.append(cost_ship)

    transport_cost = sum(costs_transport)

    # Construct a reversed auction bid
    CAC = CAPEXlev_capture + CAPEXlev_HP + OPEX + transport_cost    # [EUR/tCO2]

    fossil = plant["Fossil"] / plant["Total"]                       # [tfossil/t] 
    biogenic = 1 - fossil                                           # [tbiogenic/t] 
    fossil -= x["carbon_change"]
    biogenic += x["carbon_change"]
    FCCS = annual_CO2 * fossil /1000                                # [ktCO2/yr]
    BECCS = annual_CO2 * biogenic /1000                             # [ktCO2/yr]
    # incentives = fossil * x["ETS"] + biogenic * x["CRC"]          # [EUR/tCO2] not needed if bid consists of strike price

    if x["CRC"] < x["ETS"]:
        x["CRC"] = x["ETS"] # In such cases (~50%), CRCs are assumed integrated into the ETS
    strike_price = CAC - biogenic * x["CRC"]                        # [EUR/tCO2] relative to a fossil ETS reference price
    strike_price = strike_price * (1 + c["profit"])

    cost_details = {
        'OPEXfix': OPEXfix,
        'OPEXmakeup': OPEXmakeup,
        'OPEXenergy': OPEXenergy,
        'OPEX': OPEX,
        'CAPEXlev': CAPEXlev_capture + CAPEXlev_HP,
        'transport_cost': transport_cost,
        'CAC': CAC,
        'fossil_incentive': fossil * x['ETS'],
        'biogenic_incentive': biogenic * x['CRC'],
        # 'incentives': incentives,
        'strike_price': strike_price
    }

    return strike_price, FCCS, BECCS, Ppenalty, Qpenalty, cost_details # [MWh/yr]

def plan_CCU(plant, c, x, l, plot_single=False):
    
    # Burn fuel and capture/condition CO2
    mfuel = plant["Qwaste"] / (x["LHVf"]/3600) /3600    # [kgf/s]
    mCO2 = mfuel* x["Ccontent"] * 44/12                 # [kgCO2/s]
    mcaptured = mCO2 * x["capture_rate"]                # [kgCO2/s]
    annual_CO2 = mcaptured/1000*3600 * x["FLH"]         # [tCO2/yr]
    Qreb = mcaptured * x["q_reb"]                       # [MW]
    Pcapture = x["p_capture"] * mcaptured/1000*3600     # [MW] 
    Qhex = 0.64 * Qreb                                  # [MW] [Beiron, 2022]

    # Compress CO2 and size CAPEX
    T1 = 40 + 273.15  # Initial temperature [K] and pressure [bar] [Deng, 2019]
    P1 = 1.0         
    Wcomp_list, Qcool_list, P_list, T_list = compression_energy( 
        mcaptured=mcaptured,
        T1=T1,
        P1=P1,
        gas_type='CO2',
        n_stages=3,
        pressure_ratio=3.8,
        Tdiff=30,
        thermo_props=c["thermo_props"],
        n_is=x["n_is"],
        printing=False
    )
    Wcomp_CO2 = sum(Wcomp_list)  # [MW]
    Qcool_CO2 = sum(Qcool_list)  # [MW]
    CAPEX_compress_CO2, _ = compression_cost(Wcomp_list, 'CO2', printing=False) # [EUR]
    CAPEXcomp_CO2 = levelize_kEUR(CAPEX_compress_CO2/1000, annual_CO2, x)        # [EUR/tCO2]
    
    # Produce H2 from AEL electrolyzer
    Hi = 241.82                           # [kJ/molH2]
    nCO2 = mcaptured/44                   # [kmol/s]
    nH2 = nCO2 * 3                        # [kmol/s] synthesis stoichiometry
    mH2 = nH2 * 2                         # [kg/s] 
    QH2 = nH2 * Hi                        # [MW] 
    PH2 = QH2/x["n_electrolyzer"]         # [MWel] 
    Qrec_H2 = x["q_electrolyzer"] * PH2   # [MWth]

    # Compress H2 and size CAPEX
    T1 = 75 + 273.15        # [K] from [AEL tech, Table2.1 MSc Jacobsson & Palmgren, 2025]
    P1 = 20                 # Initial pressure [bar], assumed based on [Danish Energy Agency]
    Wcomp_list, Qcool_list, P_list, T_list = compression_energy(
        mcaptured=mH2,
        T1=T1,
        P1=P1,
        gas_type='H2',
        n_stages=2,
        pressure_ratio=1.7,
        Tdiff=60,
        thermo_props=c["thermo_props"],
        n_is=x["n_is"],
        printing=False
    )
    Wcomp_H2 = sum(Wcomp_list)  # [MW]
    Qcool_H2 = sum(Qcool_list)  # [MW]
    CAPEX_compress_H2, _ = compression_cost(Wcomp_list, 'H2', printing=False) # [EUR]
    CAPEXcomp_H2 = levelize_kEUR(CAPEX_compress_H2/1000, annual_CO2, x)        # [EUR/tCO2]

    # Produce methanol and extra Qdh - check Danish Agency Agency for method - we treat the whole synthesis plant as a single unit
    Qsteam_synthesis = x['q_synthesis'] * QH2                # [MW] 
    Qmethanol = x['n_synthesis'] * (QH2 + Qsteam_synthesis)  # [MW]
    m_methanol = Qmethanol/x['LHVmethanol'] /1000*3600*24    # [t/day] 
    Qrec_distill = x['q_distill'] * (QH2 + Qsteam_synthesis) # [MW] NOTE: optimistic assumption on heat recovery, from condensers at distillation
    Qloss = 0.02 * (QH2 + Qsteam_synthesis)                  # [MW] 

    # Penalize CHP and recover Qdh
    P = plant["P"] * (1 - Qreb/plant["Qwaste"] - Qsteam_synthesis/plant["Qwaste"]) # live steam for synthesis
    P = P - Pcapture - PH2 - Wcomp_CO2 - Wcomp_H2            # [MWel]
    Qdh = plant["Qdh"] * (1 - Qreb/plant["Qwaste"] - Qsteam_synthesis/plant["Qwaste"])
    Qdh = Qdh + Qcool_CO2 + Qcool_H2 + Qhex + (Qrec_H2 + Qrec_distill)*x["heat_optimism"] # [MW] the cooling is NECESSARY, the H2 recovery is uncertain

    Ppenalty = (plant["P"] - P) * x["FLH"]          # [MWh/yr] probably very positive
    Qpenalty = (plant["Qdh"] - Qdh) * x["FLH"]      # [MWh/yr] probably negative
    Qmethanol = Qmethanol * x["FLH"]                # [MWh/yr] positive
    
    # Estimate CAPEX and OPEX of remaining units                                            # [tCO2/yr]
    CAPEX_capture = x["CAPEXref_capture"] * (annual_CO2/(400*10**3)) ** x["k"] * x["CEPCI"] # [kEUR]
    CAPEXlev_capture = levelize_kEUR(CAPEX_capture, annual_CO2, x)                          # [EUR/tCO2]
    OPEXfix_capture = (CAPEX_capture*1000 * x["OPEXfix"]) / annual_CO2                      # [EUR/tCO2] 

    CAPEX_H2 = x["CAPEXref_H2"] * PH2 * x["CEPCI"]                                          # [kEUR]
    CAPEXlev_H2 = levelize_kEUR(CAPEX_H2, annual_CO2, x)                                    # [EUR/tCO2]
    OPEXfix_H2 = (CAPEX_H2*1000 * x["OPEXfix"]) / annual_CO2                                # [EUR/tCO2]  

    CAPEX_synthesis = x["CAPEXref_synthesis"] * m_methanol ** -0.315 *1000 * x["CEPCI"]     # [kEUR]
    CAPEXlev_synthesis = levelize_kEUR(CAPEX_synthesis, annual_CO2, x)                      # [EUR/tCO2]
    OPEXfix_synthesis = (CAPEX_synthesis*1000 * x["OPEXfix"]) / annual_CO2                  # [EUR/tCO2]

    OPEXmakeup = x["camine"]                                                                # [EUR/tCO2]
    OPEXenergy = (Ppenalty*x["celc"] + Qpenalty*x["celc"]*x["cheat"]) / annual_CO2          # [EUR/tCO2] positive celc negative cheat

    # Calculate the methanol strike price
    CAC = [CAPEXlev_capture,
           OPEXfix_capture,
           CAPEXlev_H2,
           OPEXfix_H2,
           CAPEXlev_synthesis,
           OPEXfix_synthesis,
           CAPEXcomp_CO2,
           CAPEXcomp_H2,
           OPEXmakeup,
           OPEXenergy]
    CAC = sum(CAC)                                                  # [EUR/tCO2]

    methanol_cost = CAC/1000 * 44                                   # [EUR/kmolCO2 = EUR/kmolCH3OH]
    methanol_cost = methanol_cost / 32                              # [EUR/kgCH3OH]
    strike_price = methanol_cost*1000                               # [EUR/tCH3OH]
    strike_price = strike_price * (1 + c["profit"])

    fossil = plant["Fossil"] / plant["Total"]                       # [tfossil/t] 
    biogenic = 1 - fossil                                           # [tbiogenic/t] 
    fossil -= x["carbon_change"]
    biogenic += x["carbon_change"]
    FCCU = annual_CO2 * fossil /1000                                # [ktCO2/yr]
    BCCU = annual_CO2 * biogenic /1000                              # [ktCO2/yr]

    cost_details = {                                                            
        'CAC': CAC # Add later if needed
    }

    return strike_price, FCCU, BCCU, Ppenalty, Qpenalty, Qmethanol, cost_details # [MWh/yr]

def WACCUS_EPR( 
    # constants
    question="granulates",
    CCUS="CCS", 
    agency = False,
    plants_df=None, 
    shipping_df=None,
    truck_df=None,          
    compression_df=None,
    thermo_props=None,    
    SEK_to_EUR=0.091,
    profit=0.10,
    print_auction=False,

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

    CAPEXref_capture = 3550*0.09*1000,  # [MNOK]->[kEUR] @400 ktCO2/yr [Gassnova, Demonstrasjon av Fullskala CO2-Håndtering - Rapport for Avsluttet Forprosjekt]
    CAPEXref_HP = 0.86,                 # [MEUR/MWth] [Bergander & Hellander, 2025]
    CAPEXref_H2 = 550,                  # [kEUR/MWe] [Danish Agency Excel Renewable Fuels AEC100MW]
    CAPEXref_synthesis = 1.8749,        # [MEUR] [Danish Renewable Fuels PDF has a power function of CAPEX_synthesis. Fig4, p.186.] 
    CAPEXref_loading = 63000000,        # [SEK*] @150 ktCO2/yr excluding railway track [Koldioxid på tåg, 2024]
    CAPEXref_train = 8610000,           # [EUR*] an oversized train @15 wagons, cost = 4.98 *10**6 + 242*15 *10**3 [MSc Gunnarsson, 2025]
    k = 0.67,                           # [-] [Stenström, 2025] assumed economy-of-scale factor
    CEPCI = 900,                        # [-] [University of Manchester, 2025] applies to reference CAPEX values
    
    OPEXfix = 0.02,         # [-] % of base CAPEX, calculated from [Ramboll-Malmö, 2023]
    dr = 0.075,             # [-]
    t = 25,                 # [yr]
    camine = 44,            # [SEK/tCO2] [Ramboll-Malmö, 2023]
    celc = 60,              # [EUR/MWh]
    cheat = 0.75,           # [% of elc]
    CRC = 100,              # [EUR/tCO2]
    ETS = 80,               # [EUR/tCO2] Use this report: The EU-ETS Price Through 2030 and Beyond: A closer look at drivers, models and assumptions (https://www.ecologic.eu/19034)
    pmethanol = 625,        # [EUR/t] [MSc Omar & Widgren, 2025]

    ship_uncertain = 0.10,  # [-] [-0.15,0.15]
    truck_uncertain = 0.10, # [-] [-0.15,0.15]
    train_uncertain = 0.10, # [-] [-0.15,0.15]
    stockholm = 2,          # [Mt/yr] [1,2,3] 
    malmo = 1,              # [Mt/yr] [0.5,1,2] 
    gothenburg = 1,         # [Mt/yr] [0.5,1,2] 
    storage = "oygarden",   # ["oygarden", "kalundborg"]

    # levers
    tax = 80,          # [EUR/tCO2] [50, 400] NOTE: explore discrete ranges => easier to visualize later
    recyclable = 0.15,  # [-] fraction of products possible to recycle mechanically (exempt from tax), determined by policy criteria
):
    # Store constants, uncertainties (converted to EUR or CEPCI), and levers
    c = {
        "shipping_df": shipping_df,
        "truck_df": truck_df,
        "compression_df": compression_df,
        "thermo_props": thermo_props,
        "profit": profit,
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
        "CAPEXref_HP": CAPEXref_HP,
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

    if agency:
        fund = fund/2 # Share half the budget to NV agency rather than EM

    # Create CCUS bids
    bids = []
    for _, plant in plants_df.iterrows():
        if CCUS == "CCS":
            strike_price, FCCS, BECCS, Ppenalty, Qpenalty, cost_details = plan_CCS(plant, c, x, l)
            bids.append({
                'type': 'CCS',
                'name': plant['Name'],
                'strike_price': strike_price,
                'FCCS': FCCS,
                'BECCS': BECCS,
                'Ppenalty': Ppenalty,
                'Qpenalty': Qpenalty,
                'cost_details': cost_details
            })

        if CCUS == "CCU":
            strike_price, FCCU, BCCU, Ppenalty, Qpenalty, Qmethanol, cost_details = plan_CCU(plant, c, x, l)
            bids.append({
                'type': 'CCU',
                'name': plant['Name'],
                'strike_price': strike_price,
                'FCCU': FCCU,
                'BCCU': BCCU,
                'Ppenalty': Ppenalty,
                'Qpenalty': Qpenalty,
                'Qmethanol': Qmethanol,
                'cost_details': cost_details
            })
    bids.sort(key=lambda x: x['strike_price'])
    if print_auction:
        print("\nBids (sorted):")
        print(f"{'Name':<25} {'Strike price (EUR/tCO2 or CH3OH)':>15} {'CAC (EUR/tCO2)':>18}")
        print("-" * 60)
        for bid in bids:
            print(f"{bid['name']:<25} {bid['strike_price']:>15.2f} {bid['cost_details']['CAC']:>18.2f}")

    # Run reverse auction
    remaining_fund = fund
    awarded_plants = []
    
    for bid in bids:
        if bid['type'] == 'CCS':
            # requested_amount = bid['bid'] * (bid['FCCS'] + bid['BECCS'])*1000 /(10**6) # [EUR/tCO2 * ktCO2/yr => MEUR/yr]
            requested_amount = (bid['strike_price'] - x['ETS']) * (bid['FCCS'] + bid['BECCS'])*1000 /(10**6) # [EUR/tCO2 * ktCO2/yr => MEUR/yr]
            awarded = requested_amount <= remaining_fund # [MEUR/yr]
            if awarded:
                remaining_fund -= requested_amount
            awarded_plants.append({
                'name': bid['name'],
                'type': bid['type'],
                'awarded': awarded,
                'strike': bid['strike_price'],
                'FCCS': bid['FCCS'],
                'BECCS': bid['BECCS'],
                'CCStot': bid['FCCS'] + bid['BECCS'],
                'Ppenalty': bid['Ppenalty'],
                'Qpenalty': bid['Qpenalty'],
                'FCCU': 0,
                'BCCU': 0,
                'CCUtot': 0,
                'Qmethanol': 0,
                'amount': requested_amount if awarded else 0,
                'cost_details': bid.get('cost_details', {})
            })
        else:  # CCU
            requested_amount = (bid['strike_price'] - x['pmethanol']) * (bid['Qmethanol']/ (x['LHVmethanol']/3600) /1000) /(10**6) # [MEUR/yr]
            awarded = requested_amount <= remaining_fund
            if awarded:
                remaining_fund -= requested_amount
            awarded_plants.append({
                'name': bid['name'],
                'type': bid['type'],
                'awarded': awarded,
                'strike': bid['strike_price'],
                'FCCS': 0,
                'BECCS': 0,
                'CCStot': 0,
                'Ppenalty': bid['Ppenalty'],
                'Qpenalty': bid['Qpenalty'],
                'FCCU': bid['FCCU'],
                'BCCU': bid['BCCU'],
                'CCUtot': bid['FCCU'] + bid['BCCU'],
                'Qmethanol': bid['Qmethanol'],
                'amount': requested_amount if awarded else 0
            })

    if print_auction:
        print("\nSummary of auction:")
        print(f"Total fund: {fund:.2f} MEUR/yr")
        print(f"Remaining fund: {remaining_fund:.2f} MEUR/yr")
        print(f"Number of plants awarded: {len([p for p in awarded_plants if p['awarded']])}")
        print("\nAwarded plants:")
        print(f"{'Plant Name':<20} {'Type':<6} {'Awarded':<6} {'Strikepr':>10} {'[MEUR/yr]':>10} {'FCCS':>10} {'BECCS':>10} {'CCStot':>12} {'FCCU':>10} {'BCCU':>10} {'CCUtot':>12} {'Ppenalty':>12} {'Qpenalty':>12} {'Qmethanol':>12}")
        print("-" * 160)
        for plant in awarded_plants:
            print(f"{plant['name']:<20} {plant['type']:<6} {str(plant['awarded']):<6} {plant['strike']:>10.2f} {plant['amount']:>10.2f} {plant['FCCS']:>10.2f} {plant['BECCS']:>10.2f} {plant['CCStot']:>12.2f} {plant['FCCU']:>10.2f} {plant['BCCU']:>10.2f} {plant['CCUtot']:>12.2f} {plant['Ppenalty']:>12.2f} {plant['Qpenalty']:>12.2f} {plant['Qmethanol']:>12.2f}")

    # Plot merit order curve
    if print_auction:
        plt.figure(figsize=(12, 8))
        plant_names = [plant['name'] for plant in awarded_plants]
        strike_prices = [plant['strike'] for plant in awarded_plants]
        awarded_status = [plant['awarded'] for plant in awarded_plants]
        ccstot_values = [plant['CCStot'] for plant in awarded_plants]
        
        threshold = 100
        
        # Create cumulative capacity for x-axis
        cumulative_capacity = np.cumsum(ccstot_values)
        
        # Create two separate line series for the stacked effect
        below_threshold = [min(price, threshold) for price in strike_prices]
        above_threshold = [max(0, price - threshold) for price in strike_prices]
        
        # Plot each plant's area individually
        for i, (below, above, ccstot) in enumerate(zip(below_threshold, above_threshold, ccstot_values)):
            x_start = cumulative_capacity[i] - ccstot
            x_end = cumulative_capacity[i]
            
            # Fill below threshold portion
            if below > 0:
                plt.fill_between([x_start, x_end], 0, below, color='gray', alpha=0.7, step='post')
            
            # Fill above threshold portion
            if above > 0:
                plt.fill_between([x_start, x_end], below, below + above, color='crimson', alpha=0.7, step='post')
            
            # Draw black outline around the entire plant area
            plt.plot([x_start, x_end, x_end, x_start, x_start], 
                    [0, 0, below + above, below + above, 0], 'k-', linewidth=1.5)
        
        # Add dashed pattern overlay for awarded plants
        for i, (awarded, below, above, ccstot) in enumerate(zip(awarded_status, below_threshold, above_threshold, ccstot_values)):
            if awarded:
                x_start = cumulative_capacity[i] - ccstot
                x_end = cumulative_capacity[i]
                # Add dashed pattern for below threshold portion
                if below > 0:
                    plt.fill_between([x_start, x_end], 0, below, color='gray', alpha=0.7, hatch='///', step='post', edgecolor='black', linewidth=0.5)
                # Add dashed pattern for above threshold portion
                if above > 0:
                    plt.fill_between([x_start, x_end], below, below + above, color='crimson', alpha=0.7, hatch='///', step='post', edgecolor='black', linewidth=0.5)
        
        plt.xlabel('Cumulative CCS Capacity (ktCO2/yr)', fontsize=14)
        plt.ylabel('Strike Price (EUR/tCO2 or tCH3OH)', fontsize=14)
        plt.title('Merit Order Curve - Strike Prices vs CCS Capacity', fontsize=16)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.ylim(0, max(strike_prices)*1.33)
        plt.grid(True, alpha=0.3)
        
        # Add horizontal line at threshold
        plt.axhline(y=threshold, color='black', linestyle='--', alpha=0.7, linewidth=2)
        
        # # Add legend
        # from matplotlib.patches import Patch
        # legend_elements = [Patch(facecolor='gray', alpha=0.7, label=f'Below {threshold}'),
        #                   Patch(facecolor='crimson', alpha=0.7, label=f'Above {threshold}'),
        #                   Patch(facecolor='gray', alpha=0.3, hatch='///', label='Awarded')]
        # plt.legend(handles=legend_elements, fontsize=14)
        
        # Add plant labels at the center of each capacity segment
        for i, (name, price, ccstot) in enumerate(zip(plant_names, strike_prices, ccstot_values)):
            x_pos = cumulative_capacity[i] - ccstot/2
            plt.text(x_pos, price + price*0.005, f'{price:.0f}', 
                    ha='center', va='bottom', fontsize=12, rotation=0)
        plt.tight_layout()
        plt.savefig('results/fig3_auction.png', dpi=300, bbox_inches='tight')

    # Calculate sums of awarded metrics
    total_FCCS = sum(plant['FCCS'] for plant in awarded_plants if plant['awarded'])
    total_BECCS = sum(plant['BECCS'] for plant in awarded_plants if plant['awarded'])
    total_FCCU = sum(plant['FCCU'] for plant in awarded_plants if plant['awarded'])
    total_BCCU = sum(plant['BCCU'] for plant in awarded_plants if plant['awarded'])
    total_Ppenalty = sum(plant['Ppenalty'] for plant in awarded_plants if plant['awarded'])
    total_Qpenalty = sum(plant['Qpenalty'] for plant in awarded_plants if plant['awarded'])
    total_Qmethanol = sum(plant['Qmethanol'] for plant in awarded_plants if plant['awarded'])
    potential_CCS = sum(plant['FCCS'] + plant['BECCS'] for plant in awarded_plants)
    potential_CCU = sum(plant['FCCU'] + plant['BCCU'] for plant in awarded_plants)

    output = {
        'mass_taxed': mass_taxed,           # [tpl/yr]
        'fund': fund,                       # [MEUR/yr]
        'remaining_fund': remaining_fund,   # [MEUR/yr]

        'granulates_inc': granulates_inc,   # [-]
        'products_inc': products_inc,       # [-]
        'bag_inc': bag_inc,                 # [-]

        'total_FCCS': total_FCCS,           # [ktCO2/yr]
        'total_BECCS': total_BECCS,         # [ktCO2/yr]
        'total_CCS': total_FCCS + total_BECCS, # [ktCO2/yr]
        'total_FCCU': total_FCCU,           # [ktCO2/yr]
        'total_BCCU': total_BCCU,           # [ktCO2/yr]
        'total_CCU': total_FCCU + total_BCCU, # [ktCO2/yr]
        'total_Ppenalty': total_Ppenalty,   # [MWh/yr]
        'total_Qpenalty': total_Qpenalty,   # [MWh/yr]
        'total_Qmethanol': total_Qmethanol, # [MWh/yr]

        'potential_CCS': potential_CCS,   # [ktCO2/yr]
        'potential_CCU': potential_CCU,   # [ktCO2/yr]
        'n_plants': len([plant for plant in awarded_plants if plant['awarded']]), # [n]
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
        agency=False,
        plants_df=plants_df, 
        shipping_df=shipping_df,
        truck_df=truck_df,
        compression_df=compression_df,
        thermo_props=thermo_props,    
        SEK_to_EUR=SEK_to_EUR,
        profit=0.10,
        print_auction=True,
    )

    print("\n----------------------------------------These are the simulation results:----------------------------------------")
    print("For reference, Exergi received 20 BSEK over 15 years => 1.33 BSEK/yr => 120 MEUR/yr, or 150 EUR/tCO2")
    print("Mass taxed:", output["mass_taxed"], " [tpl/yr]")
    print("Fund:", output["fund"], " [MEUR/yr]")
    print("Remaining fund:", output["remaining_fund"], " [MEUR/yr]")
    print("\nGranulates inc:", output["granulates_inc"], " [-]")
    print("Products inc:", output["products_inc"], " [-]")
    print("Bag inc:", output["bag_inc"], " [-]")
    print("\nTotal FCCS:", output["total_FCCS"], " [ktCO2/yr]")
    print("Total BECCS:", output["total_BECCS"], " [ktCO2/yr]")
    print("Total FCCU:", output["total_FCCU"], " [ktCO2/yr]")
    print("Total BCCU:", output["total_BCCU"], " [ktCO2/yr]")
    print("\nTotal Ppenalty:", output["total_Ppenalty"]/1000, " [GWh/yr]")
    print("Total Qpenalty:", output["total_Qpenalty"]/1000, " [GWh/yr]")
    print("Total Qmethanol:", output["total_Qmethanol"]/1000, " [GWh/yr]")
    
    plt.show()

    
