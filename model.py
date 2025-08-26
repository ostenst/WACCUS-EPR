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
    
    # Calculate levelized CAPEX using the levelize function
    levelized_CAPEX = levelize(CAPEX, mcaptured, x)  # [EUR/t]
    
    # # Calculate annualized CAPEX for backward compatibility
    # CRF = x["dr"] * (1 + x["dr"])**x["t"] / ((1 + x["dr"])**x["t"] - 1)
    # annualized_CAPEX = CAPEX * CRF  # [kEUR/yr]
    
    return CAPEX, levelized_CAPEX

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
        Qcool = 0 if stage == n_stages - 1 else mcaptured * cp_out * (T_actual - (T + Tdiff))    # [kJ/s] NOTE: Compressor temps should not increase... but it does in Beiron?
        
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

def levelize(CAPEX, mcaptured, x):
    """
    Levelize CAPEX to EUR/tCO2.
    
    Args:
        CAPEX: Capital expenditure [kEUR]
        mcaptured: CO2 capture rate [kg/s]
        x: dictionary containing parameters
        
    Returns:
        float: Levelized CAPEX in EUR/tCO2
    """
    # Calculate annualized CAPEX using CRF
    CRF = x["dr"] * (1 + x["dr"])**x["t"] / ((1 + x["dr"])**x["t"] - 1)
    annualized_CAPEX = CAPEX * CRF  # [kEUR/yr]
    
    # Convert to per-ton costs
    annual_CO2 = mcaptured/1000*3600 * x["FLH"]  # [tCO2/yr]
    levelized_CAPEX = annualized_CAPEX / annual_CO2 * 1000  # [EUR/tCO2]
    
    return levelized_CAPEX

def plot_CCU_CHP(plant, P, Qdh, Preb, Pcapture, PH2, Wcomp_CO2, Wcomp_H2, Qdhreb, Qhex, Qcool_CO2, Qrec_H2, Qcool_H2, Qdistill):
    """
    Create a bar plot showing power and heat changes in the CCU process.
    
    Args:
        plant: Dictionary containing plant data
        P: Final power output [MW]
        Qdh: Final district heating output [MW]
        Preb: Power for reboiler [MW]
        Pcapture: Power for capture [MW]
        PH2: Power for H2 production [kW]
        Wcomp_CO2: CO2 compression power [MW]
        Wcomp_H2: H2 compression power [MW]
        Qdhreb: Heat for reboiler [MW]
        Qhex: Heat from heat exchanger [MW]
        Qcool_CO2: CO2 cooling heat [MW]
        Qrec_H2: H2 recovery heat [kW]
        Qcool_H2: H2 cooling heat [MW]
        Qdistill: Distillation heat [MW]
    """
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Define colors
    power_colors = {
        'Initial': '#1f77b4',  # Blue
        'Final': '#2ca9b8',    # Teal
        'Reboiler': '#9b4dca', # Purple
        'Power Capture Plant': '#4d79a6',  # Darker blue
        'H2 Production': '#5d9cec', # Light blue
        'CO2 Compression': '#7fb3d5', # Sky blue
        'H2 Compression': '#a8d8ea'   # Light teal
    }
    
    heat_colors = {
        'Initial': '#d62728',  # Red
        'Final': '#ff7f0e',    # Orange
        'Reboiler': '#9b4dca', # Purple (same as power reboiler)
        'Direct HEX DH': '#e57373', # Light red
        'CO2 Cooling': '#ef9a9a',    # Pink
        'Electrolyzer Heat': '#ffab91',    # Light orange
        'H2 Cooling': '#ffcc80',     # Peach
        'Distillation Heat': '#ffe082'    # Light yellow
    }
    
    # Add grid
    ax1.grid(True, linestyle='--', alpha=0.3)
    ax2.grid(True, linestyle='--', alpha=0.3)
    
    # Power balance plot
    power_initial = plant["P"]
    power_final = P
    
    # Create power bars
    bars1 = ax1.bar(['Initial', 'Change', 'Final'], 
            [power_initial, 0, power_final], 
            color=[power_colors['Initial'], 'gray', power_colors['Final']])
    
    # Annotate initial and final power values
    ax1.text(0, power_initial, f'{power_initial:.1f}', ha='center', va='bottom')
    ax1.text(2, power_final, f'{power_final:.1f}', ha='center', va='bottom')
    
    # Create detailed power change bar
    power_components = {
        'Reboiler': -Preb,
        'Power Capture Plant': -Pcapture,
        'H2 Production': -PH2/1000,
        'CO2 Compression': -Wcomp_CO2,
        'H2 Compression': -Wcomp_H2
    }
    
    bottom = 0
    for component, value in power_components.items():
        bar = ax1.bar('Change', value, bottom=bottom, label=component, color=power_colors[component])
        # Annotate each component
        if value != 0:
            ax1.text(1, bottom + value/2, f'{value:.1f}', ha='center', va='center', color='white')
        bottom += value
    
    # Annotate total change
    total_change = sum(power_components.values())
    ax1.text(1, total_change, f'{total_change:.1f}', ha='center', va='bottom')
    
    ax1.set_title('Power Balance')
    ax1.set_ylabel('Power [MW]')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Heat balance plot
    heat_initial = plant["Qdh"]
    heat_final = Qdh
    
    # Create heat bars for initial and final states
    bars2 = ax2.bar(['Initial', 'Change', 'Final'], 
            [heat_initial, 0, heat_final], 
            color=[heat_colors['Initial'], 'gray', heat_colors['Final']])
    
    # Annotate initial and final heat values
    ax2.text(0, heat_initial, f'{heat_initial:.1f}', ha='center', va='bottom')
    ax2.text(2, heat_final, f'{heat_final:.1f}', ha='center', va='bottom')
    
    # Create detailed heat change bar
    heat_components_negative = {
        'Reboiler': -Qdhreb,
    }
    
    heat_components_positive = {
        'Direct HEX DH': Qhex,
        'CO2 Cooling': Qcool_CO2,
        'Electrolyzer Heat': Qrec_H2/1000,
        'H2 Cooling': Qcool_H2,
        'Distillation Heat': Qdistill
    }
    
    # Plot negative components first
    bottom = 0
    for component, value in heat_components_negative.items():
        bar = ax2.bar('Change', value, bottom=bottom, label=component, color=heat_colors[component])
        # Annotate each component
        if value != 0:
            ax2.text(1, bottom + value/2, f'{value:.1f}', ha='center', va='center', color='white')
        bottom += value
    
    # Reset bottom to 0 for positive components
    bottom = 0
    for component, value in heat_components_positive.items():
        bar = ax2.bar('Change', value, bottom=bottom, label=component, color=heat_colors[component])
        # Annotate each component
        if value != 0:
            ax2.text(1, bottom + value/2, f'{value:.1f}', ha='center', va='center', color='white')
        bottom += value
    
    # Annotate total change
    total_change = sum(heat_components_negative.values()) + sum(heat_components_positive.values())
    ax2.text(1, total_change, f'{total_change:.1f}', ha='center', va='bottom')
    
    ax2.set_title('Heat Balance')
    ax2.set_ylabel('Heat [MW]')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    
    # Save figure at 600 dpi
    fig.savefig('CCU_CHP_balance.png', dpi=600, bbox_inches='tight')
    
    return fig

def plot_CCU_combined(plant, P, Qdh, Preb, Pcapture, PH2, Wcomp_CO2, Wcomp_H2, Qdhreb, Qhex, Qcool_CO2, Qrec_H2, Qcool_H2, Qdistill, Qmethanol):
    """
    Create a combined bar plot showing power and heat changes in the CCU process.
    
    Args:
        plant: Dictionary containing plant data
        P: Final power output [MW]
        Qdh: Final district heating output [MW]
        Preb: Power for reboiler [MW]
        Pcapture: Power for capture [MW]
        PH2: Power for H2 production [kW]
        Wcomp_CO2: CO2 compression power [MW]
        Wcomp_H2: H2 compression power [MW]
        Qdhreb: Heat for reboiler [MW]
        Qhex: Heat from heat exchanger [MW]
        Qcool_CO2: CO2 cooling heat [MW]
        Qrec_H2: H2 recovery heat [kW]
        Qcool_H2: H2 cooling heat [MW]
        Qdistill: Distillation heat [MW]
        Qmethanol: Methanol production heat [MW]
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(15, 8))
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.3)
    
    # Define colors (same as before)
    power_colors = {
        'Initial': '#1f77b4',  # Blue
        'Final': '#2ca9b8',    # Teal
        'Reboiler': '#9b4dca', # Purple
        'Power Capture Plant': '#4d79a6',  # Darker blue
        'H2 Production': '#5d9cec', # Light blue
        'CO2 Compression': '#7fb3d5', # Sky blue
        'H2 Compression': '#a8d8ea'   # Light teal
    }
    
    heat_colors = {
        'Initial': '#d62728',  # Red
        'Final': '#ff7f0e',    # Orange
        'Reboiler': '#9b4dca', # Purple (same as power reboiler)
        'Direct HEX DH': '#e57373', # Light red
        'CO2 Cooling': '#ef9a9a',    # Pink
        'Electrolyzer Heat': '#ffab91',    # Light orange
        'H2 Cooling': '#ffcc80',     # Peach
        'Distillation Heat': '#ffe082'    # Light yellow
    }
    
    # Set up the x-axis positions
    x_positions = ['Initial', 'Change', 'Final', 'Methanol']
    x = np.arange(len(x_positions))
    width = 0.35  # Width of the bars
    
    # Plot initial state
    ax.bar(x[0] - width/2, plant["P"], width, label='Power', color=power_colors['Initial'])
    ax.bar(x[0] + width/2, plant["Qdh"], width, label='Heat', color=heat_colors['Initial'])
    
    # Annotate initial values
    ax.text(x[0] - width/2, plant["P"], f'{plant["P"]:.1f}', ha='center', va='bottom')
    ax.text(x[0] + width/2, plant["Qdh"], f'{plant["Qdh"]:.1f}', ha='center', va='bottom')
    
    # Plot change components
    # Power components
    power_components = {
        'Reboiler': -Preb,
        'Power Capture Plant': -Pcapture,
        'H2 Production': -PH2/1000,
        'CO2 Compression': -Wcomp_CO2,
        'H2 Compression': -Wcomp_H2
    }
    
    bottom = 0
    for component, value in power_components.items():
        bar = ax.bar(x[1] - width/2, value, width, bottom=bottom, label=f'Power: {component}', color=power_colors[component])
        # Annotate each component
        if value != 0:
            ax.text(x[1] - width/2, bottom + value/2, f'{value:.1f}', ha='center', va='center', color='white')
        bottom += value
    
    # Heat components
    heat_components_negative = {
        'Reboiler': -Qdhreb,
    }
    
    heat_components_positive = {
        'Direct HEX DH': Qhex,
        'CO2 Cooling': Qcool_CO2,
        'Electrolyzer Heat': Qrec_H2/1000,
        'H2 Cooling': Qcool_H2,
        'Distillation Heat': Qdistill
    }
    
    # Plot negative heat components
    bottom = 0
    for component, value in heat_components_negative.items():
        bar = ax.bar(x[1] + width/2, value, width, bottom=bottom, label=f'Heat: {component}', color=heat_colors[component])
        # Annotate each component
        if value != 0:
            ax.text(x[1] + width/2, bottom + value/2, f'{value:.1f}', ha='center', va='center', color='white')
        bottom += value
    
    # Plot positive heat components
    bottom = 0
    for component, value in heat_components_positive.items():
        bar = ax.bar(x[1] + width/2, value, width, bottom=bottom, label=f'Heat: {component}', color=heat_colors[component])
        # Annotate each component
        if value != 0:
            ax.text(x[1] + width/2, bottom + value/2, f'{value:.1f}', ha='center', va='center', color='white')
        bottom += value
    
    # Plot final state
    ax.bar(x[2] - width/2, P, width, color=power_colors['Final'])
    ax.bar(x[2] + width/2, Qdh, width, color=heat_colors['Final'])
    
    # Annotate final values
    ax.text(x[2] - width/2, P, f'{P:.1f}', ha='center', va='bottom')
    ax.text(x[2] + width/2, Qdh, f'{Qdh:.1f}', ha='center', va='bottom')
    
    # Plot methanol production
    ax.bar(x[3] - width/2, Qmethanol, width, color='red', label='Methanol Production', alpha=0.5)
    ax.text(x[3] - width/2, Qmethanol, f'{Qmethanol:.1f}', ha='center', va='bottom')
    ax.bar(x[3] + width/2, -((plant["Qwaste"]+abs(P))-(Qdh+Qmethanol)), width, color='red', label='Methanol losses', hatch='////', alpha=0.5)
    ax.text(x[3] + width/2, -((plant["Qwaste"]+abs(P))-(Qdh+Qmethanol)), f'{((plant["Qwaste"]+abs(P))-(Qdh+Qmethanol)):.1f}', ha='center', va='bottom')
    
    # Customize the plot
    ax.set_ylabel('Energy [MW]')
    ax.set_title(f'{plant["Name"]} - Waste CHP and methanol balance [MW]')
    ax.set_xticks(x)
    ax.set_xticklabels(x_positions)
    
    # Add a horizontal line at y=0
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Create custom legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=power_colors['Initial'], label='Power: Initial'),
        Patch(facecolor=heat_colors['Initial'], label='Heat: Initial'),
        Patch(facecolor=power_colors['Final'], label='Power: Final'),
        Patch(facecolor=heat_colors['Final'], label='Heat: Final'),
        Patch(facecolor=power_colors['Reboiler'], label='Reboiler (Power & Heat)'),
        Patch(facecolor='red', label='Methanol Production', alpha=0.5),
        Patch(facecolor='red', label='Methanol losses', hatch='////', alpha=0.5),
    ]
    
    # Add power components to legend
    for component in ['Power Capture Plant', 'H2 Production', 'CO2 Compression', 'H2 Compression']:
        legend_elements.append(Patch(facecolor=power_colors[component], label=f'Power: {component}'))
    
    # Add heat components to legend
    for component in ['Direct HEX DH', 'CO2 Cooling', 'Electrolyzer Heat', 'H2 Cooling', 'Distillation Heat']:
        legend_elements.append(Patch(facecolor=heat_colors[component], label=f'Heat: {component}'))
    
    ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    
    # Save figure at 600 dpi
    fig.savefig('CCU_combined_balance.png', dpi=600, bbox_inches='tight')
    
    return fig

def plan_CCU(plant, x, plot_single=True):
    # burn fuel
    mfuel = plant["Qwaste"] / (x["LHV"]/3600) /3600  # [kgf/s]
    mCO2 = mfuel* x["Ccontent"] * 44/12              # [kgCO2/s]

    # capture and compress CO2
    mcaptured = mCO2 * 0.90                          # [kgCO2/s]
    Qreb = mcaptured * x["qreb"]                     # [MW]
    Pcapture = 0.1 * mcaptured/1000*3600             # [MW] [Beiron, 2022]
    Qhex = 0.64 * Qreb                               # [MW] [Beiron, 2022]
    
    T1 = 40 + 273.15  # Initial temperature [K] [Deng, 2019]
    P1 = 1.0         # Initial pressure [bar]
    Wcomp_list, Qcool_list, P_list, T_list = compression_energy( # Setting target p and T according to [Beiron, 2025 (unpublished)]
        mcaptured=mcaptured,
        T1=T1,
        P1=P1,
        gas_type='CO2',
        n_stages=3,
        pressure_ratio=3.7,
        Tdiff=30,
        thermo_props=x["thermo_props"],
        etais=x["etais"],
        printing=False
    )
    Wcomp_CO2 = sum(Wcomp_list)  # [MW]
    cost_CO2, _ = compression_cost(Wcomp_list, 'CO2', printing=False)
    Qcool_CO2 = sum(Qcool_list)  # [MW]

    # produce H2 from AEL electrolyzer
    Hi = 241.82             # [kJ/molH2] [Formelsamling]
    nCO2 = mcaptured/44     # [kmol/s]
    nH2 = nCO2 * 3          # [kmol/s] needed
    QH2 = nH2*1000 * Hi     # [kW] 
    PH2 = QH2/0.699         # [kW] Table2.1 MSc Jacobsson & Palmgren, 2025
    Qrec_H2 = 0.154 * PH2   # [kW] [AEL tech, Fig2.1 MSc Jacobsson & Palmgren, 2025] OR [Danish Renwable Fuels 100MW AEC] NOTE: Optimistic

    # compress the H2
    mH2 = nH2 * 2           # [kg/s] 
    T1 = 75 + 273.15        # [AEL tech, Table2.1 MSc Jacobsson & Palmgren, 2025]
    P1 = 20                 # Initial pressure [bar], assumed based on [Danish]
    Wcomp_list, Qcool_list, P_list, T_list = compression_energy(
        mcaptured=mH2,
        T1=T1,
        P1=P1,
        gas_type='H2',
        n_stages=2,
        pressure_ratio=1.6,
        Tdiff=60,
        thermo_props=x["thermo_props"],
        etais=x["etais"],
        printing=False
    )
    Wcomp_H2 = sum(Wcomp_list)  # [MW]
    cost_H2, _ = compression_cost(Wcomp_list, 'H2', printing=False)
    Qcool_H2 = sum(Qcool_list)  # [MW]

    # produce methanol and extra DH - check Danish Agency for method - we treat the whole synthesis plant as a single unit
    Qsteam_synthesis = 0.08/(1-0.08) * QH2/1000 # [MW] [Danish Renewable Fuels Fig3, section 5.2 Methanol from Hydrogen and Carbon Dioxide]
    Qmethanol = 0.78 * (QH2/1000 + Qsteam_synthesis) # [MW]
    Qdistill = 0.20 * (QH2/1000 + Qsteam_synthesis) # [MW] NOTE: Optimistic assumption on heat recovery, from condensers at distillation
    Qloss = 0.02 * (QH2/1000 + Qsteam_synthesis) # [MW] 

    # penalize CHP and recover Qdh
    P = plant["P"] * (1 - Qreb/plant["Qwaste"] - Qsteam_synthesis/plant["Qwaste"])      # assuming live steam is used for reboiler AND synthesis plant
    Preb = plant["P"]*(Qreb+Qsteam_synthesis)/plant["Qwaste"] # power lost to reboiler?
    P = P - Pcapture - PH2/1000 - Wcomp_CO2 - Wcomp_H2

    Qdh = plant["Qdh"] * (1 - Qreb/plant["Qwaste"] - Qsteam_synthesis/plant["Qwaste"])
    Qdhreb = plant["Qdh"]*(Qreb+Qsteam_synthesis)/plant["Qwaste"]
    Qdh = Qdh + Qcool_CO2 + Qcool_H2 + (Qhex + Qrec_H2/1000 + Qdistill)*x["heat_optimism"] # [MW]

    Ppenalty = (plant["P"] - P) * x["FLH"] /1000          # [GWh/yr] probably very positive
    Qpenalty = (plant["Qdh"] - Qdh) * x["FLH"] /1000     # [GWh/yr] probably negative
    Qmethanol = Qmethanol * x["FLH"] /1000               # [GWh/yr] positive

    # print("These KPIs are similar to Beiron, if no recovery from Qhex, Qrec_H2, Qdistill:")
    # KPI1 = Qmethanol/x["FLH"]*1000 / (plant["Qwaste"] + (-P)) # [MW/MW]
    # KPI2 = (Qmethanol/x["FLH"]*1000 + Qdh)/ (plant["Qwaste"] + (-P)) # [MW/MW]
    # KPI21 = 1 - ((plant["Qwaste"]+abs(P))-(Qdh+Qmethanol/x["FLH"]*1000)) / (plant["Qwaste"] + (-P)) # [MW/MW]
    # KPI3 = (-P) / ((plant["P"] * (1 - Qreb/plant["Qwaste"])) + (plant["Qdh"] * (1 - Qreb/plant["Qwaste"])) + (Qmethanol/x["FLH"]*1000)) # [MW/MW]
    # KPI4 = (-P) / (Qmethanol/x["FLH"]*1000)
    # KPI5 = ((plant["Qwaste"]+abs(P))-(Qdh+Qmethanol/x["FLH"]*1000)) / (Qmethanol/x["FLH"]*1000)

    # print(f"KPI1: {KPI1:.2f}")
    # print(f"KPI2: {KPI2:.2f}")
    # print(f"KPI2: {KPI21:.2f}")
    # print(f"KPI3: {KPI3:.2f}")
    # print(f"KPI4: {KPI4:.2f}")
    # print(f"KPI5: {KPI5:.2f}")

    # Estimate CAPEX and OPEX of all units
    CAPEX, levelized_CAPEX = estimate_CAPEX(mcaptured, x)  # [kEUR, EUR/tCO2] NOTE: Includes compression/liq CAPEX...
    # CAPEX_H2 = 550/1000 * PH2 # [kEUR] [Danish] NOTE: Looks wrong, adjust to Jacobsson MSc Table2.1
    CAPEX_H2 = 375 * PH2/1000 # [kEUR/MWH2] NOTE: verify if it is per MWH2 or MWel?
    OPEX_H2 = 0.04 * CAPEX_H2 # [kEUR/yr] 
    CAPEX_synthesis = 1.09*1000 * Qmethanol/x["FLH"] # [kEUR] [Beiron, Grahn, no Danish! Includes destillation probably]
    OPEX_synthesis = 0.05 * CAPEX_synthesis # [kEUR/yr]

    levelized_CAPEX_H2 = levelize(CAPEX_H2, mcaptured, x)
    levelized_CAPEX_synthesis = levelize(CAPEX_synthesis, mcaptured, x)
    levelized_CAPEX_CO2_comp = levelize(cost_CO2/1000, mcaptured, x)  # Convert cost_CO2 from EUR to kEUR
    levelized_CAPEX_H2_comp = levelize(cost_H2/1000, mcaptured, x)    # Convert cost_H2 from EUR to kEUR
    print(levelized_CAPEX, levelized_CAPEX_H2, levelized_CAPEX_synthesis, levelized_CAPEX_CO2_comp, levelized_CAPEX_H2_comp)

    annual_CO2 = mcaptured/1000*3600 * x["FLH"]  # [tCO2/yr]
    levelized_OPEX_H2 = OPEX_H2 / annual_CO2 * 1000  # [EUR/tCO2]
    levelized_OPEX_synthesis = OPEX_synthesis / annual_CO2 * 1000  # [EUR/tCO2]

    # Summarize and bid
    CAC = levelized_CAPEX + levelized_CAPEX_H2 + levelized_CAPEX_synthesis + levelized_OPEX_H2 + levelized_OPEX_synthesis + levelized_CAPEX_CO2_comp + levelized_CAPEX_H2_comp

    costs_power = Ppenalty*1000*x["celc"] / annual_CO2 # [EUR/tCO2] 
    revenues_heat = - Qpenalty*1000*x["celc"]*x["cheat"] / annual_CO2 # [EUR/tCO2] 
    revenues_methanol = (Qmethanol*1000 * 3600 / 21.1 /1000)*x["pmethanol"] / annual_CO2 # [EUR/tCO2] [Beiron, Qmethanol[MW LHV]=> tons of methanol] 
    energy_revenues = revenues_heat + revenues_methanol - costs_power # [EUR/tCO2]
    print(f"costs_power: {costs_power:.2f} EUR/tCO2")
    print(f"revenues_heat: {revenues_heat:.2f} EUR/tCO2")
    print(f"revenues_methanol: {revenues_methanol:.2f} EUR/tCO2")
    print(f"energy_revenues: {energy_revenues:.2f} EUR/tCO2")

    fossil = plant["Fossil"] / plant["Total"]                               # [tfossil/t] share of fossil CO2
    bid = CAC - energy_revenues # [EUR/tCO2]
    print(f"CAC: {CAC:.2f} EUR/tCO2")
    print(f"ETS: {x['ETS']*fossil:.2f} EUR/tCO2")
    print(f"energy_revenues: {energy_revenues:.2f} EUR/tCO2")
    print(f"bid: {bid:.2f} EUR/tCO2")

    biogenic = 1 - fossil  
    FCCU = mcaptured*10**-6*3600 * x["FLH"] * fossil                           # [ktCO2/yr]
    BCCU = mcaptured*10**-6*3600 * x["FLH"] * biogenic                        # [ktCO2/yr]
    CCU = FCCU + BCCU                                                         # [ktCO2/yr]

    if plot_single:
        fig1 = plot_CCU_CHP(
            plant=plant,
            P=P,
            Qdh=Qdh,
            Preb=Preb,
            Pcapture=Pcapture,
            PH2=PH2,
            Wcomp_CO2=Wcomp_CO2,
            Wcomp_H2=Wcomp_H2,
            Qdhreb=Qdhreb,
            Qhex=Qhex,
            Qcool_CO2=Qcool_CO2,
            Qrec_H2=Qrec_H2,
            Qcool_H2=Qcool_H2,
            Qdistill=Qdistill
        )
        
        fig2 = plot_CCU_combined(
            plant=plant,
            P=P,
            Qdh=Qdh,
            Preb=Preb,
            Pcapture=Pcapture,
            PH2=PH2,
            Wcomp_CO2=Wcomp_CO2,
            Wcomp_H2=Wcomp_H2,
            Qdhreb=Qdhreb,
            Qhex=Qhex,
            Qcool_CO2=Qcool_CO2,
            Qrec_H2=Qrec_H2,
            Qcool_H2=Qcool_H2,
            Qdistill=Qdistill,
            Qmethanol=Qmethanol/x["FLH"]*1000
        )

    return CCU, bid, Ppenalty, Qpenalty, Qmethanol

def plan_CCS(plant, x, transport_costs, sea_distances):
    # burn fuel
    mfuel = plant["Qwaste"] / (x["LHV"]/3600) /3600  # [kgf/s]
    mCO2 = mfuel* x["Ccontent"] * 44/12              # [kgCO2/s]

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
    Ppenalty = (plant["P"] - P) * x["FLH"] /1000     # [GWh/yr]
    Qpenalty = (plant["Qdh"] - Qdh) * x["FLH"] /1000 # [GWh/yr]

    # Estimate CAPEX
    CAPEX, levelized_CAPEX = estimate_CAPEX(mcaptured, x)  # [kEUR, kEUR/yr, kEUR/t]

    # Calculate transport cost using sea distance
    distance = sea_distance(plant["hub"], x["destination"], sea_distances)  # [km]
    # print("<<< Missing Truck/Rail/Harbor costs >>>")
    # print(plant["hub"], x["destination"])
    
    hub_value = x[plant["hub"].lower()]              # Get the value (1, 2, or 3) for this hub
    scenario = f"{hub_value}Mt"                      # Convert to scenario name (1Mt, 2Mt, or 3Mt)
    
    scenario_type = "optimist" if x["optimism"] else "pessimist"
    scenario_key = f"{scenario_type}_{scenario}"
    # print(scenario_key)
    
    transport_cost = transport_costs[scenario_key].predict([[distance]])[0]  # [EUR/t]
    # print(f"Transport cost at {distance} km: {transport_cost:.2f} EUR/t")

    # Estimate OPEX
    OPEXfix = (CAPEX*1000 * x["OPEXfix"]) / (mcaptured/1000*3600 * x["FLH"])  # [EUR/t]
    OPEXmakeup = x["makeup"] * x["camine"]                                    # [EUR/t]
    OPEXenergy = (Ppenalty*1000*x["celc"] + Qpenalty*1000*x["celc"]*x["cheat"]) / (mcaptured/1000*3600 * x["FLH"])  # [EUR/t]
    OPEX = OPEXfix + OPEXmakeup + OPEXenergy     

    # Construct a reversed auction bid
    CAC = OPEX + levelized_CAPEX + transport_cost                        # [EUR/t]

    fossil = plant["Fossil"] / plant["Total"]                                 # [tfossil/t] share of fossil CO2
    biogenic = 1 - fossil                                                     # [tbiogenic/t] share of biogenic CO2
    incentives = fossil * x["ETS"] + biogenic * x["CRC"]                      # [EUR/t]
    bid = CAC - incentives          

    # Store detailed cost data
    cost_details = {
        'OPEXfix': OPEXfix,
        'OPEXmakeup': OPEXmakeup,
        'OPEXenergy': OPEXenergy,
        'OPEX': OPEX,
        'levelized_CAPEX': levelized_CAPEX,
        'transport_cost': transport_cost,
        'CAC': CAC,
        'fossil_incentive': fossil * x['ETS'],
        'biogenic_incentive': biogenic * x['CRC'],
        'incentives': incentives,
        'bid': bid
    }

    FCCS = mcaptured*10**-6*3600 * x["FLH"] * fossil                           # [ktCO2/yr]
    BECCS = mcaptured*10**-6*3600 * x["FLH"] * biogenic                        # [ktCO2/yr]

    return FCCS, BECCS, bid, Ppenalty, Qpenalty, cost_details

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

def plot_awarded_metrics(output):
    """
    Create a bar plot of awarded metrics with two y-axes.
    
    Args:
        output (dict): Dictionary containing the awarded metrics
    """
    # Create figure and axis with two y-axes
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax2 = ax1.twinx()
    
    # Data for plotting
    metrics = ['FCCS+BECCS', 'CCU', 'Ppenalty', 'Qpenalty', 'Qmethanol']
    values = [output['total_FCCS_BECCS'], output['total_CCU'], 
              output['total_Ppenalty'], output['total_Qpenalty'], 
              output['total_Qmethanol']]
    
    # Colors for bars
    colors = ['#1f77b4', '#2ca9b8', '#d62728', '#ff7f0e', '#9467bd']
    
    # Create bars
    bars = ax1.bar(metrics, values, color=colors)
    
    # Set labels and title
    ax1.set_xlabel('Metrics')
    ax1.set_ylabel('CO2 [ktCO2/yr]', color='#1f77b4')
    ax2.set_ylabel('Energy [GWh/yr]', color='#d62728')
    plt.title('Awarded Metrics Summary')
    
    # Set y-axis limits and colors
    ax1.tick_params(axis='y', labelcolor='#1f77b4')
    ax2.tick_params(axis='y', labelcolor='#d62728')
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}',
                ha='center', va='bottom')
    
    # Add grid
    ax1.grid(True, linestyle='--', alpha=0.3)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig('awarded_metrics.png', dpi=600, bbox_inches='tight')
    
    return fig

def plot_product_increases(products_df):
    """
    Create a bar plot of product price increases.
    
    Args:
        products_df (pd.DataFrame): DataFrame containing product information
    """
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Sort products by price increase percentage
    sorted_df = products_df.sort_values('price_increase_percent', ascending=False)
    
    # Create bars
    bars = ax.bar(sorted_df['name'], sorted_df['price_increase_percent'], 
                 color='#2ca9b8')
    
    # Set labels and title
    ax.set_xlabel('Products')
    ax.set_ylabel('Price Increase [%]')
    plt.title('Product Price Increases Due to Carbon Tax')
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom')
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.3)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save figure
    plt.savefig('product_increases.png', dpi=600, bbox_inches='tight')
    
    return fig

def plot_plant_cost_breakdown(ccs_plants):
    """
    Create a cost breakdown plot for CCS plants showing individual cost components and incentives.
    
    Args:
        ccs_plants (list): List of CCS plant dictionaries with cost_details
    """
    if not ccs_plants:
        return None
        
    # Define the specific cost categories to plot (positive y-axis)
    positive_categories = ['OPEXfix', 'OPEXmakeup', 'OPEXenergy', 'levelized_CAPEX', 'transport_cost']
    
    # Define incentive categories to plot (negative y-axis)
    negative_categories = ['fossil_incentive', 'biogenic_incentive']
    
    # Create the plot
    fig3, ax = plt.subplots(figsize=(15, 8))
    
    # Set up the x-axis positions
    plant_names = [plant['name'] for plant in ccs_plants]
    x = np.arange(len(plant_names))
    width = 0.8  # Full width for each plant bar
    
    # Define colors for different cost categories (reds and yellows)
    colors = ['#d73027', '#f46d43', '#fdae61', '#fee08b', '#ffffbf']  # Red to yellow gradient
    
    # Create stacked bars for positive costs (above x-axis)
    bottom = np.zeros(len(plant_names))
    for i, (category, color) in enumerate(zip(positive_categories, colors)):
        values = [plant['cost_details'][category] for plant in ccs_plants]
        ax.bar(x, values, width, label=category, color=color, bottom=bottom)
        bottom += values
    
    # Create bars for incentives (below x-axis)
    bottom = np.zeros(len(plant_names))  # Start at 0 for negative stacking
    incentive_colors = ['#1a9850', '#66c2a5']  # Green to blue gradient
    for i, (category, color) in enumerate(zip(negative_categories, incentive_colors)):
        values = [-plant['cost_details'][category] for plant in ccs_plants]  # Make negative for display
        ax.bar(x, values, width, label=category, color=color, bottom=bottom)
        bottom += values  # This makes bottom more negative for next bar
    
    # Customize the plot with font size 12
    ax.set_xlabel('Plants', fontsize=12)
    ax.set_ylabel('Cost [EUR/tCO2]', fontsize=12)
    ax.set_title('Cost Breakdown by Plant (CCS Plants Only)', fontsize=12)
    ax.set_xticks(x + width/2)
    ax.set_xticklabels(plant_names, rotation=45, ha='right', fontsize=12)
    ax.tick_params(axis='y', labelsize=12)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12, reverse=True)
    ax.grid(True, linestyle='--', alpha=0.3)
    
    # Add a horizontal line at y=0
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    # Add value labels for each cost component in the stacked bars
    for i, plant in enumerate(ccs_plants):
        if i != len(ccs_plants) - 1:
            continue
        # Add labels for positive costs (above x-axis)
        bottom = 0
        for j, category in enumerate(positive_categories):
            value = plant['cost_details'][category]
            if value > 0:  # Only add label if value is significant
                ax.text(x[i] + width/2, bottom + value/2, 
                       f'{int(round(value))}', ha='center', va='center', 
                       color='black', fontsize=12, fontweight='bold')
            bottom += value
        
        # Add labels for incentives (below x-axis)
        bottom = 0
        for j, category in enumerate(negative_categories):
            value = -plant['cost_details'][category]  # Make negative for display
            if abs(value) > 0:  # Only add label if value is significant
                ax.text(x[i] + width/2, bottom + value/2, 
                       f'{int(round(abs(plant["cost_details"][category])))}', ha='center', va='center', 
                       color='black', fontsize=12, fontweight='bold')
            bottom += value
        
        # Add total cost label on top of the positive bar
        total_positive_cost = sum(plant['cost_details'][cat] for cat in positive_categories)
        ax.text(x[i] + width/2, total_positive_cost, 
               f'{total_positive_cost:.1f}', ha='center', va='bottom', 
               color='black', fontsize=12, fontweight='bold')
    
    # Add bid values as black dots at their actual bid values on the y-axis
    for i, plant in enumerate(ccs_plants):
        # Plot the bid value as a black dot at the actual bid value, centered on the bar
        ax.scatter(x[i], plant['bid'], 
                  color='gray', s=100, zorder=5, marker='o')
        
        # Add bid value label next to the dot
        ax.text(x[i] + 0.1, plant['bid'], 
               f'{round(plant["bid"]):.1f}', ha='left', va='center', 
               color='gray', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('plant_cost_breakdown.png', dpi=600, bbox_inches='tight')
    
    return fig3

def WACCUS_EPR( 
    # uncertainties
    mKN39 = 884393,     # [t/a] plastic products mappable under KN39 [IVL]
    pKN39 = 46000,      # [SEK/tpl] [IVL]
    recyclable = 0.15,  # [-] fraction of simple products possible to recycle mechanically
    fraction = 0.75,    # [-] fraction of carbon in plastic [Isabel]

    LHV = 11,           # [MJ/kgf] [Hammar]
    Ccontent = 0.298,   # [kgC/kgf]
    fossil = 0.40,      # [-] NOTE: assumption not needed: emissions data is available and used
    qreb = 3.5,         # [MJ/kgCO2]
    COP = 3,

    FLH = 8000,
    dr = 0.075,
    t = 25,
    # FOAK = 0.45/2,      # [-] [Beiron, 2024] applies to CO2 capture and conditioning 
    FOAK = 0, # FOR EON MEETING
    OPEXfix = 0.03,     # [-] [Beiron, 2024] % of CAPEX no, calculated from Ramboll BECCS Malmö [2-5%]
    camine = 2000,      # [EUR/m3] [Beiron, 2024]
    celc = 60,          # [EUR/MWh]
    cheat = 0.80,       # [% of elc]
    CRC = 100,          # [EUR/tCO2]
    ETS = 80,           # [EUR/tCO2]
    pmethanol = 625,    # [EUR/t] NOTE: CHECK CCU THESIS FOR PRICE ESTIMATES, e.g. 1750 EUR/t... NO now they say 700?

    lulea = 1,          # [1,2,3] [Mt/yr]
    sundsvall = 1,      # [1,2,3] [Mt/yr]
    stockholm = 1,      # [1,2,3] [Mt/yr]
    malmo = 1,          # [1,2,3] [Mt/yr]
    gothenburg = 1,     # [1,2,3] [Mt/yr]
    optimism = False,   # [True, False]
    destination = "oygarden",  # ["oygarden", "kalundborg"]
    heat_optimism = 1,  # [0,1] assumed % of waste heat that can be recovered to DH

    # levers 
    tax = 110,          # [EUR/tCO2]

    # constants
    plants = None,
    case = ["CCUS"],    # ["CCUS", "CCU", "CCS"] conduct analysis separately for the three cases
    k = 0.6857,         # [-] [Stenström, 2025]
    CAPEX_ref = 3715 * 87,  # [MNOK] -> [kEUR] NOTE: can pick other CELSIO CAPEX from source:[Gassnova, Demonstrasjon av Fullskala CO2-Håndtering - Rapport for Avsluttet Forprosjekt]
    captured_ref = 400,     # [ktCO2/yr]
    transport_costs = None,  # Dictionary of transport cost interpolation functions
    sea_distances = None,    # Dictionary of pre-calculated sea distances
    thermo_props = None,     # Dictionary of thermodynamic properties

    SEK_TO_EUR = 0.091,     # [EUR/SEK] Exchange rate
    makeup = 0.584/1000,    # [m3/tCO2] [Kumar, 2023]
    etais = 0.80,           # [-]
):
    x = {
        "mKN39": mKN39,
        "recyclable": recyclable,
        "pKN39": pKN39,
        "fraction": fraction,
        
        "LHV": LHV,
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
        "pmethanol": pmethanol,
        "heat_optimism": heat_optimism,

        "tax": tax,
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

    # Plot fund values as a function of tax level for two cases
    tax_levels = np.linspace(50, 300, 100)
    mass_taxed_1 = 1.2
    mass_taxed_2 = 0.6 # [Mtpl /yr]

    fund_1 = mass_taxed_1 * fraction * 3.66 * tax_levels  # [MEUR/yr]
    fund_2 = mass_taxed_2 * fraction * 3.66 * tax_levels  # [MEUR/yr]

    plt.figure()
    plt.plot(tax_levels, fund_1, label=f"mass_taxed = {mass_taxed_1:.1f} Mt plastic/yr")
    plt.plot(tax_levels, fund_2, label=f"mass_taxed = {mass_taxed_2:.1f} Mt plastic/yr")
    plt.axhline(120, color='red', linestyle='--', label='120 MEUR/yr = Exergi subsidy')
    plt.xlabel("Tax level [EUR/tCO2]", fontsize=14)
    plt.ylabel("Fund [MEUR/yr]", fontsize=14)
    plt.title("Fund as function of tax level", fontsize=14)
    plt.legend(fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('fund_potential.png', dpi=600, bbox_inches='tight')

    mass_taxed = mKN39 * (1 - recyclable) * fraction # [tC/yr]
    mass_CO2 = mass_taxed * 3.66 # [tCO2/yr]
    fund = mass_CO2 * tax # [EUR/yr]
    fund /= 10**6 # [MEUR/yr]

    # Forcing an example for EON:
    mass_CO2 = 1.2 * fraction * 3.66 # [MtCO2/yr]
    fund = mass_CO2 * tax # [MEUR/yr]

    products_df = pd.read_csv("data/products.csv")
    products_df['tax_amount'] = products_df['weight per unit [kg]']/1000 * (products_df['fossil carbon content [mass%]']/100) * 3.66 * tax  # [EUR]
    products_df['price_increase_percent'] = (products_df['tax_amount'] / products_df['price per unit [EUR]']) * 100  # [%]
    
    print("\nProduct Price Increases:")
    print(products_df[['name', 'price per unit [EUR]', 'tax_amount', 'price_increase_percent']].to_string())

    # RQ2: subsidy costs
    if case == "CCUS":
        CCU_names = ["Renova","SAKAB","Filbornaverket","Garstadverket","Sjolunda"]
        CCS_names = ["Handeloverket","Bristaverket","Vasteras KVV","Hogdalenverket","Bolanderna"]
    elif case == "CCU":
        # CCU_names = ["Renova","SAKAB","Filbornaverket","Garstadverket","Sjolunda","Handeloverket","Bristaverket","Vasteras KVV","Hogdalenverket","Bolanderna"]
        CCU_names = ["Renova"]
        CCS_names = []
    elif case == "CCS":
        CCU_names = []
        CCS_names = ["Renova","Hogdalenverket","Sjolunda","Korstaverket","Garstadverket","Vasteras KVV","Handeloverket","Bolanderna","Filbornaverket","Bristaverket"]

    # Collect all bids
    bids = []
    for _, plant in plants.iterrows():
        if plant["Name"] in CCS_names:
            FCCS, BECCS, bid, Ppenalty, Qpenalty, cost_details = plan_CCS(plant, x, transport_costs, sea_distances)
            bids.append({
                'name': plant['Name'],
                'type': 'CCS',
                'bid': bid,
                'FCCS': FCCS,
                'BECCS': BECCS,
                'Ppenalty': Ppenalty,
                'Qpenalty': Qpenalty,
                'cost_details': cost_details
            })

        if plant["Name"] in CCU_names:
            CCU, bid, Ppenalty, Qpenalty, Qmethanol = plan_CCU(plant, x)
            bids.append({
                'name': plant['Name'],
                'type': 'CCU',
                'bid': bid,
                'CCU': CCU,
                'Ppenalty': Ppenalty,
                'Qpenalty': Qpenalty,
                'Qmethanol': Qmethanol
            })

    # Sort bids by bid amount (ascending)
    bids.sort(key=lambda x: x['bid'])

    # Initialize results
    remaining_fund = fund
    awarded_plants = []

    print("\nAuction Results:")
    print(f"{'Plant Name':<20} {'Type':<6} {'Bid':>10} {'Awarded':>12} {'Remaining Fund':>15}")
    print("-" * 70)

    # Distribute funds
    for bid in bids:
        if bid['type'] == 'CCS':
            requested_amount = bid['bid'] * (bid['FCCS'] + bid['BECCS']) * 1000 /(10**6) # [EUR/tCO2 * ktCO2/yr => MEUR/yr]
            awarded = requested_amount <= remaining_fund # [MEUR/yr]
            if awarded:
                remaining_fund -= requested_amount
            awarded_plants.append({
                'name': bid['name'],
                'type': bid['type'],
                'awarded': awarded,
                'bid': bid['bid'],
                'FCCS': bid['FCCS'],
                'BECCS': bid['BECCS'],
                'FCCS+BECCS': bid['FCCS'] + bid['BECCS'],
                'Ppenalty': bid['Ppenalty'],
                'Qpenalty': bid['Qpenalty'],
                'CCU': 0,
                'Qmethanol': 0,
                'amount': requested_amount if awarded else 0,
                'cost_details': bid.get('cost_details', {})
            })
        else:  # CCU
            requested_amount = bid['bid'] * bid['CCU'] * 1000 /(10**6)
            awarded = requested_amount <= remaining_fund
            if awarded:
                remaining_fund -= requested_amount
            awarded_plants.append({
                'name': bid['name'],
                'type': bid['type'],
                'awarded': awarded,
                'bid': bid['bid'],
                'FCCS': 0,
                'BECCS': 0,
                'FCCS+BECCS': 0,
                'Ppenalty': bid['Ppenalty'],
                'Qpenalty': bid['Qpenalty'],
                'CCU': bid['CCU'],
                'Qmethanol': bid['Qmethanol'],
                'amount': requested_amount if awarded else 0
            })

        print(f"{bid['name']:<20} {bid['type']:<6} {bid['bid']:>10.2f} {requested_amount:>12.2f} {remaining_fund:>15.2f}")

    print("\nSummary:")
    print(f"Total fund: {fund:.2f} MEUR/yr")
    print(f"Remaining fund: {remaining_fund:.2f} MEUR/yr")
    print(f"Number of plants awarded: {len([p for p in awarded_plants if p['awarded']])}")
    print("\nAwarded plants:")
    print(f"{'Plant Name':<20} {'Type':<6} {'Awarded':<6} {'[EUR/t]':>10} {'[MEUR/yr]':>10} {'FCCS':>10} {'BECCS':>10} {'FCCS+BECCS':>12} {'CCU':>10} {'Ppenalty':>12} {'Qpenalty':>12} {'Qmethanol':>12}")
    print("-" * 140)
    for plant in awarded_plants:
        print(f"{plant['name']:<20} {plant['type']:<6} {str(plant['awarded']):<6} {plant['bid']:>10.2f} {plant['amount']:>10.2f} {plant['FCCS']:>10.2f} {plant['BECCS']:>10.2f} {plant['FCCS+BECCS']:>12.2f} {plant['CCU']:>10.2f} {plant['Ppenalty']:>12.2f} {plant['Qpenalty']:>12.2f} {plant['Qmethanol']:>12.2f}")

    # Calculate sums of awarded metrics
    total_FCCS_BECCS = sum(plant['FCCS+BECCS'] for plant in awarded_plants if plant['awarded'])
    total_CCU = sum(plant['CCU'] for plant in awarded_plants if plant['awarded'])
    total_Ppenalty = sum(plant['Ppenalty'] for plant in awarded_plants if plant['awarded'])
    total_Qpenalty = sum(plant['Qpenalty'] for plant in awarded_plants if plant['awarded'])
    total_Qmethanol = sum(plant['Qmethanol'] for plant in awarded_plants if plant['awarded'])

    print("\nTotal awarded metrics:")
    print(f"FCCS+BECCS: {total_FCCS_BECCS:.2f} ktCO2/yr")
    print(f"CCU: {total_CCU:.2f} ktCO2/yr")
    print(f"Ppenalty: {total_Ppenalty:.2f} GWh/yr")
    print(f"Qpenalty: {total_Qpenalty:.2f} GWh/yr")
    print(f"Qmethanol: {total_Qmethanol:.2f} GWh/yr")

    output = {
        'total_FCCS_BECCS': total_FCCS_BECCS,
        'total_CCU': total_CCU,
        'total_Ppenalty': total_Ppenalty,
        'total_Qpenalty': total_Qpenalty,
        'total_Qmethanol': total_Qmethanol,
        'remaining_fund': remaining_fund,
        'product_increases': products_df[['name', 'price per unit [EUR]', 'tax_amount', 'price_increase_percent']].to_dict('records'),
        'bid_data': bids,
    }
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
    # fig, ax = plot_transport_costs(transport_costs, r2_scores, df, show_plot=False)
    
    # reading plant data and assign these to transport hubs
    plants = pd.read_csv("data/plants.csv")
    plants = assign_hub(plants)
    print("\nPlant to Hub Assignments:")
    print(plants[['Name', 'hub', 'distance_to_hub']].to_string())
    
    # Run the model
    output = WACCUS_EPR(
        plants=plants, 
        k=k,                          # For CAPEX=CAPEX_ref⋅(mCO2/mCO2_ref)^k
        case="CCU", 
        transport_costs=transport_costs,
        sea_distances=sea_distances,  # Pass pre-calculated distances dict
        thermo_props=thermo_props     # Pass thermodynamic properties
    )
    
    # Create and save the plots
    fig1 = plot_awarded_metrics(output)

    # Plot cost breakdown for each plant
    bid_data = output['bid_data']
    
    # Filter for CCS plants that have cost_details
    ccs_plants = [bid for bid in bid_data if bid['type'] == 'CCS' and 'cost_details' in bid]
    positive_categories = ['OPEXfix', 'OPEXmakeup', 'OPEXenergy', 'levelized_CAPEX', 'transport_cost']    
    negative_categories = ['fossil_incentive', 'biogenic_incentive']
    fig3 = plot_plant_cost_breakdown(ccs_plants)
    
    print("Most CCU plants are near profitable already - consider adding higher electrolyzer costs etc., also no ETS incentive?")
    print("I should verify the methanol production - is it reasonable really?")

    # print("--- Feedback from Johanna/Judit ---")
    # print("The KPIs are ish similar to those in Johanna's study. However, I am to optimistic about recovering waste heat.")
    # print("Compressors: These cannot increase to the Tsynthesis, they should be reduced to 40C ish in every stage")
    # print("-> However, I might neglect that issue. Otherwise, I must add a heater to re-heat the CO2 and H2 entering the synthesis")
    # print("The largest heat recovery units are: from the amine capture plant (reboiler), ALK electrolyzer, and the methanol synthesis.")
    # print("-> These deserve greater attention - mainly, I must verify that the heat temperatures are ok for DH applications")
    # print("-> Check how much heat the synthesis produces, it should be transferred to the destillation first, and THEN MAYBE some of this remaining heat is available for either the qreboiler (exciting option!) or for DH (but check temperatures)")
    # print("-> It is far from certain that you can recover this much heat from the destillation - its not necesarily true that 80percent of the energy in raw methanol becomes methanol and that 20percent to DH.")
    # print("----> Check with Johanna for sources on that, possibly? Also, we need a pure destillation (cannot do without) since we use it for plastic/chemicals")
    # print("-> It's uncertain how much heat can be recovered from electrolyzers - ask Tharun! He has the percentage and termperatures ... or maybe the MSc thesis")
    # print("Finally, double check the conversion from H2 to methanol in synthesis - can I do it as an energy conversion, 100prc H2 to 100prc methanol? Do it via reaciton formula and LHV values as well! Check!")
    # print("-> This is suspicious because one H2 should be lost (mass-wise) for each methanol molecule produced: CO2+3H2=>CH3OH+H2O")
    # print(" .... She thinks I MUST DO EXERGY ANALYSIS... since heat is different... well, I don't think so, since I only need MONEY analysis!")
    print(" ")
    print("I should ask EON about the feasibility of CCU - seems like power costs are higher than methanol revenues?!")
    plt.show()

