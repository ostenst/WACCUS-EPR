import CoolProp.CoolProp as CP
import numpy as np
import matplotlib.pyplot as plt

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

def compression_energy(mcaptured, T1, P1, gas_type='CO2', n_stages=4, pressure_ratio=3.0, Tdiff=0, thermo_props=None, etais=0.8, printing=False):
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
        tuple: (Wcomp_list, Qcool_list, P_list, T_list, T_actual_intermediate_list)
            Wcomp_list: List of compression work for each stage [MW]
            Qcool_list: List of cooling requirements for each stage [MW]
            P_list: List of pressures at each stage [bar]
            T_list: List of temperatures at each stage [K]
            T_actual_intermediate_list: List of lists containing intermediate T_actual values for each stage
    """
    Wcomp_list = []
    Qcool_list = []
    P_list = [P1]
    T_list = [T1]
    T_actual_intermediate_list = []  # New list to store intermediate T_actual values for each stage
    
    T = T1
    P = P1
    
    for stage in range(n_stages):
        # Get properties at current temperature
        kappa = get_property_at_temp(thermo_props, gas_type, T, 'kappa')
        cp_in = get_property_at_temp(thermo_props, gas_type, T, 'cp')
        
        # Calculate next pressure
        P_next = P * pressure_ratio
        P_list.append(P_next)
        
        # Calculate intermediate pressures and temperatures for this stage
        # Create pressure steps from current P to P_next
        n_intermediate_steps = 10  # Number of intermediate pressure steps
        P_intermediate = np.linspace(P, P_next, n_intermediate_steps + 1)
        T_actual_intermediate = []
        
        # Calculate compression temperatures
        for i in range(len(P_intermediate)):
            if i == 0:
                # First point is the current state
                T_actual_intermediate.append(T)
            else:
                # Calculate actual temperature for this pressure level
                P_current = P_intermediate[i]
                T_isentropic_intermediate = T * (P_current/P)**((kappa-1)/kappa)
                T_actual_intermediate.append(T + (T_isentropic_intermediate - T)/etais)

        # Add cooling phase (constant pressure, decreasing temperature)
        T_final_compression = T_actual_intermediate[-1]  # Final compression temperature
        T_cooling_hardcoded = np.linspace(T_final_compression, T + Tdiff, n_intermediate_steps + 1)
        
        # Extend pressure array for cooling phase (constant pressure at P_next)
        P_cooling = np.full(n_intermediate_steps + 1, P_next)
        P_intermediate = np.concatenate([P_intermediate, P_cooling])
        
        # Extend temperature array with cooling temperatures
        T_actual_intermediate.extend(T_cooling_hardcoded)
        
        # Store intermediate temperatures for this stage
        T_actual_intermediate_list.append(T_actual_intermediate)
        
        # Calculate final isentropic and actual temperatures (at P_next)
        T_isentropic = T * (P_next/P)**((kappa-1)/kappa)
        T_actual = T + (T_isentropic - T)/etais
        
        
        # Get properties at actual temperature
        cp_out = get_property_at_temp(thermo_props, gas_type, T_actual, 'cp')
        
        # Calculate work and cooling
        Wcomp = mcaptured * (cp_in + cp_out)/2 * (T_actual - T)  # [kJ/s]
        
        # Calculate cooling only if not the last stage
        Qcool = 0 if stage == n_stages - 1 else mcaptured * cp_out * (T_actual - (T + Tdiff))    # [kJ/s] NOTE: Compressor temps should not increase... but it does in Beiron?
        T_list.append(T+Tdiff) # The new temperature

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
    
    return Wcomp_list, Qcool_list, P_list, T_list, T_actual_intermediate_list

def plot_compression_curves(P_list, T_actual_intermediate_list, gas_type='CO2'):
    """
    Plot pressure vs T_actual for each compression stage.
    
    Args:
        P_list: List of pressures at each stage [bar]
        T_actual_intermediate_list: List of lists containing intermediate T_actual values for each stage
        gas_type: Type of gas for labeling
    """
    plt.figure(figsize=(10, 6))
    
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
    
    for stage, T_intermediate in enumerate(T_actual_intermediate_list):
        # Create pressure array for this stage
        P_start = P_list[stage]
        P_end = P_list[stage + 1]
        
        # The pressure array now includes both compression and cooling phases
        # Compression phase: P_start to P_end
        # Cooling phase: constant at P_end
        n_compression_points = len(T_intermediate) // 2  # Half for compression, half for cooling
        P_compression = np.linspace(P_start, P_end, n_compression_points)
        P_cooling = np.full(len(T_intermediate) - n_compression_points, P_end)
        P_intermediate = np.concatenate([P_compression, P_cooling])
        
        # Convert temperatures to Celsius for plotting
        T_celsius = [T - 273.15 for T in T_intermediate]
        
        # Plot compression phase (inverted axes: T vs P)
        plt.plot(T_celsius[:n_compression_points], P_compression, 
                color=colors[stage % len(colors)], 
                linewidth=2, 
                marker='o', 
                markersize=4,
                label=f'Stage {stage + 1} - Compression')
        
        # Plot cooling phase (inverted axes: T vs P)
        plt.plot(T_celsius[n_compression_points:], P_cooling, 
                color=colors[stage % len(colors)], 
                linewidth=2, 
                marker='s', 
                markersize=4,
                linestyle='--',
                label=f'Stage {stage + 1} - Cooling')
    
    plt.xlabel('Temperature [°C]', fontsize=12)
    plt.ylabel('Pressure [bar]', fontsize=12)
    plt.title(f'{gas_type} Compression & Cooling: Temperature vs Pressure for Each Stage', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('compression_curves.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_co2_phase_diagram():
    """
    Simple plot of CO2 phase diagram using CoolProp's build_phase_envelope function.
    """
    # Create the plot
    plt.figure(figsize=(10, 8))
    
    # Build phase envelope using CoolProp
    PE = CP.AbstractState('HEOS', 'CO2')
    PE.build_phase_envelope('')
    
    # Get phase envelope data
    T_pe = np.array(PE.get_phase_envelope_data().T)
    P_pe = np.array(PE.get_phase_envelope_data().p) / 1e5  # Convert to bar
    
    # Plot the phase envelope
    plt.plot(T_pe, P_pe, 'b-', linewidth=2, label='Phase Envelope')
    
    # Get critical point
    T_critical = CP.PropsSI('Tcrit', 'CO2')
    P_critical = CP.PropsSI('Pcrit', 'CO2') / 1e5  # Convert to bar
    
    # Mark critical point
    plt.plot(T_critical, P_critical, 'ko', markersize=8, label='Critical Point')
    plt.annotate('Critical Point', (T_critical, P_critical), 
                xytext=(10, 10), textcoords='offset points', fontsize=10)
    
    # Add labels
    plt.xlabel('Temperature [K]', fontsize=12)
    plt.ylabel('Pressure [bar]', fontsize=12)
    plt.title('CO2 Phase Diagram', fontsize=14)
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Add phase labels
    plt.text(220, 10, 'Gas', fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
    plt.text(300, 50, 'Liquid', fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
    plt.text(350, 0.5, 'Supercritical', fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))
    
    plt.tight_layout()
    plt.savefig('co2_phase_diagram.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    thermo_props = get_thermo_properties()
    Wcomp_list, Qcool_list, P_list, T_list, T_actual_intermediate_list = compression_energy(
        16, 
        40+273.15, 
        1, 
        gas_type='CO2', 
        n_stages=3, 
        pressure_ratio=2, 
        Tdiff=0, 
        thermo_props=thermo_props, 
        etais=0.8, 
        printing=True)
    
    print("Main results:", Wcomp_list, Qcool_list, P_list, T_list)
    print("\nIntermediate T_actual values for each stage:")
    for stage, T_intermediate in enumerate(T_actual_intermediate_list):
        print(f"Stage {stage + 1}: {T_intermediate}")
    
    # Create the plot
    plot_compression_curves(P_list, T_actual_intermediate_list, 'CO2')
    
    # Create CO2 phase diagram
    plot_co2_phase_diagram()

if __name__ == "__main__":
    main()
