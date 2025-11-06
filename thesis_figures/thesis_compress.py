import CoolProp.CoolProp as CP
import numpy as np
import matplotlib.pyplot as plt

color = '#CF73A4'

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
    # plt.savefig('compression_curves.png', dpi=300, bbox_inches='tight')
    # plt.show()

def plot_co2_phase_diagram():
    """
    Simple plot of CO2 phase diagram showing different phase boundaries.
    """
    fluid = 'CO2'
    
    # Get critical and triple point properties
    Tcrit = CP.PropsSI('Tcrit', fluid)
    Pcrit = CP.PropsSI('Pcrit', fluid) / 1e5  # Convert to bar
    
    # CO2 triple point (known values)
    Ttrip = 216.58  # K
    Ptrip = 5.18    # bar
    
    # Create the plot
    plt.figure(figsize=(10, 8))
    
    # 1. Saturation line (liquid-gas boundary)
    T_range = np.linspace(Ttrip, Tcrit, 200)
    p_sat = []
    for T in T_range:
        try:
            p = CP.PropsSI("P", "T", T, "Q", 0, fluid) / 1e5  # Q=0: saturated liquid
            p_sat.append(p)
        except:
            p_sat.append(np.nan)
    
    plt.semilogy(T_range, p_sat, 'b-', linewidth=2, label="Saturation line")
    
    # 2. Sublimation line (solid-gas boundary)
    # Note: CoolProp does NOT support sublimation pressure directly
    # Using empirical correlation for CO2: log10(P) = A - B/T (P in bar, T in K)
    # Based on experimental data for CO2 sublimation
    # Triple point: T=216.58 K, P=5.18 bar
    T_subl = np.linspace(194, Ttrip, 100)  # Below triple point (down to ~-79°C)
    p_subl = []
    for T in T_subl:
        try:
            # Empirical correlation: log10(P/bar) = 10.955 - 1351/T (valid for ~194-216K)
            # Better: ln(P) = a*ln(T) + b (smoother across range)
            # Using Clausius-Clapeyron form: ln(P/Ptrip) = (Lsub/R) * (1/Ttrip - 1/T)
            # Lsub (sublimation enthalpy) ≈ 29.5 kJ/mol for CO2
            Lsub = 29500  # J/mol
            R = 8.314  # J/mol/K
            p = Ptrip * np.exp((Lsub / R) * (1/Ttrip - 1/T))
            p_subl.append(p)
        except:
            p_subl.append(np.nan)
    
    plt.semilogy(T_subl, p_subl, 'g-', linewidth=2, label="Sublimation line")
    
    # 3. Melting line (solid-liquid boundary) - approximate
    T_melt = np.linspace(Ttrip, 300, 50)
    p_melt = []
    for T in T_melt:
        try:
            # Approximate melting pressure (increases with temperature)
            p = Ptrip + (T - Ttrip) * 100  # Rough approximation
            p_melt.append(p)
        except:
            p_melt.append(np.nan)
    
    plt.semilogy(T_melt, p_melt, 'r-', linewidth=2, label="Melting line")
    
    # 4. Mark critical point
    plt.scatter([Tcrit], [Pcrit], color="red", s=100, label="Critical point")
    plt.annotate('Critical Point', (Tcrit, Pcrit), 
                xytext=(10, 10), textcoords='offset points', fontsize=10)
    plt.hlines(Pcrit, xmin=218, xmax=385, colors='grey', linestyles='dashed')
    plt.vlines(Tcrit, ymin=25, ymax=90, colors='grey', linestyles='dashed')

    # 5. Mark triple point
    plt.scatter([Ttrip], [Ptrip], color="blue", s=100, label="Triple point")
    plt.annotate('Triple Point', (Ttrip, Ptrip), 
                xytext=(10, -15), textcoords='offset points', fontsize=10)


    # Add labels and formatting
    plt.xlabel('Temperature [K]', fontsize=12)
    plt.ylabel('Pressure [bar]', fontsize=12)
    plt.title('CO2 Phase Diagram', fontsize=14)
    plt.xlim(200, 410)  # Set x-axis limit to 400 K
    plt.ylim(0.01, 200)  # Set y-axis limit to 1000 bar
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Add phase region labels
    # plt.text(220, 10, 'Gas', fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
    # plt.text(300, 50, 'Liquid', fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
    # plt.text(350, 0.5, 'Supercritical', fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))
    # plt.text(200, 0.1, 'Solid', fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
    
    plt.tight_layout()
    # plt.savefig('co2_phase_diagram.png', dpi=300, bbox_inches='tight')
    # plt.show()
    
    # Create second figure with linear y-axis
    plt.figure(figsize=(10, 8))
    
    # Plot the same phase boundaries
    plt.plot(T_range, p_sat, 'b-', linewidth=2, label="Saturation line")
    plt.plot(T_subl, p_subl, 'g-', linewidth=2, label="Sublimation line")
    plt.plot(T_melt, p_melt, 'r-', linewidth=2, label="Melting line")
    
    # Mark critical point
    plt.scatter([Tcrit], [Pcrit], color="red", s=100, label="Critical point")
    plt.annotate('Critical Point', (Tcrit, Pcrit), 
                xytext=(10, 10), textcoords='offset points', fontsize=10)
    plt.hlines(Pcrit, xmin=218, xmax=385, colors='grey', linestyles='dashed')
    plt.vlines(Tcrit, ymin=25, ymax=90, colors='grey', linestyles='dashed')
    
    # Mark triple point
    plt.scatter([Ttrip], [Ptrip], color="blue", s=100, label="Triple point")
    plt.annotate('Triple Point', (Ttrip, Ptrip), 
                xytext=(10, -15), textcoords='offset points', fontsize=10)
    
    # Add labels and formatting (linear y-axis)
    plt.xlabel('Temperature [K]', fontsize=12)
    plt.ylabel('Pressure [bar]', fontsize=12)
    plt.title('CO2 Phase Diagram (Linear Scale)', fontsize=14)
    plt.xlim(200, 410)  # Set x-axis limit to 400 K
    plt.ylim(0.01, 200)  # Set y-axis limit to 1100 bar
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    # plt.savefig('co2_phase_diagram_linear.png', dpi=300, bbox_inches='tight')
    # plt.show()

def plot_compression_on_phase_diagram(P_list, T_actual_intermediate_list, thermo_props, gas_type='CO2'):
    """
    Plot CO2 phase diagram with compression curves overlaid.
    """
    fluid = 'CO2'
    
    # Get critical and triple point properties
    Tcrit = CP.PropsSI('Tcrit', fluid)
    Pcrit = CP.PropsSI('Pcrit', fluid) / 1e5  # Convert to bar
    
    # CO2 triple point (known values)
    Ttrip = 216.58  # K
    Ptrip = 5.18    # bar
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # 1. Saturation line (liquid-gas boundary)
    T_range = np.linspace(Ttrip, Tcrit, 200)
    p_sat = []
    for T in T_range:
        try:
            p = CP.PropsSI("P", "T", T, "Q", 0, fluid) / 1e5  # Q=0: saturated liquid
            p_sat.append(p)
        except:
            p_sat.append(np.nan)
    
    plt.plot(T_range, p_sat, 'k-', linewidth=3, label="Saturation line", alpha=0.7)
    
    # 2. Sublimation line (solid-gas boundary)
    # Note: CoolProp does NOT support sublimation pressure directly
    # Using empirical correlation for CO2: log10(P) = A - B/T (P in bar, T in K)
    # Based on experimental data for CO2 sublimation
    # Triple point: T=216.58 K, P=5.18 bar
    T_subl = np.linspace(194, Ttrip, 100)  # Below triple point (down to ~-79°C)
    p_subl = []
    for T in T_subl:
        try:
            # Empirical correlation: log10(P/bar) = 10.955 - 1351/T (valid for ~194-216K)
            # Better: ln(P) = a*ln(T) + b (smoother across range)
            # Using Clausius-Clapeyron form: ln(P/Ptrip) = (Lsub/R) * (1/Ttrip - 1/T)
            # Lsub (sublimation enthalpy) ≈ 29.5 kJ/mol for CO2
            Lsub = 29500  # J/mol
            R = 8.314  # J/mol/K
            p = Ptrip * np.exp((Lsub / R) * (1/Ttrip - 1/T))
            p_subl.append(p)
        except:
            p_subl.append(np.nan)
    
    plt.plot(T_subl, p_subl, 'k-', linewidth=3, label="Sublimation line", alpha=0.7)
    
    # 3. Melting line (solid-liquid boundary) - approximate
    T_melt = np.linspace(Ttrip, 300, 50)
    p_melt = []
    for T in T_melt:
        try:
            # Approximate melting pressure (increases with temperature)
            p = Ptrip + (T - Ttrip) * 100  # Rough approximation
            p_melt.append(p)
        except:
            p_melt.append(np.nan)
    
    plt.plot(T_melt, p_melt, 'k-', linewidth=3, label="Melting line", alpha=0.7)
    
    # 4. Mark critical point
    plt.scatter([Tcrit], [Pcrit], color="black", s=100, label="Critical point", zorder=5)
    plt.annotate('Critical Point', (Tcrit, Pcrit), 
                xytext=(10, 10), textcoords='offset points', fontsize=10)
    plt.hlines(Pcrit, xmin=218, xmax=385, colors='grey', linestyles='dashed', alpha=0.5)
    plt.vlines(Tcrit, ymin=25, ymax=90, colors='grey', linestyles='dashed', alpha=0.5)
    
    # 5. Mark triple point
    plt.scatter([Ttrip], [Ptrip], color="black", s=100, label="Triple point", zorder=5)
    plt.annotate('Triple Point', (Ttrip, Ptrip), 
                xytext=(10, -15), textcoords='offset points', fontsize=10)
    
    # 6. Overlay compression curves
    colors_comp = ['orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan', 'magenta']
    
    for stage, T_intermediate in enumerate(T_actual_intermediate_list):
        # Create pressure array for this stage
        P_start = P_list[stage]
        P_end = P_list[stage + 1]
        
        # The pressure array now includes both compression and cooling phases
        n_compression_points = len(T_intermediate) // 2  # Half for compression, half for cooling
        P_compression = np.linspace(P_start, P_end, n_compression_points)
        P_cooling = np.full(len(T_intermediate) - n_compression_points, P_end)
        
        # Convert temperatures to Kelvin for plotting
        T_kelvin = [T for T in T_intermediate]
        
        # Plot compression phase
        plt.plot(T_kelvin[:n_compression_points], P_compression, 
                color=colors_comp[stage % len(colors_comp)], 
                linewidth=2, 
                marker='o', 
                markersize=6,
                label=f'Stage {stage + 1} - Compression', zorder=4)
        
        # Plot cooling phase
        plt.plot(T_kelvin[n_compression_points:], P_cooling, 
                color=colors_comp[stage % len(colors_comp)], 
                linewidth=2, 
                marker='s', 
                markersize=6,
                linestyle='--',
                label=f'Stage {stage + 1} - Cooling', zorder=4)
    
    # Add labels and formatting
    plt.xlabel('Temperature [K]', fontsize=12)
    plt.ylabel('Pressure [bar]', fontsize=12)
    plt.title(f'{gas_type} Phase Diagram with Compression Process', fontsize=14)
    plt.xlim(200, 410)  # Set x-axis limit to 400 K
    plt.ylim(0.01, 200)  # Set y-axis limit to 100 bar
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    # plt.savefig('co2_phase_diagram_with_compression.png', dpi=300, bbox_inches='tight')
    # plt.show()
    
    # ------------------- Create second figure with logarithmic y-axis -------------------
    plt.figure(figsize=(16, 8))
    
    # Plot the same phase boundaries (convert to Celsius)
    plt.plot(T_range-273.15, p_sat, 'k-', linewidth=2, label="Saturation line", alpha=0.7)
    plt.plot(T_subl-273.15, p_subl, 'k-', linewidth=2, label="Sublimation line", alpha=0.7)
    plt.plot(T_melt-273.15, p_melt, 'k-', linewidth=2, label="Melting line", alpha=0.7)
    
    # Mark critical point
    plt.scatter([Tcrit-273.15], [Pcrit], color="black", s=100, label="Critical point", zorder=5)
    plt.annotate('Critical Point', (Tcrit-273.15, Pcrit), 
                xytext=(10, 10), textcoords='offset points', fontsize=14)
    plt.hlines(Pcrit, xmin=218-273.15, xmax=385-273.15, colors='grey', linestyles='dashed', alpha=0.5)
    plt.vlines(Tcrit-273.15, ymin=25, ymax=150, colors='grey', linestyles='dashed', alpha=0.5)
    
    # Mark triple point
    plt.scatter([Ttrip-273.15], [Ptrip], color="black", s=100, label="Triple point", zorder=5)
    plt.annotate('Triple Point', (Ttrip-273.15, Ptrip), 
                xytext=(10, -15), textcoords='offset points', fontsize=14)
    
    # Overlay compression curves
    for stage, T_intermediate in enumerate(T_actual_intermediate_list):
        # Create pressure array for this stage
        P_start = P_list[stage]
        P_end = P_list[stage + 1]
        
        # The pressure array now includes both compression and cooling phases
        n_compression_points = len(T_intermediate) // 2  # Half for compression, half for cooling
        P_compression = np.linspace(P_start, P_end, n_compression_points)
        P_cooling = np.full(len(T_intermediate) - n_compression_points, P_end)
        
        # Convert temperatures to Celsius for plotting
        T_celsius = [T - 273.15 for T in T_intermediate]
        
        # Plot compression phase
        plt.plot(T_celsius[:n_compression_points], P_compression, 
                color=color, 
                linewidth=2, 
                zorder=4)
        
        # Plot cooling phase
        plt.plot(T_celsius[n_compression_points:], P_cooling, 
                color=color, 
                linewidth=2, 
                linestyle='-',
                zorder=4)
                
    # Plot a single line from (312.5 K, P_cooling_last) to (245 K, P_cooling_last) using the last stage's cooling pressure
    plt.plot([312.5-273.15, 245-5-273.15], [P_cooling, P_cooling], #5K buffer, and 245K is from Deng @15 bar delivery pressure
             color=color,
             linewidth=2,
             linestyle='-',
             label='39.4°C to -33.2°C')

    plt.scatter([373.15-273.15], [200], color=color, s=100, marker='*', label='100°C, 200 bar', zorder=6)
    
    # Simple curved line from (245K, P_cooling) to (373.15K, 200 bar)
    T_start = 245-5  # K -buffer
    P_start = P_cooling  # bar
    T_end = 373.15  # K
    P_end = 200  # bar
    
    # Create a simple curved line with a few points
    n_points = 16
    T_curve = np.linspace(T_start, T_end, n_points)
    P_curve = np.linspace(P_start, P_end, n_points)
    
    # Add a slight curve by modifying the middle points
    for i in range(1, n_points-1):
        # Add a small upward curve in the middle
        curve_factor = 0.04 * np.sin(np.pi * i / (n_points-1))
        P_curve[i] += curve_factor * (P_end - P_start)
    
    # Plot the compression line
    plt.plot(T_curve-273.15, P_curve, 
             color=color, 
             linewidth=2, 
             linestyle='--',
             label='Compression: -33.2°C to 100°C', zorder=4)

    # Add labels and formatting (logarithmic y-axis)
    plt.xlabel('Temperature [°C]', fontsize=14)
    plt.ylabel('Pressure [bar]', fontsize=14)
    plt.title(f'{gas_type} Phase Diagram with Compression Process (Log Scale)', fontsize=16)
    plt.xlim(200-273.15, 410-273.15)  # Set x-axis limit to -73°C to 137°C
    plt.ylim(0.8, 230)  # Set y-axis limit to 100 bar
    plt.yscale('log')  # Set logarithmic y-axis
    plt.grid(True, alpha=0.3)
    
    # Add grey horizontal lines for y-axis ticks
    y_ticks = plt.yticks()[0]
    for tick in y_ticks:
        plt.axhline(y=tick, color='grey', linestyle='-', alpha=0.7, linewidth=0.5)
    
    # Add horizontal lines at standard logarithmic scale values
    log_values = [0.8, 0.9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200]
    for value in log_values:
        if 0.8 <= value <= 230:  # Only plot within the y-axis limits
            plt.axhline(y=value, color='black', linestyle='-', alpha=0.5, linewidth=0.3)
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=14)
    
    # Increase tick label sizes
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.tick_params(axis='both', which='minor', labelsize=12)
    
    plt.tight_layout()
    plt.savefig('co2_compression_log.png', dpi=450, bbox_inches='tight')
    # plt.show()

def main():
    thermo_props = get_thermo_properties()
    Wcomp_list, Qcool_list, P_list, T_list, T_actual_intermediate_list = compression_energy(
        16, 
        40+273.15, 
        1, 
        gas_type='CO2', 
        n_stages=3, 
        pressure_ratio=2.5, 
        Tdiff=0, 
        thermo_props=thermo_props, 
        etais=0.8, 
        printing=True)
    
    
    # # Create the plot
    # plot_compression_curves(P_list, T_actual_intermediate_list, 'CO2')
    
    # # Create CO2 phase diagram
    # plot_co2_phase_diagram()
    
    # Create CO2 phase diagram with compression curves
    plot_compression_on_phase_diagram(P_list, T_actual_intermediate_list, thermo_props, 'CO2')

    plt.show()
if __name__ == "__main__":
    main()
