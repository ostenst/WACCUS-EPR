import pandas as pd
import numpy as np

# Load the data
experiments_ccs = pd.read_csv("results/experiments_ccs.csv")
experiments_ccu = pd.read_csv("results/experiments_ccu.csv")
outcomes_df_ccs = pd.read_csv("results/outcomes_ccs.csv")
outcomes_df_ccu = pd.read_csv("results/outcomes_ccu.csv")

def calculate_90th_percentile_range(data, kpi_name=None, debug=False):
    """Calculate 90th percentile range (5th to 95th percentile) for given data"""
    if debug:
        print(f"Input data shape: {data.shape}")
        print(f"Input data type: {type(data)}")
    
    # Apply transformations before calculating percentiles
    if kpi_name in ['mass_taxed', 'total_Qmethanol', 'total_Ppenalty']:
        data = data / 10**3  # Convert to millions
    elif kpi_name in ['granulates_inc', 'products_inc']:
        data = data * 100  # Convert to percentage
    
    p5 = np.percentile(data, 5)
    p95 = np.percentile(data, 95)
    
    # Use 3 decimal places for percentage KPIs, 2 decimal places for millions, whole numbers for others
    if kpi_name in ['granulates_inc', 'products_inc']:
        if debug:
            print(f"5th percentile: {p5:.1f}")
            print(f"95th percentile: {p95:.1f}")
        return f"{p5:.1f} - {p95:.1f}"
    elif kpi_name in ['mass_taxed', 'total_Qmethanol', 'total_Ppenalty']:
        if debug:
            print(f"5th percentile: {p5:.0f}")
            print(f"95th percentile: {p95:.0f}")
        return f"{p5:.0f} - {p95:.0f}"
    else:
        if debug:
            print(f"5th percentile: {p5:.0f}")
            print(f"95th percentile: {p95:.0f}")
        return f"{p5:.0f} - {p95:.0f}"

# Get unique tax levels
tax_levels_ccs = sorted(experiments_ccs['tax'].unique())
tax_levels_ccu = sorted(experiments_ccu['tax'].unique())
all_tax_levels = sorted(list(set(tax_levels_ccs + tax_levels_ccu)))

print(f"Tax levels found: {all_tax_levels}")

# Define the KPIs to analyze
ccs_kpis = [
    'mass_taxed',
    'granulates_inc', 
    'products_inc',
    'fund',
    'total_FCCS',
    'total_BECCS',
    'total_CCS'
]

ccu_kpis = [
    'total_FCCU',
    'total_BCCU', 
    'total_CCU',
    'total_Qmethanol',
    'total_Ppenalty'
]

# Create the results table
results_data = []

# Store fractions for each tax level
fraction_10_CCS = {}
fraction_10_CCU = {}

# Add CCS KPIs
for kpi in ccs_kpis:
    if kpi in outcomes_df_ccs.columns:
        kpi_data = {'KPI': kpi}
        
        for tax_level in all_tax_levels:
            # Filter CCS data for this tax level
            if tax_level in tax_levels_ccs:
                tax_mask = experiments_ccs['tax'] == tax_level
                if tax_mask.any():
                    outcomes_df_temporary = outcomes_df_ccs.loc[tax_mask]
                    fraction_10 = np.sum(outcomes_df_temporary['n_plants'] == 10) / len(outcomes_df_temporary['n_plants'])
                    fraction_10_CCS[tax_level] = fraction_10
                    data = outcomes_df_ccs.loc[tax_mask, kpi]
                    range_90 = calculate_90th_percentile_range(data, kpi)
                    kpi_data[f'Tax {tax_level}'] = range_90
                else:
                    kpi_data[f'Tax {tax_level}'] = 'N/A'
            else:
                kpi_data[f'Tax {tax_level}'] = 'N/A'
        
        results_data.append(kpi_data)
    else:
        print(f"Warning: {kpi} not found in CCS data")

# Add CCU KPIs  
for kpi in ccu_kpis:
    if kpi in outcomes_df_ccu.columns:
        kpi_data = {'KPI': kpi}
        
        for tax_level in all_tax_levels:
            # Filter CCU data for this tax level
            if tax_level in tax_levels_ccu:
                tax_mask = experiments_ccu['tax'] == tax_level
                if tax_mask.any():
                    outcomes_df_temporary = outcomes_df_ccu.loc[tax_mask]
                    fraction_10 = np.sum(outcomes_df_temporary['n_plants'] == 10) / len(outcomes_df_temporary['n_plants'])
                    if tax_level not in fraction_10_CCU:
                        fraction_10_CCU[tax_level] = fraction_10
                    data = outcomes_df_ccu.loc[tax_mask, kpi]
                    range_90 = calculate_90th_percentile_range(data, kpi)
                    kpi_data[f'Tax {tax_level}'] = range_90
                else:
                    kpi_data[f'Tax {tax_level}'] = 'N/A'
            else:
                kpi_data[f'Tax {tax_level}'] = 'N/A'
        
        results_data.append(kpi_data)
    else:
        print(f"Warning: {kpi} not found in CCU data")

# Add fraction_10_CCS as a KPI row
fraction_kpi = {'KPI': 'fraction_10_CCS'}
for tax_level in all_tax_levels:
    if tax_level in fraction_10_CCS:
        fraction_kpi[f'Tax {tax_level}'] = f"{fraction_10_CCS[tax_level]:.3f}"
    else:
        fraction_kpi[f'Tax {tax_level}'] = 'N/A'
results_data.append(fraction_kpi)

# Add fraction_10_CCU as a KPI row
fraction_kpi = {'KPI': 'fraction_10_CCU'}
for tax_level in all_tax_levels:
    if tax_level in fraction_10_CCU:
        fraction_kpi[f'Tax {tax_level}'] = f"{fraction_10_CCU[tax_level]:.3f}"
    else:
        fraction_kpi[f'Tax {tax_level}'] = 'N/A'
results_data.append(fraction_kpi)

# Create DataFrame and display
results_df = pd.DataFrame(results_data)

print("\n" + "="*80)
print("KPI ANALYSIS - 90th PERCENTILE RANGES BY TAX LEVEL")
print("="*80)
print(results_df.to_string(index=False))
print("="*80)

# Save the table as CSV
results_df.to_csv("results/kpi_table.csv", index=False)
print(f"\nTable saved to results/kpi_table.csv")
