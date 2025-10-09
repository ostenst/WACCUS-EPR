import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the results from the controller
experiments = pd.read_csv('results/experiments_ccu.csv')
outcomes_df = pd.read_csv('results/outcomes_ccu.csv')
print("Number of scenarios originally:", len(experiments))

celc_filter = (experiments['celc'] > 50) & (experiments['celc'] <= 70) # Based on SEA long term scenarios
experiments = experiments[celc_filter]
outcomes_df = outcomes_df[celc_filter]
print("Number of scenarios remaining:", len(experiments))

# Create box plot for total_CCU by tax level
max_potential_CCU = outcomes_df["potential_CCU"].max()
print("Maximum potential_CCU:", max_potential_CCU, "ktCO2/yr")

fig, ax = plt.subplots(figsize=(12, 6))
tax_levels = experiments['tax'].unique()
tax_levels = sorted(tax_levels)  # Sort tax levels for better visualization

# Convert ktCO2/yr to ktCH3OH/yr:
outcomes_df["kt_CH3OH"] = outcomes_df["total_CCU"] * 32/44 # [ktCO2/yr] -> [ktCH3OH/yr] from molar mass

# Prepare data for box plot and calculate fraction with n_plants=10
box_data = []
box_labels = []
fraction_10_plants = []
for tax_level in tax_levels:
    mask = experiments['tax'] == tax_level
    ccu_values = outcomes_df.loc[mask, 'total_CCU'].values
    fccu_values = outcomes_df.loc[mask, 'total_FCCU'].values
    bccu_values = outcomes_df.loc[mask, 'total_BCCU'].values
    n_plants_values = outcomes_df.loc[mask, 'n_plants'].values
    fraction_10 = np.sum(n_plants_values == 10) / len(n_plants_values)
    
    # Calculate statistics for FCCS and BECCS (80% confidence intervals)
    fccu_mean = np.mean(fccu_values)
    fccu_p5 = np.percentile(fccu_values, 10)
    fccu_p95 = np.percentile(fccu_values, 90)
    
    bccu_mean = np.mean(bccu_values)
    bccu_p5 = np.percentile(bccu_values, 10)
    bccu_p95 = np.percentile(bccu_values, 90)

    ccu_mean = np.mean(ccu_values)
    ccu_p5 = np.percentile(ccu_values, 10)
    ccu_p95 = np.percentile(ccu_values, 90)
    
    box_data.append(ccu_values)
    box_labels.append(f'{int(tax_level)}')
    fraction_10_plants.append(fraction_10)
    print(f"Tax level: {tax_level}, %n_plants=10: {fraction_10}")
    print(f"  FCCU - Mean: {fccu_mean*32/44:.1f}, 90% range: {fccu_p5*32/44:.1f}-{fccu_p95*32/44:.1f} ktMETHANOL/yr")
    print(f"  BCCU - Mean: {bccu_mean*32/44:.1f}, 90% range: {bccu_p5*32/44:.1f}-{bccu_p95*32/44:.1f} ktMETHANOL/yr")
    print(f"  CCU - Mean: {ccu_mean*32/44:.1f}, 90% range: {ccu_p5*32/44:.1f}-{ccu_p95*32/44:.1f} ktMETHANOL/yr")

# Create box plot with colored boxes
bp = ax.boxplot(box_data, tick_labels=box_labels, patch_artist=True)

# Color boxes based on fraction of scenarios with n_plants=10
cmap = plt.cm.magma  
for patch, fraction in zip(bp['boxes'], fraction_10_plants):
    color = cmap(fraction)  # fraction ranges from 0 to 1
    patch.set_facecolor(color)
    patch.set_alpha(1.0)

# Create second y-axis for methanol production
ax2 = ax.twinx()
ax2.set_ylabel('Total Methanol Production (kt/yr)', fontsize=14)

# Synchronize the second y-axis ticks with the first y-axis
y1_ticks = ax.get_yticks()
ax2.set_yticks(y1_ticks)
ax2.set_yticklabels([f'{round(tick * 32/44, -1):.0f}' for tick in y1_ticks])
ax2.set_ylim(ax.get_ylim())
ax2.tick_params(axis='y', which='major', labelsize=12)

# # Add annotations for fraction values
# for i, (fraction, tax_level) in enumerate(zip(fraction_10_plants, tax_levels)):
#     ax.text(i + 1, np.median(box_data[i]), f'{fraction:.2f}', 
#             ha='center', va='top', fontweight='bold', fontsize=10,
#             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

# # Add horizontal line for maximum potential CCU
# ax.axhline(y=max_potential_CCU, color='black', linestyle='--', linewidth=2, 
#            label=f'Max Potential CCU: {max_potential_CCU:.1f} ktCO2/yr')
# ax.legend(loc='lower right')

# Create separate figure for colorbar
fig_cbar, ax_cbar = plt.subplots(figsize=(0.75, 6))
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
sm.set_array([])
cbar = plt.colorbar(sm, cax=ax_cbar)
cbar.set_label('Fraction of scenarios with n_plants=10', fontsize=14)
plt.savefig('results/fig2_ccu_colorbar.png', dpi=300, bbox_inches='tight')
plt.close(fig_cbar)  # Close the colorbar figure

ax.set_xlabel('Tax Level (EUR/tCO2)', fontsize=14)
ax.set_ylabel('Total CCU (ktCO2/yr)', fontsize=14)
ax.set_title('Total CCU by Tax Level\n(Colored by fraction of scenarios with n_plants=10)', fontsize=16)
ax.tick_params(axis='both', which='major', labelsize=12)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('results/fig2_ccu_capacity.png', dpi=300, bbox_inches='tight')

plt.show()