import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the results from the controller
experiments = pd.read_csv('results/experiments_ccs.csv')
outcomes_df = pd.read_csv('results/outcomes_ccs.csv')
print("Number of scenarios originally:", len(experiments))

# celc_filter = (experiments['celc'] > 50) & (experiments['celc'] <= 70) # Based on SEA long term scenarios
# experiments = experiments[celc_filter]
# outcomes_df = outcomes_df[celc_filter]
print("Number of scenarios remaining:", len(experiments))

# Create box plot for total_CCS by tax level
max_potential_CCS = outcomes_df["potential_CCS"].max()
print("Maximum potential_CCS:", max_potential_CCS, "ktCO2/yr")

fig, ax = plt.subplots(figsize=(12, 6))
tax_levels = experiments['tax'].unique()
tax_levels = sorted(tax_levels)  # Sort tax levels for better visualization

# Prepare data for box plot and calculate fraction with n_plants=10
box_data = []
box_labels = []
fraction_10_plants = []
for tax_level in tax_levels:
    mask = experiments['tax'] == tax_level
    ccs_values = outcomes_df.loc[mask, 'total_CCS'].values
    fccs_values = outcomes_df.loc[mask, 'total_FCCS'].values
    beccs_values = outcomes_df.loc[mask, 'total_BECCS'].values
    n_plants_values = outcomes_df.loc[mask, 'n_plants'].values
    fraction_10 = np.sum(n_plants_values == 10) / len(n_plants_values)
    
    # Calculate statistics for FCCS and BECCS (80% confidence intervals)
    fccs_mean = np.mean(fccs_values)
    fccs_p10 = np.percentile(fccs_values, 10)
    fccs_p90 = np.percentile(fccs_values, 90)
    
    beccs_mean = np.mean(beccs_values)
    beccs_p10 = np.percentile(beccs_values, 10)
    beccs_p90 = np.percentile(beccs_values, 90)
    
    box_data.append(ccs_values)
    box_labels.append(f'{int(tax_level)}')
    fraction_10_plants.append(fraction_10)
    print(f"Tax level: {tax_level}, %n_plants=10: {fraction_10}")
    print(f"  FCCS - Mean: {fccs_mean:.1f}, 80% range: {fccs_p10:.1f}-{fccs_p90:.1f} ktCO2/yr")
    print(f"  BECCS - Mean: {beccs_mean:.1f}, 80% range: {beccs_p10:.1f}-{beccs_p90:.1f} ktCO2/yr")

# Create box plot with colored boxes
bp = ax.boxplot(box_data, tick_labels=box_labels, patch_artist=True)

# Color boxes based on fraction of scenarios with n_plants=10
cmap = plt.cm.magma  
for patch, fraction in zip(bp['boxes'], fraction_10_plants):
    color = cmap(fraction)  # fraction ranges from 0 to 1
    patch.set_facecolor(color)
    patch.set_alpha(1.0)

# # Add annotations for fraction values
# for i, (fraction, tax_level) in enumerate(zip(fraction_10_plants, tax_levels)):
#     ax.text(i + 1, np.median(box_data[i]), f'{fraction:.2f}', 
#             ha='center', va='top', fontweight='bold', fontsize=10,
#             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

# Add horizontal line for maximum potential CCS
# ax.axhline(y=max_potential_CCS, color='black', linestyle='--', linewidth=2, 
#            label=f'Max Potential CCS: {max_potential_CCS:.1f} ktCO2/yr')
# ax.legend()

# Add colorbar
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax)
cbar.set_label('Fraction of scenarios with n_plants=10', fontsize=12)

ax.set_xlabel('Tax Level (EUR/tCO2)', fontsize=14)
ax.set_ylabel('Total CCS (ktCO2/yr)', fontsize=14)
ax.set_title('Total CCS by Tax Level\n(Colored by fraction of scenarios with n_plants=10)', fontsize=16)
ax.tick_params(axis='both', which='major', labelsize=12)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('results/fig1_ccs_capacity.png', dpi=300, bbox_inches='tight')



# Create violin plot for price increases at multiple tax levels
fig2, ax2 = plt.subplots(figsize=(12, 6))

# Define tax levels to plot
violin_tax_levels = [50, 80, 110, 140, 170, 200, 230]
price_metrics = ['granulates_inc', 'products_inc', 'bag_inc']

# Color violins using the same colormap and fraction as the first figure
cmap = plt.cm.magma

# Create violins for each tax level
for tax_level in violin_tax_levels:
    # Filter data for current tax level
    tax_mask = experiments['tax'] == tax_level
    violin_data = []
    for metric in price_metrics:
        values = outcomes_df.loc[tax_mask, metric].values
        violin_data.append(values)
    
    # Create violins positioned around the tax level
    violin_positions = [tax_level - 10, tax_level, tax_level + 10]
    parts = ax2.violinplot(violin_data, positions=violin_positions, 
                           showmeans=False, showmedians=True, widths=25)
    
    # Color violins using the same colormap and fraction as the corresponding boxplot
    tax_index = list(tax_levels).index(tax_level)
    fraction = fraction_10_plants[tax_index]
    
    for i, pc in enumerate(parts['bodies']):
        color = cmap(fraction)  # Use the same fraction as the corresponding tax level
        pc.set_facecolor(color)
        pc.set_alpha(1.0)
    
    # Color all whiskers and median lines black
    for key in ['cmeans', 'cmedians', 'cbars', 'cmins', 'cmaxes']:
        if key in parts:
            parts[key].set_color('black')
            parts[key].set_linewidth(1.5)

# Set x-axis to show all tax levels
ax2.set_xticks(violin_tax_levels)
ax2.set_xticklabels([str(level) for level in violin_tax_levels])

# Add colorbar
sm2 = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
sm2.set_array([])
cbar2 = plt.colorbar(sm2, ax=ax2)
cbar2.set_label('Fraction of scenarios with n_plants=10', fontsize=12)

ax2.set_xlabel('Tax Level (EUR/tCO2)', fontsize=14)
ax2.set_ylabel('Price Increase [-]', fontsize=14)
ax2.set_title('Price Increase Distribution at Tax Levels 50-300 EUR/tCO2', fontsize=16)
ax2.set_xlim(25, 325)
ax2.tick_params(axis='both', which='major', labelsize=12)
ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('results/fig1_ccs_increases.png', dpi=300, bbox_inches='tight')




# Create third figure for fund by tax level
fig3, ax3 = plt.subplots(figsize=(12, 4))

# Prepare data for fund box plot
fund_box_data = []
fund_box_labels = []
for tax_level in tax_levels:
    mask = experiments['tax'] == tax_level
    fund_values = outcomes_df.loc[mask, 'fund'].values
    
    fund_box_data.append(fund_values)
    fund_box_labels.append(f'{int(tax_level)}')

# Create box plot with colored boxes
bp3 = ax3.boxplot(fund_box_data, tick_labels=fund_box_labels, patch_artist=True)

# Color boxes based on fraction of scenarios with n_plants=10 (same as first figure)
for patch, fraction in zip(bp3['boxes'], fraction_10_plants):
    color = cmap(fraction)  # fraction ranges from 0 to 1
    patch.set_facecolor(color)
    patch.set_alpha(1.0)

# # Add annotations for fraction values
# for i, (fraction, tax_level) in enumerate(zip(fraction_10_plants, tax_levels)):
#     ax3.text(i + 1, np.median(fund_box_data[i]), f'{fraction:.2f}', 
#             ha='center', va='top', fontweight='bold', fontsize=10,
#             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

# Add colorbar
sm3 = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
sm3.set_array([])
cbar3 = plt.colorbar(sm3, ax=ax3)
cbar3.set_label('Fraction of scenarios with n_plants=10', fontsize=12)

ax3.set_xlabel('Tax Level (EUR/tCO2)', fontsize=14)
ax3.set_ylabel('Fund (MEUR/yr)', fontsize=14)
ax3.set_title('Fund by Tax Level\n(Colored by fraction of scenarios with n_plants=10)', fontsize=16)
ax3.tick_params(axis='both', which='major', labelsize=12)
ax3.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('results/fig1_ccs_fund.png', dpi=300, bbox_inches='tight')
plt.show()
