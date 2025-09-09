import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
from matplotlib.patches import Patch

# Load the data
df = pd.read_csv("data/thesis_fig2.csv", encoding='latin-1')

# Filter for pulp and paper plants
pulp_plants = df[df['Industry/Plant type'] == 'Pulp and paper']

# Print CO2 emission summaries
print("CO2 Emissions Summary for Pulp and Paper Plants:")
print(f"Total CO2 emissions: {pulp_plants['Total'].sum():.1f} ktCO2/yr")
print(f"Fossil CO2 emissions: {pulp_plants['Fossil'].sum():.1f} ktCO2/yr")
print(f"Biogenic CO2 emissions: {pulp_plants['Biogenic'].sum():.1f} ktCO2/yr")
print(f"Number of plants: {len(pulp_plants)}")
print()

# Load Europe shapefile and convert to WGS84
europe = gpd.read_file("data/shapefiles/Europe/Europe_merged.shp").to_crs("EPSG:4326")

# Create the plot
fig, ax = plt.subplots(figsize=(10, 8))

# Plot the landmass
europe.plot(ax=ax, edgecolor="black", facecolor="whitesmoke")

# Separate integrated and standalone plants
integrated_plants = pulp_plants[pulp_plants['Integration'] == 'Integrated']
standalone_plants = pulp_plants[pulp_plants['Integration'] == 'Standalone']

# Plot standalone plants as solid circles
ax.scatter(standalone_plants['Longitude'], standalone_plants['Latitude'], 
          s=standalone_plants['Total'], alpha=0.7, 
          c=standalone_plants['Green'], cmap='magma', vmin=0, vmax=1,
          edgecolors='black', linewidth=0.5)

# Plot integrated plants as hatched circles
scatter = ax.scatter(integrated_plants['Longitude'], integrated_plants['Latitude'], 
                    s=integrated_plants['Total'], alpha=0.7, 
                    c=integrated_plants['Green'], cmap='magma', vmin=0, vmax=1,
                    edgecolors='black', linewidth=0.5, hatch='///')

# Set map bounds to focus on Sweden
ax.set_xlim(10, 25)
ax.set_ylim(55, 70)
ax.set_aspect(1.90)

# Remove ticks
ax.set_xticks([])
ax.set_yticks([])

# Annotate the 3 largest plants, with text exactly centered on the xy coordinate
top_3_plants = pulp_plants.nlargest(3, 'Total')
for _, plant in top_3_plants.iterrows():
    ax.annotate(
        f"{plant['Total']:.0f} ktCO2",
        (plant['Longitude'], plant['Latitude']),
        xytext=(0, 0), textcoords='offset points',
        fontsize=8, ha='center', va='center'
    )

# Add colorbar
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Green Fraction (1=100% biogenic, 0=100% fossil)')

# Add legend for integration types
legend_elements = [Patch(facecolor='moccasin', edgecolor='black', label='Standalone'),
                   Patch(facecolor='moccasin', edgecolor='black', hatch='///', label='Integrated')]
ax.legend(handles=legend_elements, loc='upper left')

ax.set_title('Pulp and Paper Plants in Sweden - CO2 Emissions (bubble size = Total ktCO2/yr)')

# Save the first figure
fig.savefig('pulp_plants_map.png', dpi=600, bbox_inches='tight')
# Create second map for all non-pulp and paper plants
print("\n" + "="*50)
print("CO2 Emissions Summary for Non-Pulp and Paper Plants:")
non_pulp_plants = df[df['Industry/Plant type'] != 'Pulp and paper']
print(f"Total CO2 emissions: {non_pulp_plants['Total'].sum():.1f} ktCO2/yr")
print(f"Fossil CO2 emissions: {non_pulp_plants['Fossil'].sum():.1f} ktCO2/yr")
print(f"Biogenic CO2 emissions: {non_pulp_plants['Biogenic'].sum():.1f} ktCO2/yr")
print(f"Number of plants: {len(non_pulp_plants)}")
print()

# Create the second figure
fig2, ax2 = plt.subplots(figsize=(10, 8))

# Plot the landmass
europe.plot(ax=ax2, edgecolor="black", facecolor="whitesmoke")

# Plot all non-pulp plants as solid circles
scatter2 = ax2.scatter(non_pulp_plants['Longitude'], non_pulp_plants['Latitude'], 
                      s=non_pulp_plants['Total'], alpha=0.7, 
                      c=non_pulp_plants['Green'], cmap='magma', vmin=0, vmax=1,
                      edgecolors='black', linewidth=0.5)

# Set map bounds to focus on Sweden
ax2.set_xlim(10, 25)
ax2.set_ylim(55, 70)
ax2.set_aspect(1.90)

# Remove ticks
ax2.set_xticks([])
ax2.set_yticks([])

# Add colorbar
cbar2 = plt.colorbar(scatter2, ax=ax2)
cbar2.set_label('Green Fraction (1=100% biogenic, 0=100% fossil)')

ax2.set_title('All Other Plants in Sweden - CO2 Emissions (bubble size = Total ktCO2/yr)')

# Save the second figure
fig2.savefig('other_plants_map.png', dpi=600, bbox_inches='tight')


# # Create third figure with both maps side by side
# print("\n" + "="*50)
# print("Creating combined figure...")

# fig3, (ax3_left, ax3_right) = plt.subplots(1, 2, figsize=(20, 8))

# # Left map: Non-pulp plants
# europe.plot(ax=ax3_left, edgecolor="black", facecolor="whitesmoke")
# scatter3_left = ax3_left.scatter(non_pulp_plants['Longitude'], non_pulp_plants['Latitude'], 
#                                 s=non_pulp_plants['Total'], alpha=0.7, 
#                                 c=non_pulp_plants['Green'], cmap='magma', vmin=0, vmax=1,
#                                 edgecolors='black', linewidth=0.5)
# ax3_left.set_xlim(10, 25)
# ax3_left.set_ylim(55, 70)
# ax3_left.set_aspect(1.90)
# ax3_left.set_xticks([])
# ax3_left.set_yticks([])
# ax3_left.set_title('All Other Plants in Sweden')

# # Right map: Pulp plants
# europe.plot(ax=ax3_right, edgecolor="black", facecolor="whitesmoke")
# ax3_right.scatter(standalone_plants['Longitude'], standalone_plants['Latitude'], 
#                  s=standalone_plants['Total'], alpha=0.7, 
#                  c=standalone_plants['Green'], cmap='magma', vmin=0, vmax=1,
#                  edgecolors='black', linewidth=0.5)
# ax3_right.scatter(integrated_plants['Longitude'], integrated_plants['Latitude'], 
#                  s=integrated_plants['Total'], alpha=0.7, 
#                  c=integrated_plants['Green'], cmap='magma', vmin=0, vmax=1,
#                  edgecolors='black', linewidth=0.5, hatch='///')
# ax3_right.set_xlim(10, 25)
# ax3_right.set_ylim(55, 70)
# ax3_right.set_aspect(1.90)
# ax3_right.set_xticks([])
# ax3_right.set_yticks([])
# ax3_right.set_title('Pulp and Paper Plants in Sweden')

# # Add shared colorbar
# cbar3 = fig3.colorbar(scatter3_left, ax=[ax3_left, ax3_right], shrink=0.8)
# cbar3.set_label('Green Fraction (1=100% biogenic, 0=100% fossil)')

# # Add legend for integration types on the right map
# legend_elements = [Patch(facecolor='moccasin', edgecolor='black', label='Standalone'),
#                    Patch(facecolor='moccasin', edgecolor='black', hatch='///', label='Integrated')]
# ax3_right.legend(handles=legend_elements, loc='upper left')

# plt.subplots_adjust(wspace=0.1)
plt.show()