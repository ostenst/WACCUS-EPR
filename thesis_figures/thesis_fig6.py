import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Read the CSV file
df = pd.read_csv('Data-package/Data-package/EU_emissions.csv')

# Extract LULUCF emissions data
lulucf_data = df[df.iloc[:, 0] == 'LULUCF emissions'].iloc[0, 1:].values.astype(float)
lulucf_years = df.columns[1:].astype(int).values

# Create extended year range (2005-2050)
years_full = np.arange(2005, 2051)

# Interpolate LULUCF data to full year range
lulucf_interpolated = np.interp(years_full, lulucf_years, lulucf_data)

# Extract ETS and Effort sharing emissions data
ets_data = df[df.iloc[:, 0] == 'EU ETS emissions'].iloc[0, 1:].values.astype(float)
effort_data = df[df.iloc[:, 0] == 'Effort sharing emissions'].iloc[0, 1:].values.astype(float)

# Interpolate data to full year range
ets_interpolated = np.interp(years_full, lulucf_years, ets_data)
effort_interpolated = np.interp(years_full, lulucf_years, effort_data)

# Figure 1: LULUCF Emissions
plt.figure(figsize=(10, 6))
plt.stackplot(years_full, lulucf_interpolated, 
              labels=['LULUCF Emissions'], 
              colors=[cm.magma(0.2)], alpha=0.7)
plt.xlabel('Year', fontsize=15)
plt.ylabel('LULUCF Emissions (Mt CO₂)', fontsize=15)
plt.title('LULUCF Emissions vs Year (Stackplot)', fontsize=15)
plt.grid(True, alpha=0.3)
 
plt.xlim(2005, 2060)
plt.ylim(-600, 4600)
plt.savefig('lulucf_emissions.png', dpi=450, bbox_inches='tight')

# Figure 2: ETS Emissions
plt.figure(figsize=(10, 6))

# Create stackplot with emissions
plt.stackplot(years_full, ets_interpolated,
              labels=['EU ETS Emissions'], 
              colors=[cm.magma(0.6)], alpha=0.7)

plt.xlabel('Year', fontsize=15)
plt.ylabel('EU ETS Emissions (Mt CO₂)', fontsize=15)
plt.title('EU ETS Emissions vs Year (Stackplot)', fontsize=15)
plt.grid(True, alpha=0.3)
 
plt.xlim(2005, 2060)
plt.ylim(-600, 4600)
plt.savefig('ets_emissions.png', dpi=450, bbox_inches='tight')

# Figure 3: Effort Sharing Emissions
plt.figure(figsize=(10, 6))

# Create stackplot with emissions
plt.stackplot(years_full, effort_interpolated,
              labels=['Effort Sharing Emissions'], 
              colors=[cm.magma(0.8)], alpha=0.7)
plt.xlabel('Year', fontsize=15)
plt.ylabel('Effort Sharing Emissions (Mt CO₂)', fontsize=15)
plt.title('Effort Sharing Emissions vs Year (Stackplot)', fontsize=15)
plt.grid(True, alpha=0.3)
 
plt.xlim(2005, 2060)
plt.ylim(-600, 4600)
plt.savefig('effort_sharing_emissions.png', dpi=450, bbox_inches='tight')

# Calculate maximum of Effort Sharing Regulation (ESR) emissions
# Filter out NaN values from the original data
effort_data_clean = effort_data[~np.isnan(effort_data)]
effort_years_clean = lulucf_years[~np.isnan(effort_data)]

max_effort_emissions = np.max(effort_data_clean)
max_year_idx = np.argmax(effort_data_clean)
max_year = effort_years_clean[max_year_idx]

print(f'Maximum Effort Sharing Regulation (ESR) emissions: {max_effort_emissions:.2f} Mt CO₂')
print(f'Year of maximum emissions: {max_year}')

# Now update the ETS plot with the offset
plt.figure(2)  # Switch to the second figure (ETS plot)
plt.clf()  # Clear the current plot

# Create stackplot with emissions offset by maximum ESR value
# Add the offset to the ETS data
ets_offset = ets_interpolated + max_effort_emissions
plt.stackplot(years_full, [max_effort_emissions * np.ones_like(years_full), ets_interpolated],
              labels=['Max ESR Level', 'EU ETS Emissions'], 
              colors=[cm.magma(0.3), cm.magma(0.6)], alpha=0.7)

plt.xlabel('Year', fontsize=15)
plt.ylabel('EU ETS Emissions (Mt CO₂)', fontsize=15)
plt.title('EU ETS Emissions vs Year (Stackplot)', fontsize=15)
plt.grid(True, alpha=0.3)
plt.xlim(2005, 2060)
plt.ylim(-600, 4600)
plt.savefig('ets_emissions.png', dpi=450, bbox_inches='tight')

plt.show()
