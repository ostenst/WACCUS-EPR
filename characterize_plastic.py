import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read labeled plastic data
plastics = pd.read_csv('data/plastic_supply_labeled.csv')

# Replace DS values in "Satt pa marknaden 2023" with 2019 data
col_2023 = 'Satt pa marknaden 2023 (ton)'
col_2019 = 'Satt pa marknaden 2019 (ton)'
plastics.loc[plastics[col_2023] == 'DS', col_2023] = plastics.loc[plastics[col_2023] == 'DS', col_2019].values

# Omit the row with "Beskrivning KN" = Summa for analysis
plastics = plastics[plastics['Beskrivning KN'] != 'Summa']

# Print the unique categories (not nan)
categories = plastics['Category'].dropna().unique()
print("We have the following plastic categories:")
print(categories)

# Count how many entries equal "DS" in "Satt pa marknaden" columns
ds_count_2023 = (plastics['Satt pa marknaden 2023 (ton)'] == 'DS').sum()
ds_count_2019 = (plastics['Satt pa marknaden 2019 (ton)'] == 'DS').sum()
print(f"\nNumber of 'DS' entries in 'Satt pa marknaden 2023 (ton)': {ds_count_2023}")
print(f"Number of 'DS' entries in 'Satt pa marknaden 2019 (ton)': {ds_count_2019}")
print(f"Total data rows (excluding Summa): {len(plastics)}")

# For rows belonging to categories, print its (shortened) category, name, and "Satt pa marknaden 2023 (ton)" value
for index, row in plastics.iterrows():
    if row['Category'] in categories:
        print(f"{row['Category'][:10]}: {row['Beskrivning KN'][:10]} {row['Satt pa marknaden 2023 (ton)']}")

# Print sum of 2023 market placement
sum_2023 = pd.to_numeric(plastics['Satt pa marknaden 2023 (ton)'], errors='coerce').sum()
print(f"\nTotal 'Satt pa marknaden 2023 (ton)': {sum_2023:,.0f}")

# For each category, calculate its mass fraction relative to the total mass "Satt pa marknaden 2023 (ton)" and print sorted by mass fraction
category_mass_fraction_sorted = []
print("\nMass fraction of the categories:")
for category in categories:
    category_mass = plastics[plastics['Category'] == category]['Satt pa marknaden 2023 (ton)'].astype(float).sum()
    total_mass = plastics['Satt pa marknaden 2023 (ton)'].astype(float).sum()
    category_mass_fraction = category_mass / total_mass
    category_mass_fraction_sorted.append((category, category_mass_fraction))
category_mass_fraction_sorted.sort(key=lambda x: x[1], reverse=True)
for category, mass_fraction in category_mass_fraction_sorted:
    print(f"{category[:30]}: {mass_fraction:.2%}")

# Map the categories to their carbon content [%C] - mainly caring about mass flows > 5% of total mass flow
carbon_map = {}
for category in categories:
    if category == 'Polymerer av eten, i obearbetad form':
        carbon_map[category] = {'PE': 0.856}
    elif category == 'Polymerer av propen eller av andra olefiner i obearbetod form':
        carbon_map[category] = {'PP': 0.850}
    elif category == 'Polymerer ov styren,i obeorhetod form':
        carbon_map[category] = {'PS': 0.904}
    elif category == 'Polymerer av vinylklorid eller av andra halogenerade olefiner i obearbetad form':
        carbon_map[category] = {'PVC': 0.396}
    else:
        carbon_map[category] = {'Other': 0.752} # The global mean

# Print the carbon map
print("\nCarbon content [%C] for the categories:")
for category, data in carbon_map.items():
    name, carbon_content = list(data.items())[0]
    print(f"{name}: {carbon_content:.2%}")

# Group by category and sum mass flows
category_sums = plastics.groupby('Category')['Satt pa marknaden 2023 (ton)'].apply(lambda x: pd.to_numeric(x, errors='coerce').sum())

# Sort by mass flow (descending)
category_sums = category_sums.sort_values(ascending=False)

# Extract carbon values and short names from nested dict
carbon_values = [list(carbon_map[cat].values())[0] for cat in category_sums.index]
short_names = [list(carbon_map[cat].keys())[0] for cat in category_sums.index]

# Map carbon content to colors using magma colormap
colors = plt.cm.magma_r((np.array(carbon_values) - min(carbon_values)) / (max(carbon_values) - min(carbon_values)))

fig, ax = plt.subplots(figsize=(14, 6))
bars = ax.bar(range(len(category_sums)), category_sums.values, color=colors)
# ax.set_xlabel('Category', fontsize=12)
ax.set_ylabel('Supplied plastic [tpl/yr]', fontsize=12)
ax.set_title('Plastic flows per type (colored by carbon content)', fontsize=14)
ax.set_xticks(range(len(category_sums)))
ax.set_xticklabels(short_names, rotation=45, ha='right')

# Annotate each bar with carbon content
for i, (bar, carbon) in enumerate(zip(bars, carbon_values)):
    if carbon == 0.396:
        color = 'black'
    else:
        color = 'white'
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 0.5, 
            f'{carbon:.2f}', ha='center', va='center', fontsize=11, fontweight='normal', color=color)

sm = plt.cm.ScalarMappable(cmap='magma_r', norm=plt.Normalize(vmin=min(carbon_values), vmax=max(carbon_values)))
sm.set_array([])
plt.colorbar(sm, ax=ax, label='Carbon content [tC/tpl]')
plt.tight_layout()

# Mean carbon content is the weighted average of the carbon content of the categories
mean_carbon_content = np.sum(category_sums.values * carbon_values) / np.sum(category_sums.values)
mean_emission_factor = mean_carbon_content * 3.66 # [kgCO2/kgpl]
print(f"\nMean carbon content: {mean_carbon_content:.2%} [kgC/kgpl]")
print(f"Mean emission factor: {mean_emission_factor:.2f} [kgCO2/kgpl]")

plt.show()
