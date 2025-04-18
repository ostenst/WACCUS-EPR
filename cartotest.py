import geopandas as gpd
import pandas as pd
import cartogram
import matplotlib.pyplot as plt

# Load world country polygons
world = gpd.read_file("data/ne_110m_admin_0_countries_lakes.shp")
world = world.rename(columns={"NAME": "country"})  # Adjust if needed

# Load CO₂ emissions per capita data
co2_per_capita = pd.read_csv("https://ourworldindata.org/grapher/co-emissions-per-capita.csv?v=1&csvType=full&useColumnShortNames=true", storage_options={'User-Agent': 'Our World In Data data fetch/1.0'})

# Check the column names to ensure we use the right one for country and emissions
print(co2_per_capita.columns)

# Filter the data for 2023 emissions
co2_per_capita_2023 = co2_per_capita[co2_per_capita['Year'] == 2023]

# Harmonize country names in the CO2 data
co2_per_capita_2023['Entity'] = co2_per_capita_2023['Entity'].str.strip()  # Assuming the column is called 'Entity'

# Harmonize the country names in the world shapefile
world['country'] = world['country'].str.strip()

# Reproject the world map to a projected CRS (e.g., Mollweide)
world = world.to_crs("ESRI:54009")  # Mollweide projection

# Merge the datasets on the 'Entity' column from CO2 data and 'country' column from the shapefile
df = world.merge(co2_per_capita_2023[['Entity', 'emissions_total_per_capita']], left_on='country', right_on='Entity', how='left')

# Drop rows where there are no CO2 emissions data
df = df.dropna(subset=['emissions_total_per_capita'])

# Increase accuracy by setting max_iterations and max_average_error
c = cartogram.Cartogram(
    df,
    "emissions_total_per_capita")
# c = cartogram.Cartogram(
#     df,
#     "emissions_total_per_capita",  # Data to use for cartogram
#     max_iterations=99,             # Max iterations for the algorithm
#     max_average_error=0.05         # Desired accuracy (lower value = more accurate)
# )

# Plot the original map and the cartogram side by side
fig, ax = plt.subplots(1, 2, figsize=(16, 8))

df.plot(ax=ax[0], column='emissions_total_per_capita', cmap='OrRd', legend=True)
ax[0].set_title("Original World Map (by CO₂ per capita in 2023)")

c.plot(ax=ax[1], column='emissions_total_per_capita', cmap='OrRd', legend=True)
ax[1].set_title("Contiguous Cartogram (by CO₂ per capita in 2023)")

plt.tight_layout()
plt.show()
