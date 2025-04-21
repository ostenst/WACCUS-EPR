# LINK: https://github.com/austromorph/python-cartogram
import geopandas as gpd
import pandas as pd
import cartogram
import matplotlib.pyplot as plt

# Load world country polygons
world = gpd.read_file("data/ne_110m_admin_0_countries_lakes.shp")
world = world.rename(columns={"NAME": "country"})

# Load cumulative CO₂ emissions data from local CSV
# Expected columns: 'Entity', 'Year', 'Cumulative emissions'
co2_cumulative = pd.read_csv("cumulative-co-emissions.csv")
co2_cumulative["Cumulative emissions"] = co2_cumulative["Cumulative CO₂ emissions"]

# Filter for year 2023
co2_2023 = co2_cumulative[co2_cumulative["Year"] == 2023].copy()

# print(sorted(world['country'].unique()))
# print(sorted(co2_2023['Entity'].unique()))
world_countries = set(world['country'])
co2_countries = set(co2_2023['Entity'])
missing_in_co2 = world_countries - co2_countries
missing_in_world = co2_countries - world_countries
# print("\n  In shapefile but not in CO₂ data:", sorted(missing_in_co2))
# print("\n  In CO₂ data but not in shapefile:", sorted(missing_in_world))
missing = set(co2_2023["Entity"]) - set(world["country"])
print(sorted(missing))

name_mapping = {
    "United States" : "United States of America",
    # Add more as needed
}

# Harmonize country name fields
co2_2023["Entity"] = co2_2023["Entity"].str.strip()
co2_2023["Entity"] = co2_2023["Entity"].replace(name_mapping)
world["country"] = world["country"].str.strip()

# Reproject to Mollweide for accurate area
world = world.to_crs("ESRI:54009")

# Merge GeoDataFrame with emissions data
df = world.merge(co2_2023[["Entity", "Cumulative emissions"]], left_on="country", right_on="Entity", how="left")

# Drop countries without emission data
df = df.dropna(subset=["Cumulative emissions"])
print(df.head())
print(df.columns)

# Create the cartogram, adjusting accuracy
c = cartogram.Cartogram(
    df,
    "Cumulative emissions",
    max_iterations=9,
    max_average_error=0.05
)

print(c.head())
print(c.columns)
c.explore()

# Plot original and cartogram maps
fig, ax = plt.subplots(1, 2, figsize=(16, 8))

df.plot(ax=ax[0], column="Cumulative emissions", cmap="OrRd", legend=True)
ax[0].set_title("Original World Map (by Cumulative CO₂ Emissions, 2023)")

c.plot(ax=ax[1], column="Cumulative emissions", cmap="OrRd", legend=True)
ax[1].set_title("Contiguous Cartogram (by Cumulative CO₂ Emissions, 2023)")

plt.tight_layout()
plt.show()
