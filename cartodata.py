import pandas as pd
import requests

# Load the datasets
co2_per_capita = pd.read_csv("https://ourworldindata.org/grapher/co-emissions-per-capita.csv?v=1&csvType=full&useColumnShortNames=true", storage_options = {'User-Agent': 'Our World In Data data fetch/1.0'})
print(co2_per_capita.head())
# gdp_per_capita = pd.read_csv("gdp_per_capita.csv")  # Replace with your file path
# cumulative_co2 = pd.read_csv("cumulative_co2_emissions.csv")  # Replace with your file path

# # Standardize country names by stripping whitespace and normalizing format
# co2_per_capita['country'] = co2_per_capita['country'].str.strip()
# gdp_per_capita['country'] = gdp_per_capita['country'].str.strip()
# cumulative_co2['country'] = cumulative_co2['country'].str.strip()

# # Merge the datasets by 'country'
# df = pd.merge(co2_per_capita, gdp_per_capita, on="country", how="inner")
# df = pd.merge(df, cumulative_co2, on="country", how="inner")

# # Check the first few rows to verify
# print(df.head())