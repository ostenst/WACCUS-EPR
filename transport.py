import pandas as pd
import numpy as np

plants_df = pd.read_csv('data/plants.csv')
df = pd.read_csv('data/transport_costs.csv')

# Add shipping costs for 0.5Mt
capex_reference = df['optimist_1Mt']*1 # absolute costs for 1Mt
capex_optimist = capex_reference * (0.5/1)**0.7
df['optimist_0.5Mt'] = capex_optimist / 0.5 # specific costs for 0.5Mt

capex_reference = df['pessimist_1Mt']*1 
capex_pessimist = capex_reference * (0.5/1)**0.7
df['pessimist_0.5Mt'] = capex_pessimist / 0.5 

column_order = ['distance', 'optimist_0.5Mt', 'pessimist_0.5Mt', 'optimist_1Mt', 'pessimist_1Mt', 
                'optimist_2Mt', 'pessimist_2Mt', 'optimist_3Mt', 'pessimist_3Mt']
shipping_df = df[column_order]

print("Transport costs with 0.5Mt:")
print(shipping_df)

def get_shipping_cost(distance, mt_capacity=0.5, scenario='optimist', shipping_df=shipping_df):
    """Get interpolated shipping cost for given distance and capacity"""
    column_name = f"{scenario}_{mt_capacity}Mt"
    distances = shipping_df['distance'].values[::-1]
    costs = shipping_df[column_name].values[::-1]
    return np.interp(distance, distances, costs)


# Estimate the transport costs for each plant
print("\nPlant Names:")
print(plants_df['Name'].tolist())

# Plant=Handeloverket
distance_oygarden = 1507
distance_kalundborg = 842

cost_oygarden = get_shipping_cost(distance_oygarden, mt_capacity=0.5, shipping_df=shipping_df)
cost_kalundborg = get_shipping_cost(distance_kalundborg, mt_capacity=0.5, shipping_df=shipping_df)

transport_costs = {
    "Handeloverket": {"Oygarden": cost_oygarden,"Kalundborg": cost_kalundborg}
}

# Plant=Vasteras KVV
distance_oygarden = 1607
distance_kalundborg = 948

cost_oygarden = get_shipping_cost(distance_oygarden, mt_capacity=0.5, shipping_df=shipping_df)
cost_kalundborg = get_shipping_cost(distance_kalundborg, mt_capacity=0.5, shipping_df=shipping_df)

transport_costs["Vasteras KVV"] = {"Oygarden": cost_oygarden, "Kalundborg": cost_kalundborg}

# Plant=Sjolunda
pipeline_length = 1000 # [m]
mCO2 = plants_df.loc[plants_df['Name'] == 'Sjolunda', 'Total'].iloc[0] * 0.90 * 1000 # [tCO2/yr]
# pipeline_cost = 0.02 + 260*(pipeline_length/1000)**0.07 * (mCO2/(1*10**6))**-0.61 # [EUR/(t*km)] [transport study] 1=km and t/yr
# pipeline_cost = pipeline_cost * 1 *11 # [SEK/t]
# pipeline_cost = 20 # [EUR/(t/h*m)] CAPEX, Gunnarsson (2024) MSc thesis
print("pipeline costs are weird")


print(f"\nTransport costs summary:")
for plant, costs in transport_costs.items():
    print(f"{plant}: {costs}")