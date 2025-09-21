import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.interpolate import griddata


plants_df = pd.read_csv('data/plants.csv')
df = pd.read_csv('data/shipping_costs.csv')
trucks = pd.read_csv('data/truck_costs.csv')

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

# Extrapolate costs for 1st and 2nd shortest distances using linear regression
# Sort by distance to get shortest distances first
shipping_sorted = shipping_df.sort_values('distance').reset_index(drop=True)

# Use 3rd-6th shortest distances (indices 2-5) to predict 1st-2nd (indices 0-1)
train_distances = shipping_sorted['distance'].iloc[2:6].values.reshape(-1, 1)
target_distances = shipping_sorted['distance'].iloc[0:2].values.reshape(-1, 1)

for mt in [0.5, 1, 2, 3]:
    for scenario in ['optimist', 'pessimist']:
        column = f"{scenario}_{mt}Mt"
        # Train on 3rd-6th distances
        train_costs = shipping_sorted[column].iloc[2:6].values
        reg = LinearRegression()
        reg.fit(train_distances, train_costs)
        predicted_costs = reg.predict(target_distances)
        
        # Update the dataframe - hard code replaced!
        shipping_sorted.loc[0, column] = predicted_costs[0]
        shipping_sorted.loc[1, column] = predicted_costs[1]

shipping_df = shipping_sorted.sort_index()

# Debug: Print the extrapolated values
print("\nDebug - Extrapolated shipping costs:")
print(shipping_df[['distance', 'optimist_0.5Mt', 'pessimist_0.5Mt']].round(2))
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Plot optimistic scenarios
for mt in [0.5, 1, 2, 3]:
    column = f"optimist_{mt}Mt"
    ax1.plot(shipping_df['distance'], shipping_df[column], 'o-', label=f'{mt}Mt', linewidth=2, markersize=6)

ax1.set_xlabel('Distance (km)')
ax1.set_ylabel('Cost (SEK)')
ax1.set_title('Optimistic Shipping Costs')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot pessimistic scenarios
for mt in [0.5, 1, 2, 3]:
    column = f"pessimist_{mt}Mt"
    ax2.plot(shipping_df['distance'], shipping_df[column], 'o-', label=f'{mt}Mt', linewidth=2, markersize=6)

ax2.set_xlabel('Distance (km)')
ax2.set_ylabel('Cost (SEK)')
ax2.set_title('Pessimistic Shipping Costs')
ax2.legend()
ax2.grid(True, alpha=0.3)
plt.tight_layout()

def get_shipping_cost(distance, mt_capacity=0.5, scenario='optimist', df=shipping_df):
    """Get interpolated shipping cost for given distance and capacity"""
    column_name = f"{scenario}_{mt_capacity}Mt"
    
    # Sort by distance for proper interpolation
    sorted_df = df.sort_values('distance')
    distances = sorted_df['distance'].values
    costs = sorted_df[column_name].values
    
    result = np.interp(distance, distances, costs)
    return result

def get_truck_cost(mass_co2, distance, trucks_df=trucks):
    """Get interpolated truck cost for given mass and distance using 2D interpolation"""
    
    # Extract data points
    masses = trucks_df['mass'].values
    distances = trucks_df['km'].values
    costs = trucks_df['SEK/ton'].values
    
    # Create grid points for interpolation
    points = np.column_stack((masses, distances))
    
    # Check if point is within bounds, if not use extrapolation
    mass_min, mass_max = masses.min(), masses.max()
    dist_min, dist_max = distances.min(), distances.max()
    
    if mass_co2 < mass_min or mass_co2 > mass_max or distance < dist_min or distance > dist_max:
        # Use nearest neighbor for extrapolation
        cost = griddata(points, costs, (mass_co2, distance), method='nearest')
    else:
        # Use linear interpolation
        cost = griddata(points, costs, (mass_co2, distance), method='linear')
    
    return cost

# Estimate the transport costs for each plant
print("\nPlant Names:")
print(plants_df['Name'].tolist())

# Plant=Handeloverket
distance_oygarden = 1507
distance_kalundborg = 842

cost_oygarden = get_shipping_cost(distance_oygarden, mt_capacity=0.5)
cost_kalundborg = get_shipping_cost(distance_kalundborg, mt_capacity=0.5)

print(f"\nHandeloverket - Oygarden: {distance_oygarden} km -> {cost_oygarden:.2f}")
print(f"Handeloverket - Kalundborg: {distance_kalundborg} km -> {cost_kalundborg:.2f}")

transport_costs = {
    "Handeloverket": {"Oygarden": cost_oygarden,"Kalundborg": cost_kalundborg}
}

# Plant=Vasteras KVV
distance_oygarden = 1607
distance_kalundborg = 948

cost_oygarden = get_shipping_cost(distance_oygarden, mt_capacity=0.5)
cost_kalundborg = get_shipping_cost(distance_kalundborg, mt_capacity=0.5)

transport_costs["Vasteras KVV"] = {"Oygarden": cost_oygarden, "Kalundborg": cost_kalundborg}

# Plant=Sjolunda
pipeline_length = 1000 # [m]
annual_CO2 = plants_df.loc[plants_df['Name'] == 'Sjolunda', 'Total'].iloc[0] * 0.90 * 1000 # [tCO2/yr]
mCO2 = annual_CO2*1000/8000/3600 # [kgCO2/s] assuming 8000 FLH
nCO2 = mCO2/44 # [kmolCO2/s]
VCO2 = nCO2*22.4 # [m3CO2/s] assuming ideal gas at standard conditions
vCO2 = 16 # [m/s] [Tharun, 2025] check paper "Enhancing early ..." and Appendix
Mcomp = 1.2 # [-] margin
Cfcomp = 568 # [EUR/m2]
CAPEX_pipeline = (2*np.pi*((VCO2/vCO2)/np.pi)**0.5 * (pipeline_length*Mcomp)) * Cfcomp # [EUR2015] [Tharun, 2025]
CAPEX_pipeline = CAPEX_pipeline * 950 / 555 # [EUR] adjusted CEPCI
r, n = 0.10, 20
annualized_capex = CAPEX_pipeline * r * (1 + r)**n / ((1 + r)**n - 1) # [EUR/yr]
levelized_capex = annualized_capex / annual_CO2 *11 # [EUR/tCO2]=>[SEK/tCO2]
print(f"CAPEX pipeline: {CAPEX_pipeline:} EUR")
print(f"Annualized CAPEX: {annualized_capex:} EUR/year")
print(f"Levelized CAPEX: {levelized_capex:} EUR/tCO2")

distance_oygarden = 848
distance_kalundborg = 206

cost_oygarden = get_shipping_cost(distance_oygarden, mt_capacity=0.5)
cost_kalundborg = get_shipping_cost(distance_kalundborg, mt_capacity=0.5)
transport_costs["Sjolunda"] = {"Oygarden": cost_oygarden, "Kalundborg": cost_kalundborg, "Pipeline": levelized_capex}

# Plant = Renova
distance_port = 15 # [km]
annual_CO2 = plants_df.loc[plants_df['Name'] == 'Renova', 'Total'].iloc[0] * 0.90 * 1000 # [tCO2/yr]
cost_truck = get_truck_cost(annual_CO2, distance_port) # [SEK/tCO2]

distance_oygarden = 622
distance_kalundborg = 222

cost_oygarden = get_shipping_cost(distance_oygarden, mt_capacity=0.5)
cost_kalundborg = get_shipping_cost(distance_kalundborg, mt_capacity=0.5)
transport_costs["Renova"] = {"Oygarden": cost_oygarden, "Kalundborg": cost_kalundborg, "Truck": cost_truck, }


print(f"\nTransport costs summary:")
for plant, costs in transport_costs.items():
    formatted_costs = {k: f"{float(v):.2f}" for k, v in costs.items()}
    print(f"{plant}: {formatted_costs}")
