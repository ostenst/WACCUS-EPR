import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.patheffects as path_effects

def plot_europe():
    """Plot a part of Europe using the shapefile data."""
    # Define origins and destinations
    origins = [
        ("Lulea", 22.2, 65.6),
        ("Sundsvall", 17.3, 62.4),
        ("Stockholm/Norvik", 17.9, 58.9),
        ("Malmo", 12., 55.6),
        ("Goteborg", 11.8, 57.6),
    ]

    destinations = [
        ("Northern Lights", 4.2, 60.4),
        ("Kalundborg", 10.8, 55.6),
    ]
    
    # Read the Europe shapefile and convert to WGS84 coordinate system
    europe = gpd.read_file("shapefiles/Europe/Europe_merged.shp").to_crs("EPSG:4326")
    
    # Read plants data from plants.csv
    plants_df = pd.read_csv("data/plants.csv")

    # Print the sum of the Totals
    total_sum = plants_df['Total'].sum()
    print(f"Sum of Totals: {total_sum} ktCO2/yr")

    plt.figure(figsize=(12, 5))
    plt.bar(plants_df['Name'], plants_df['Total'], color='green', alpha=0.7)
    plt.ylabel('Total (ktCO2/yr)')
    plt.xlabel('Plant Name')
    plt.title('Total CO2 Emissions per Plant')
    plt.xticks(rotation=90)
    plt.tight_layout()

    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    
    # Plot the landmass
    europe.plot(ax=ax, edgecolor="black", facecolor="whitesmoke")
    
    # Plot origins as crimson diamonds
    for name, lon, lat in origins:
        ax.scatter(lon, lat, marker='D', s=75, color='crimson', edgecolor='black', linewidth=1, zorder=5)
    
    # Plot destinations as crimson diamonds
    for name, lon, lat in destinations:
        ax.scatter(lon, lat, marker='D', s=75, color='maroon', edgecolor='black', linewidth=1, zorder=5)
    
    # Hard-coded list of plants that should be colored blue
    # green_plants = [
    #     "Renova",
    #     "Hogdalenverket", 
    #     "Sjolunda",
    #     "Korstaverket",
    #     "Garstadverket",
    #     "Vasteras KVV",
    #     "Handeloverket",
    #     "Bolanderna",
    #     "Filbornaverket",
    #     "Bristaverket"
    # ]
    green_plants = [
        "Renova",
        "Hogdalenverket", 
        "Sjolunda",
        # "Korstaverket",
        "Garstadverket",
        "Vasteras KVV",
        "Handeloverket",
        # "Bolanderna",
        # "Filbornaverket",
        # "Bristaverket"
    ]
    # green_plants = []
    
    # Sort plants by Total value and identify the 10 largest
    plants_sorted = plants_df.sort_values('Total', ascending=False)
    top_10_plants = plants_sorted.head(10)
    
    # Print the names of the 10 largest plants
    print("\nTop 10 largest plants:")
    for i, (_, plant) in enumerate(top_10_plants.iterrows(), 1):
        print(f"{i:2d}. {plant['Name']}: {plant['Total']:.1f} ktCO2/yr")
    
    # Plot all plants with different colors based on hard-coded list
    for _, plant in plants_df.iterrows():
        lat, lon = plant['Latitude'], plant['Longitude']
        total_value = plant['Total']
        # Scale the bubble size - adjust the scaling factor as needed
        bubble_size = total_value * 1.2  # Much smaller scaling factor to fit bubbles on map
        
        # Color plants based on hard-coded list: blue if in list, grey otherwise
        if plant['Name'] in green_plants:
            color = 'mediumseagreen'
            alpha = 0.8
        else:
            color = 'grey'
            alpha = 0.4
            
        ax.scatter(lon, lat, s=bubble_size, color=color, alpha=alpha, edgecolor='black', linewidth=0.5, zorder=4)
    
    # Formatting the plot - set the view limits
    ax.set_xlim(2, 24)
    ax.set_ylim(53.5, 70)
    ax.set_aspect(1.90) 

    # Add title and labels
    ax.set_title("Europe Map View", fontsize=14, fontweight='bold')
    ax.set_xlabel("Longitude (°E)", fontsize=12)
    ax.set_ylabel("Latitude (°N)", fontsize=12)
    
    # Add legend
    ax.scatter([], [], marker='D', s=100, color='crimson', edgecolor='black', linewidth=1, label='Hubs')
    ax.scatter([], [], marker='D', s=100, color='maroon', edgecolor='black', linewidth=1, label='Storage')
    ax.scatter([], [], s=200, color='mediumseagreen', alpha=0.8, edgecolor='black', linewidth=0.5, label='Selected plants')
    ax.scatter([], [], s=200, color='grey', alpha=0.4, edgecolor='black', linewidth=0.5, label='Other plants')
    ax.legend(loc='upper left')
    
    # Save the figure at 600 DPI
    fig.savefig('map.png', dpi=600, bbox_inches='tight')
    
    # Show the plot
    plt.show()

if __name__ == "__main__":
    plot_europe()
