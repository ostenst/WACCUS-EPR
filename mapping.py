import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.patheffects as path_effects
import matplotlib.cm as cm
import numpy as np


def plot_europe(mode="CCS", debug=False):
    """Plot a part of Europe using the shapefile data."""
    if debug:
        print(f"Mode: {mode}")

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

    europe = gpd.read_file("data/shapefiles/Europe/Europe_merged.shp").to_crs("EPSG:4326")

    plants_df = pd.read_csv("data/plants.csv")
    plants_df["color"] = plants_df["Biogenic"] / (plants_df["Biogenic"] + plants_df["Fossil"])
    magma = cm.get_cmap('magma')

    granulates_diameter = 1258600 * 0.85 * 3.66  # [tCO2/yr]
    granulates_coordinates = (58, 11.8)  # (lat, lon)
    products_diameter = 884300 * 0.85 * 3.66  # [tCO2/yr]
    products_coordinates = (56, 5)

    total_sum = plants_df['Total'].sum()
    if debug:
        print(f"Granulates diameter: {granulates_diameter}")
        print(f"Products diameter: {products_diameter}")
        print(f"Sum of Totals: {total_sum} ktCO2/yr")

    plt.figure(figsize=(12, 5))
    plt.bar(plants_df['Name'], plants_df['Total'], color='green', alpha=0.7)
    plt.ylabel('Total (ktCO2/yr)')
    plt.xlabel('Plant Name')
    plt.title('Total CO2 Emissions per Plant')
    plt.xticks(rotation=90)
    plt.tight_layout()

    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    europe.plot(ax=ax, edgecolor="black", facecolor="whitesmoke")

    plant_color = {"CCS": magma(0.60), "CCU": magma(0.20), "gasification": magma(0.9)}[mode]

    if mode == "CCS":
        for name, lon, lat in destinations:
            ax.scatter(lon, lat, marker='D', s=140, color=magma(0.60),
                       edgecolor='black', linewidth=1, zorder=5)

    plants_sorted = plants_df.sort_values('Total', ascending=False)
    top_10_plants = plants_sorted.head(10)

    if debug:
        print("\nTop 10 largest plants:")
        for i, (_, plant) in enumerate(top_10_plants.iterrows(), 1):
            print(f"{i:2d}. {plant['Name']}: {plant['Total']:.1f} ktCO2/yr")

    scaling = 1.2
    x_min, x_max = 2, 24
    y_min, y_max = 53.5, 70

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect(1.90)

    # Compute pie_size (axes fraction) to match the CCS scatter bubble diameter
    s_val = granulates_diameter / 1000 * scaling
    fig.canvas.draw()
    bbox = ax.get_window_extent(fig.canvas.get_renderer())
    marker_diameter_px = 2 * np.sqrt(s_val / np.pi) * fig.dpi / 72
    pie_size = marker_diameter_px / min(bbox.width, bbox.height)
    if debug:
        print(f"Computed pie_size: {pie_size:.3f}")

    if mode == "CCS":
        ax.scatter(granulates_coordinates[1], granulates_coordinates[0],
                   s=s_val, color='grey', alpha=0.8,
                   edgecolor='black', linewidth=1, zorder=4)
    elif mode == "CCU":
        cx, cy = granulates_coordinates[1], granulates_coordinates[0]
        x_frac = (cx - x_min) / (x_max - x_min)
        y_frac = (cy - y_min) / (y_max - y_min)
        pie_ax = ax.inset_axes([x_frac - pie_size / 2, y_frac - pie_size / 2,
                                pie_size, pie_size])
        pie_ax.pie([1, 2], colors=[magma(0.20), 'grey'],
                   wedgeprops={'edgecolor': 'black', 'linewidth': 1, 'alpha': 0.8})
        pie_ax.patch.set_alpha(0)
    elif mode == "gasification":
        ax.scatter(15.088126, 58.528987, s=s_val,
                   color=magma(0.20), alpha=0.8, edgecolor='black', linewidth=1, zorder=4)
        cx, cy = granulates_coordinates[1], granulates_coordinates[0]
        x_frac = (cx - x_min) / (x_max - x_min)
        y_frac = (cy - y_min) / (y_max - y_min)
        pie_ax = ax.inset_axes([x_frac - pie_size / 2, y_frac - pie_size / 2,
                                pie_size, pie_size])
        pie_ax.pie([1, 2], colors=[magma(0.20), 'grey'],
                   wedgeprops={'edgecolor': 'black', 'linewidth': 1, 'alpha': 0.8})
        pie_ax.patch.set_alpha(0)

    lons = []
    lats = []
    sizes = []

    for _, plant in plants_df.iterrows():
        lons.append(plant['Longitude'])
        lats.append(plant['Latitude'])
        sizes.append(plant['Total'] * scaling)

    ax.scatter(lons, lats, s=sizes, color=plant_color, alpha=0.8,
               edgecolor='black', linewidth=1, zorder=5)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("")

    fig.savefig('map.png', dpi=450, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    plot_europe(mode="CCU")
