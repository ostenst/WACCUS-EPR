import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.patheffects as path_effects
import matplotlib.cm as cm
from matplotlib.lines import Line2D

MAIN_PLANT_LEGEND_LABELS = {
    "CCS": "CHP with CCS [ktCO2 p.a.]",
    "CCU": "CHP with CCU [ktCO2eq p.a.]",
    "gasification": "Replaced CHP [ktCO2eq p.a.]",
}


def _normalize_plant_name(name) -> str:
    return str(name).strip().lower()


def _other_plant_co2_calibration(
    plants_path: str = "data/plants.csv",
    debug: bool = False,
) -> tuple[float, float]:
    """Median Qwaste/Qdh and Total/Qwaste [ktCO2/yr per MW] from characterized main plants."""
    main = pd.read_csv(plants_path)
    qwaste_per_qdh = (main["Qwaste"] / main["Qdh"]).median()
    total_per_qwaste = (main["Total"] / main["Qwaste"]).median()
    if debug:
        print(
            f"Other-plant CO2 calibration: Qwaste/Qdh={qwaste_per_qdh:.3f}, "
            f"Total/Qwaste={total_per_qwaste:.3f} ktCO2/yr per MW"
        )
    return float(qwaste_per_qdh), float(total_per_qwaste)


def _load_other_plants(main_names, debug: bool = False) -> pd.DataFrame:
    """Merge other-plant capacities with coordinates; drop sites already in main set."""
    other = pd.read_csv("data/plants_other.csv")
    coords = pd.read_csv("data/plants_other_coordinates.csv")
    coords["Latitude"] = pd.to_numeric(coords["Latitude"], errors="coerce")
    coords["Longitude"] = pd.to_numeric(coords["Longitude"], errors="coerce")

    other["_key"] = other["Plant Name"].map(_normalize_plant_name)
    coords["_key"] = coords["Name"].map(_normalize_plant_name)
    merged = other.merge(
        coords[["_key", "Latitude", "Longitude"]],
        on="_key",
        how="inner",
    )
    main_keys = {_normalize_plant_name(n) for n in main_names}
    merged = merged[~merged["_key"].isin(main_keys)].copy()

    qdh = pd.to_numeric(merged["Heat output (MWheat)"], errors="coerce")
    qwaste_ratio, total_per_qwaste = _other_plant_co2_calibration(debug=debug)
    qwaste_est = qdh * qwaste_ratio
    merged["total_ktco2"] = qwaste_est * total_per_qwaste

    if debug:
        missing_coords = other.loc[~other["_key"].isin(coords["_key"]), "Plant Name"].tolist()
        if missing_coords:
            print(f"Other plants without coordinates (skipped): {missing_coords}")
        print(f"Other plants to plot: {len(merged)}")
    return merged


def _bubble_legend_handle(
    color,
    edgecolor: str = "black",
    alpha: float = 0.8,
    markersize: float = 10,
):
    return Line2D(
        [0],
        [0],
        marker="o",
        linestyle="None",
        markerfacecolor=color,
        markeredgecolor=edgecolor,
        markeredgewidth=1,
        markersize=markersize,
        alpha=alpha,
    )


def _map_bubble_legend(
    mode: str,
    plant_color,
    ccu_color,
    has_other_plants: bool,
    debug: bool = False,
) -> tuple[list[Line2D], list[str]]:
    """Legend handles for map bubble types (lower-left placement in plot_europe)."""
    handles: list[Line2D] = []
    labels: list[str] = []

    if has_other_plants:
        handles.append(_bubble_legend_handle("0.55", edgecolor="0.25", alpha=0.75))
        labels.append("Other CHP plants [ktCO2 p.a.]")

    handles.append(_bubble_legend_handle("black", alpha=0.90))
    labels.append("Plastic supply [ktCO2eq p.a.]")

    handles.append(_bubble_legend_handle(plant_color, alpha=0.8))
    labels.append(MAIN_PLANT_LEGEND_LABELS[mode])

    if mode == "gasification":
        handles.append(_bubble_legend_handle(ccu_color, alpha=0.8))
        labels.append("Methanol via gasification [ktCO2eq p.a.]")

    if debug:
        print(f"_map_bubble_legend: mode={mode}, labels={labels}")
    return handles, labels


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

    ccu_color = magma(0.20)
    plant_color = {"CCS": magma(0.60), "CCU": ccu_color, "gasification": magma(0.9)}[mode]

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
    y_min, y_max = 52.5, 69.5

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect(1.90)

    s_val = granulates_diameter / 1000 * scaling
    ax.scatter(
        granulates_coordinates[1],
        granulates_coordinates[0],
        s=s_val,
        color="black",
        alpha=0.90,
        edgecolor="black",
        linewidth=1,
        zorder=4,
    )
    if mode == "gasification":
        ax.scatter(
            15.088126,
            58.528987,
            s=s_val,
            color=ccu_color,
            alpha=0.8,
            edgecolor="black",
            linewidth=1,
            zorder=4,
        )

    lons = []
    lats = []
    sizes = []

    for _, plant in plants_df.iterrows():
        lons.append(plant['Longitude'])
        lats.append(plant['Latitude'])
        sizes.append(plant['Total'] * scaling)

    ax.scatter(lons, lats, s=sizes, color=plant_color, alpha=0.8,
               edgecolor='black', linewidth=1, zorder=5)

    other_plants = _load_other_plants(plants_df["Name"], debug=debug)
    has_other_plants = not other_plants.empty
    if has_other_plants:
        other_sizes = other_plants["total_ktco2"] * scaling
        ax.scatter(
            other_plants["Longitude"],
            other_plants["Latitude"],
            s=other_sizes,
            color="0.55",
            alpha=0.75,
            edgecolor="0.25",
            linewidth=0.8,
            zorder=4,
        )

    legend_handles, legend_labels = _map_bubble_legend(
        mode,
        plant_color,
        ccu_color,
        has_other_plants=has_other_plants,
        debug=debug,
    )
    ax.legend(
        legend_handles,
        legend_labels,
        loc="lower left",
        fontsize=10,
        framealpha=0.92,
        edgecolor="0.35",
    )

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("")

    fig.savefig('map.png', dpi=450, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    plot_europe(mode="CCS")
