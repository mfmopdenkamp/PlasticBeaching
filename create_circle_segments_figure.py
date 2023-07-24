import numpy as np
import plotter
import matplotlib.pyplot as plt
from shapely.geometry import Point
import geopandas as gpd
from matplotlib.patches import Circle, Wedge


def visualize_segments(lon, lat, alphas, distances):
    width = 0.15
    lonlatbox = (lon - width, lon + width, lat - width, lat + width)
    fig, ax = plotter.get_marc_subplots(extent=lonlatbox, show_coastlines=False)

    dir_wind = 30

    # Plot the isosceles triangle sectors and full circles
    for alpha, distance in zip(alphas, distances):
        if alpha < 2 * np.pi:
            start_angle = dir_wind - np.degrees(alpha / 2)
            end_angle = dir_wind + np.degrees(alpha / 2)
            wedge = Wedge((lon, lat), distance, start_angle, end_angle, fill=False, edgecolor='red')
            ax.add_patch(wedge)
        else:
            circle = Circle((lon, lat), distance, fill=False, edgecolor='green')
            ax.add_patch(circle)

    # Plot the drifter location
    drifter = gpd.GeoDataFrame(geometry=[Point(lon, lat)], crs="EPSG:4326")
    drifter.plot(ax=ax, color='yellow', markersize=20, label='Drifter')

    # Draw an arrow indicating the wind direction
    wind_length = 0.05  # adjust this value as needed
    dx = wind_length * np.cos(np.radians(dir_wind))
    dy = wind_length * np.sin(np.radians(dir_wind))
    ax.arrow(lon, lat, dx, dy, color='black', width=0.001, head_width=0.005, length_includes_head=True,
             label='Wind direction')

    plt.legend()

    plt.savefig('figures/circle_segments.png', dpi=300)

    plt.show()


# Choose a location near the Galapagos Islands
lon, lat = -91.15, -0.65

base_alpha = 45
base_distance = 10000
alphas = 360 / np.array([16, 8, 6, 4, 2, 4 / 3])
distances = base_distance * np.sqrt(base_alpha / alphas)

# Add full circles
alphas = np.append(alphas, np.array([360, 360, 360, 360]))
alphas *= np.pi / 180  # Convert to radians
distances = np.append(distances, np.ones(4) * base_distance * [1, 3/4, 1/2, 1/4])

visualize_segments(lon, lat, alphas, distances / 111320)
