#!/usr/bin/env python
# coding: utf-8

from pathlib import Path
from matplotlib import colormaps
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.cluster import DBSCAN
import geopandas as gpd
from shapely.geometry import Point
import contextily as ctx


home_dir = Path.home()
inside_airbnb_data_dir = home_dir / 'Programming/data/inside-airbnb/london'
inside_airbnb_work_dir = (home_dir /
                          'Programming/Python/machine-learning-exercises/'
                          'short-term-rents-in-london')


inside_airbnb_data_file = (inside_airbnb_data_dir /
                           'selected_short_term_rentals_with_distances.csv')
inside_airbnb_df = pd.read_csv(inside_airbnb_data_file,
                               keep_default_na=False, thousands=',')

inside_airbnb_df.drop(['room_type', 'nearest_station'], axis=1, inplace=True)

mplstyle_file = inside_airbnb_work_dir / 'barplot-style.mplstyle'
plt.style.use(mplstyle_file)

# Using DBSCAN for cluster similarity search
coordinates = inside_airbnb_df[['latitude', 'longitude']]
db = DBSCAN(eps=0.03, min_samples=10, metric='haversine').fit(coordinates)
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print("Estimated number of clusters: {:d}".format(n_clusters_))
print("Estimated number of noise points: {:d}".format(n_noise_))
silhouette_score = metrics.silhouette_score(coordinates, labels)
print(f"Silhouette Coefficient: {silhouette_score:.3f}")


def calculate_cluster_center(cluster_label):
    coords = coordinates[labels == cluster_label]
    cluster_center = coords.mean(axis=0)
    print(f"Cluster center {cluster_label}; "
          f"latitude: {cluster_center.iloc[0]:.5f}, "
          f"longitude: {cluster_center.iloc[1]:.5f}")
    return cluster_center


for index in list(np.unique(labels)):
    _ = calculate_cluster_center(index)


def plot_clusters(coordinates, filename, eps=0.03, min_samples=10):
    db = DBSCAN(eps=eps, min_samples=min_samples,
                metric='haversine').fit(coordinates)
    cluster_labels = db.labels_
    unique_cluster_labels = np.unique(cluster_labels)
    print(f'Unique cluster labels: {unique_cluster_labels}')

    # Create GeoDataFrame of points with labels
    geometry = [Point(lon, lat) for lat, lon in coordinates.values]
    gdf_points = gpd.GeoDataFrame({'label': cluster_labels},
                                  geometry=geometry, crs="EPSG:4326")
    # Project all to Web Mercator
    gdf_points = gdf_points.to_crs(epsg=3857)

    cluster_center_dict = {}
    for c_label in unique_cluster_labels:
        cluster_coords = coordinates[cluster_labels == c_label]
        cluster_center = cluster_coords.mean(axis=0)
        gdf_center = gpd.GeoDataFrame(
            geometry=[Point(cluster_center.iloc[1],
                            cluster_center.iloc[0])], crs="EPSG:4326")
        # Project all to Web Mercator
        gdf_center = gdf_center.to_crs(epsg=3857)
        cluster_center_dict[c_label] = gdf_center

    # Load London boroughs
    boroughs_path = inside_airbnb_data_dir / 'neighbourhoods.geojson'
    gdf_boroughs = gpd.read_file(boroughs_path).to_crs(epsg=3857)

    # Plotting
    fig, ax = plt.subplots()
    gdf_boroughs.boundary.plot(ax=ax, linewidth=0.5, color='gray', alpha=0.6)

    colormap = colormaps['Set1'].colors
    cluster_colors = {label: colors.rgb2hex(colormap[i])
                      for i, label in enumerate(unique_cluster_labels)}

    # Plot property points with cluster colors
    for c_label in unique_cluster_labels:
        subset = gdf_points[gdf_points['label'] == c_label]
        color = cluster_colors[c_label]
        marker = 'o' if c_label != -1 else 'x'
        subset.plot(ax=ax, color=color, markersize=30, marker=marker,
                    alpha=0.8, label=f'Cluster {c_label}')
        # Plot cluster center
        gdf_center = cluster_center_dict[c_label]
        gdf_center.plot(ax=ax, color='red', marker='*', markersize=100,
                        label='Cluster center')

    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)
    ax.set_title("DBSCAN clustered properties with boroughs in London",
                 fontsize=15)
    ax.legend()
    ax.axis('off')
    plt.tight_layout()
    path_plots = inside_airbnb_work_dir / 'plots'
    path_plots.mkdir(exist_ok=True, parents=True)
    plot_filename = path_plots / filename
    if not plot_filename.exists():
        plt.savefig(plot_filename, dpi=144, bbox_inches='tight')


plot_clusters(coordinates, 'london_clusters.png', eps=0.03, min_samples=10)
