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

# working directories
home_dir = Path.home()
data_dir = home_dir / 'Programming/data/inside-airbnb/london/2024-12-11/'
inside_airbnb_raw_data_dir = data_dir / 'raw/'
inside_airbnb_modified_data_dir = data_dir / 'modified/'

work_dir = home_dir / 'Programming/Python/machine-learning-exercises/'
inside_airbnb_work_dir = work_dir / 'short-term-rents-in-london/'

plots_dir = inside_airbnb_work_dir / 'plots'
plots_dir.mkdir(parents=True, exist_ok=True)
sil_coeff_dir = inside_airbnb_work_dir / 'plots' / 'silhouette_coefficients'
sil_coeff_dir.mkdir(parents=True, exist_ok=True)

inside_airbnb_data_file = (
    inside_airbnb_modified_data_dir /
    'selected_short_term_rentals_for_modeling.csv'
    )
inside_airbnb_df = pd.read_csv(inside_airbnb_data_file,
                               keep_default_na=False, thousands=',')

mplstyle_file = inside_airbnb_work_dir / 'barplot-style.mplstyle'
plt.style.use(mplstyle_file)


def plot_clusters(coordinates, eps=0.03, min_samples=10):
    db = DBSCAN(eps=eps, min_samples=min_samples,
                metric='haversine').fit(coordinates)
    cluster_labels = db.labels_
    unique_cluster_labels = np.unique(cluster_labels)
    print(f'Unique cluster labels in cluster plot: '
          f'{unique_cluster_labels}')

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
                            cluster_center.iloc[0])], crs="EPSG:4326"
            )
        # Project all to Web Mercator
        gdf_center = gdf_center.to_crs(epsg=3857)
        cluster_center_dict[c_label] = gdf_center

    # Load London boroughs
    boroughs_path = inside_airbnb_raw_data_dir / 'neighbourhoods.geojson'
    gdf_boroughs = gpd.read_file(boroughs_path).to_crs(epsg=3857)

    # Plotting
    fig, ax = plt.subplots()

    # Plot boroughs
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
        gdf_center.plot(ax=ax, color='red', marker='*',
                        markersize=100, label='Cluster center')

    # Add basemap
    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)

    # Final touches
    ax.set_title(
        'DBSCAN clustered properties with boroughs in London',
        fontsize=15
        )
    ax.legend()
    ax.axis('off')

    plt.tight_layout()
    plot_filename = plots_dir / 'london_clusters.png'
    if not plot_filename.exists():
        plt.savefig(plot_filename, dpi=144, bbox_inches='tight')


# set parameters
eps = 0.03
min_samples = 400

# plotting cluster points in London, UK
coordinates = inside_airbnb_df[['latitude', 'longitude']]
plot_clusters(coordinates, eps=eps, min_samples=min_samples)


# Fine-tuning silhouette coefficient
sil_score_dict = {}
eps_range = np.linspace(0.005, 0.05, 10)
print('Calculating optimal value for eps...')
for eps_param in eps_range:
    db = DBSCAN(eps=eps_param, min_samples=min_samples,
                metric='haversine').fit(coordinates)
    labels = db.labels_
    if len(set(labels)) >= 2:
        sil_score = metrics.silhouette_score(coordinates, labels)
        sil_score_dict[eps_param] = sil_score
    else:
        print(f'Silhouette score cannot be computed: only '
              f'1 cluster found for eps={eps_param.round(2)}')

fig, ax = plt.subplots()
ax.plot(list(sil_score_dict.keys()), list(sil_score_dict.values()),
        label='min_samples={}'.format(min_samples))
ax.legend(loc=4, handlelength=0, handletextpad=0, prop={'size': 15})
ax.set_xlabel("eps")
ax.set_ylabel("Silhouette coefficient")
ax.set_title("Silhouette coefficient for different values of eps")

sil_coeff_filename = sil_coeff_dir / 'sil-coeff-eps.png'
if not sil_coeff_filename.exists():
    plt.savefig(sil_coeff_filename, dpi=144, bbox_inches='tight')


sil_score_dict = {}
min_sample_range = np.linspace(10, 1000, 21)
print('Calculating optimal value for min_samples...')
for min_sample in min_sample_range:
    db = DBSCAN(eps=eps, min_samples=int(min_sample),
                metric='haversine').fit(coordinates)
    labels = db.labels_
    if len(set(labels)) >= 2:
        sil_score = metrics.silhouette_score(coordinates, labels)
        sil_score_dict[min_sample] = sil_score
    else:
        print(f'Silhouette score cannot be computed: '
              f'only 1 cluster found for min_sample={min_sample.round(2)}')

fig, ax = plt.subplots()
ax.plot(list(sil_score_dict.keys()), list(sil_score_dict.values()),
        label='eps={}'.format(eps))
ax.legend(loc=1, handlelength=0, handletextpad=0, prop={'size': 15})
ax.set_xlabel("min_sample")
ax.set_ylabel("Silhouette coefficient")
ax.set_title("Silhouette coefficient for different values of min_sample")

sil_coeff_filename = sil_coeff_dir / 'sil-coeff-min-sample.png'
if not sil_coeff_filename.exists():
    plt.savefig(sil_coeff_filename, dpi=144, bbox_inches='tight')
