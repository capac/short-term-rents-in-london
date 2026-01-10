#!/usr/bin/env python
# coding: utf-8

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.cluster import DBSCAN


home_dir = Path.home()
inside_airbnb_data_dir = (
    home_dir / 'Programming/data/inside-airbnb/london/2024-12-11'
    )
inside_airbnb_data_file = (
    inside_airbnb_data_dir / 'selected_short_term_rentals_for_modeling.csv'
    )
inside_airbnb_df = pd.read_csv(inside_airbnb_data_file,
                               keep_default_na=False, thousands=',')

coordinates = inside_airbnb_df[['latitude', 'longitude']]

# Using DBSCAN for cluster similarity search
eps = 0.03
min_samples = 400
db = DBSCAN(
    eps=eps,
    min_samples=min_samples,
    metric='haversine').fit(coordinates)
labels = db.labels_
print(f'Using DBSCAN with haversine metric and '
      f'eps={eps} and min_samples={min_samples}:\n')

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)
print('Estimated number of clusters: {:d}'.format(n_clusters_))
print('Estimated number of noise points: {:d}'.format(n_noise_))
silhouette_score = metrics.silhouette_score(coordinates, labels)
print(f'Silhouette Coefficient: {silhouette_score:.3f}\n')


def calculate_cluster_center(cluster_label):
    coords = coordinates[labels == cluster_label]
    cluster_center = coords.mean(axis=0)
    print(f'Cluster center {cluster_label}; '
          f'latitude: {cluster_center.iloc[0]:.5f}, '
          f'longitude: {cluster_center.iloc[1]:.5f}')


if __name__ == "__main__":
    for index in list(np.unique(labels)):
        calculate_cluster_center(index)
