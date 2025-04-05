#!/usr/bin/env python
# coding: utf-8

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.cluster import DBSCAN

home_dir = Path.home()
inside_airbnb_data_dir = home_dir / 'Programming/data/inside-airbnb/london'
inside_airbnb_work_dir = (home_dir /
                          'Programming/Python/machine-learning-exercises/'
                          'short-term-rents-in-london')


inside_airbnb_data_file = (inside_airbnb_data_dir /
                           'selected_short_term_rentals_with_distances.csv')
inside_airbnb_df = pd.read_csv(inside_airbnb_data_file,
                               keep_default_na=False, thousands=',')

# matplotlib custom style sheet
mplstyle_file = inside_airbnb_work_dir / 'barplot-style.mplstyle'
plt.style.use(mplstyle_file)

path_plots = inside_airbnb_work_dir / 'plots/silhouette_coefficients'
path_plots.mkdir(exist_ok=True, parents=True)

# Using DBSCAN for cluster similarity search
X_geo = inside_airbnb_df[['latitude', 'longitude']]
db = DBSCAN(eps=0.03, min_samples=10, algorithm='auto').fit(X_geo)
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

# Fine-tuning silhouette coefficient
sil_score_list = []
sil_score_range = np.linspace(0.01, 0.1, 10)
for eps_param in sil_score_range:
    db = DBSCAN(eps=eps_param, min_samples=10, algorithm='auto').fit(X_geo)
    labels = db.labels_
    sil_score = metrics.silhouette_score(X_geo, labels)
    sil_score_list.append(sil_score)


# Silhouette coefficient for different values of eps
fig, ax = plt.subplots()
ax.plot(sil_score_range, sil_score_list)
ax.set_xlabel("eps")
ax.set_ylabel("Silhouette coefficient")
ax.set_title("Silhouette coefficient for different values of eps")
plot_filename = path_plots / 'sil-coeff-eps.png'
if not plot_filename.exists():
    plt.savefig(plot_filename, dpi=144, bbox_inches='tight')


sil_score_list = []
min_sample_range = np.linspace(2, 30, 29)
for min_sample in min_sample_range:
    db = DBSCAN(eps=0.07, min_samples=int(min_sample),
                algorithm='auto').fit(X_geo)
    labels = db.labels_
    sil_score = metrics.silhouette_score(X_geo, labels)
    sil_score_list.append(sil_score)


# Silhouette coefficient for different values of min_sample
fig, ax = plt.subplots()
ax.plot(min_sample_range, sil_score_list)
ax.set_xlabel("min_sample")
ax.set_ylabel("Silhouette coefficient")
ax.set_title("Silhouette coefficient for different values of min_sample")
plot_filename = path_plots / 'sil-coeff-min-sample.png'
if not plot_filename.exists():
    plt.savefig(plot_filename, dpi=144, bbox_inches='tight')
