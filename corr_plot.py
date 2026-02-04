#!/usr/bin/env python
# coding: utf-8

from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# working directories
home_dir = Path.home()
data_dir = home_dir / 'Programming/data/inside-airbnb/london/2024-12-11/'
inside_airbnb_modified_data_dir = data_dir / 'modified/'

work_dir = home_dir / 'Programming/Python/machine-learning-exercises/'
inside_airbnb_work_dir = work_dir / 'short-term-rents-in-london'

# output direcotry for plots
plot_dir = inside_airbnb_work_dir / 'plots'
plot_dir.mkdir(exist_ok=True, parents=True)

inside_airbnb_data_file = (
    inside_airbnb_modified_data_dir /
    'selected_short_term_rentals_for_modeling.csv'
    )
inside_airbnb_df = pd.read_csv(inside_airbnb_data_file,
                               keep_default_na=False, thousands=',')

# Matplotlib stylesheet
plt.style.use('lineplot-style.mplstyle')

num_dtypes_df = inside_airbnb_df.select_dtypes('number')
col_names = ['accommodates', 'availability_365',
             'bathrooms', 'bedrooms', 'crime_rate',
             'days_from_last_review',
             'distance_to_nearest_tube_station',
             'latitude', 'longitude']

# removing outliers 5.5 IQR below and above the first and third quartiles
# for each numerical feature, very useful to remove extreme price outliers
Q1 = num_dtypes_df.quantile(0.25)
Q3 = num_dtypes_df.quantile(0.75)
IQR = Q3 - Q1
multiplier = 5.5
lower = Q1 - multiplier * IQR
upper = Q3 + multiplier * IQR

clean_num_dtypes_df = num_dtypes_df[
    ~((num_dtypes_df < lower) | (num_dtypes_df > upper)).any(axis=1)
    ]

# scatter plot grid with regression fit and R2 values
fig, axes = plt.subplots(figsize=(12, 7))
font_size = 10
corr_df = clean_num_dtypes_df.corr()
mask = np.zeros_like(corr_df)
mask[np.tril_indices_from(mask)] = True

hm = sns.heatmap(
    data=corr_df, cmap='viridis',
    ax=axes, annot=True, fmt='1.4f', mask=mask,
    annot_kws={'size': font_size}
    )
axes.tick_params(labelsize=font_size)

cbar = hm.collections[0].colorbar
cbar.ax.tick_params(labelsize=font_size)

for tick_label in (axes.get_xticklabels(), axes.get_yticklabels()):
    plt.setp(
        tick_label, ha='right', rotation_mode='anchor', rotation=45
        )

tick_labels_list = []
for tick_label in (axes.get_xticklabels(), axes.get_yticklabels()):
    tick_labels_list.append(t.get_text().replace('_', ' ') for t in tick_label)

axes.set_xticklabels(tick_labels_list[0])
axes.set_yticklabels(tick_labels_list[1])

plt.tight_layout()
plt.grid(True, linestyle='-.')

# set source text
axes.text(x=0.08, y=-0.01,
          s='''Source: "Data from Inside Airbnb, CrimeRate.co.uk '''
          '''and Transport for London (TfL)."''',
          transform=fig.transFigure,
          ha='left', fontsize=8, alpha=0.7)

fig.suptitle('Correlation of numerical features',
             fontsize=14, fontweight='bold')
fig.subplots_adjust(top=0.92)
fig.savefig(plot_dir / 'corr_plot.png')
