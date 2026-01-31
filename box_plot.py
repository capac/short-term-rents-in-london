#!/usr/bin/env python
# coding: utf-8

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl
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

# Matplotlib stylesheet
plt.style.use('boxplot-style.mplstyle')

inside_airbnb_data_file = (
    inside_airbnb_modified_data_dir /
    'selected_short_term_rentals_for_modeling.csv'
    )
inside_airbnb_df = pd.read_csv(inside_airbnb_data_file,
                               keep_default_na=False, thousands=',')

num_dtypes_df = inside_airbnb_df.select_dtypes('number')
col_names = ['accommodates', 'availability_365',
             'bathrooms', 'bedrooms', 'crime_rate',
             'days_from_last_review', 'price',
             'distance_to_nearest_tube_station',
             'latitude', 'longitude']

# colormap
color_list = ['#A6CEE3', '#1F78B4', '#B2DF8A', '#33A02C',
              '#FB9A99', '#E31A1C', '#FDBF6F', '#FF7F00',
              '#CAB2D6', '#6A3D9A', '#FFD700', '#B15928']
cmap = mpl.colors.ListedColormap(color_list)
colors = cmap.colors

# removing outliers 5.5 IQR below and above the first and third quartiles
Q1 = num_dtypes_df.quantile(0.25)
Q3 = num_dtypes_df.quantile(0.75)
IQR = Q3 - Q1
multiplier = 5.5
lower = Q1 - multiplier * IQR
upper = Q3 + multiplier * IQR

clean_num_dtypes_df = num_dtypes_df[
    ~((num_dtypes_df < lower) | (num_dtypes_df > upper)).any(axis=1)
    ]

# boxplot
fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(8, 16))
for col, color, ax in zip(col_names, colors, axes.flatten()):
    bplot = ax.boxplot(
        clean_num_dtypes_df[col], widths=0.25, patch_artist=True
        )
    ax.set_xticks([])
    mod_col = ' '.join(col.capitalize().split('_'))
    ax.set_xlabel(mod_col, fontsize=11)
    ax.set_ylabel('')
    for patch in bplot['boxes']:
        patch.set_facecolor(color)
        patch.set_edgecolor('0.2')
        patch.set_alpha(0.95)

# set source text
ax.text(x=0.08, y=0.01,
        s='''Source: "Data from Inside Airbnb, CrimeRate.co.uk, '''
        '''Foursquare and Transport for London (TfL)."''',
        transform=fig.transFigure,
        ha='left', fontsize=9, alpha=0.7)

fig.suptitle('Boxplots of numerical features in '
             'dataset', fontsize=20, fontweight='bold',)

fig.subplots_adjust(top=0.92)
fig.savefig(plot_dir / 'box_plot.png')
