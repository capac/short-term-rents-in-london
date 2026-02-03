#!/usr/bin/env python
# coding: utf-8

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import linregress
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

# colormap
color_list = ['#1F78B4', '#33A02C', '#FB9A99',
              '#E31A1C', '#FDBF6F', '#FF7F00',
              '#CAB2D6', '#6A3D9A', '#B15928']
cmap = mpl.colors.ListedColormap(color_list)
colors = cmap.colors

# scatter plot grid with regression fit and R2 values
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(11, 8), sharex=True)

for col_name, color, ax in zip(col_names, colors, axes.flatten()):
    x_data = clean_num_dtypes_df['price']
    y_data = clean_num_dtypes_df[col_name]
    slope, intercept, r2, p, stderr = linregress(x_data, y_data)
    ax.scatter(
        x_data, y_data, c=color,
        label=(rf'$R^2$: {r2:.3f}'+'\n'+rf'$p$: {p:.3f}'),
        alpha=0.4, marker='o',
        )
    ax.plot(x_data, slope*x_data + intercept, color='k', ls='-.', lw=1)
    ax.legend(loc='best', fontsize=10, frameon=False)
    ax.set_xlabel('Price', fontsize=9)
    mod_col_name = ' '.join(col_name.capitalize().split('_'))
    ax.set_ylabel(mod_col_name, fontsize=9)

# set source text
ax.text(x=0.08, y=-0.01,
        s='''Source: "Data from Inside Airbnb, CrimeRate.co.uk '''
        '''and Transport for London (TfL)."''',
        transform=fig.transFigure,
        ha='left', fontsize=8, alpha=0.7)

fig.suptitle('Regression of numerical features versus price',
             fontsize=14, fontweight='bold')
fig.subplots_adjust(top=0.92)
fig.savefig(plot_dir / 'reg_plot.png')
