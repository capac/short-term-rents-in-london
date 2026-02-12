#!/usr/bin/env python
# coding: utf-8

from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import pandas as pd
import numpy as np

# working directories
home_dir = Path.home()
inside_airbnb_modified_data_dir = (
    home_dir / 'Programming/data/inside-airbnb/london/2024-12-11/modified/'
    )
inside_airbnb_modified_data_file = (
    inside_airbnb_modified_data_dir /
    'selected_prices_with_property_amenities.csv'
    )

work_dir = home_dir / 'Programming/Python/machine-learning-exercises/'
inside_airbnb_work_dir = work_dir / 'short-term-rents-in-london/'

inside_airbnb_df = pd.read_csv(inside_airbnb_modified_data_file)
categories = list(
    set(inside_airbnb_df.columns[1:]) -
    set(['Furniture', 'Miscellaneous'])
    )

# set Matplotlib stylesheet
plt.style.use('barplot-style.mplstyle')

# output direcotry for plots
plot_dir = inside_airbnb_work_dir / 'plots'
plot_dir.mkdir(exist_ok=True, parents=True)

fig, axes = plt.subplots(nrows=3, ncols=6, figsize=(12, 8))
for category, ax in zip(categories, axes.flatten()):
    label_with = 'With '+category.lower()
    label_without = 'Without '+category.lower()
    filter_with_cat = inside_airbnb_df.loc[:, category] == 1
    filter_without_cat = inside_airbnb_df.loc[:, category] == 0
    price_with_cat = inside_airbnb_df['price'].loc[filter_with_cat]
    price_with_cat_mean = np.mean(price_with_cat)
    price_without_cat = inside_airbnb_df['price'].loc[filter_without_cat]
    price_without_cat_mean = np.mean(price_without_cat)
    ax.bar(
        [label_with, label_without],
        [price_with_cat_mean, price_without_cat_mean], width=0.6,
        alpha=0.7,
        )
    ax.set_ylabel('Price (£)', fontsize=8)
    plt.setp(ax.get_yticklabels(), fontsize=8)
    ax.tick_params(axis='y', which='major', pad=5)
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.grid(axis='y', which='minor', linestyle=':')
    ax.set_ylim([0, 205])
    plt.setp(
        ax.get_xticklabels(), ha='right',
        rotation_mode='anchor', rotation=45,
        fontsize=7
        )
plt.tight_layout()

# set source text
fig.text(x=0.08, y=-0.01,
         s='''Source: "Data from Inside Airbnb."''',
         transform=fig.transFigure,
         ha='left', fontsize=8, alpha=0.7)

fig.suptitle('Property amenity features vs average rent price',
             fontsize=11, fontweight='bold')
fig.subplots_adjust(top=0.92)
fig.savefig(plot_dir / 'property_features_vs_average_price.png')
