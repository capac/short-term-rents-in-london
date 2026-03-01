#!/usr/bin/env python
# coding: utf-8

from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import pandas as pd
import numpy as np
from scipy.stats import t, ttest_ind, sem

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
    # data with category feature
    filter_with_cat = inside_airbnb_df.loc[:, category] == 1
    price_with_cat = inside_airbnb_df['price'].loc[filter_with_cat]
    price_with_cat_mean = np.mean(price_with_cat)
    price_with_cat_stderr = sem(price_with_cat, ddof=1)
    # 97.5% quantile of a t-distribution with n−2 degrees of freedom correction
    correction_with_cat = t.ppf(0.975, len(price_with_cat)-2)
    # data without category feature
    filter_without_cat = inside_airbnb_df.loc[:, category] == 0
    price_without_cat = inside_airbnb_df['price'].loc[filter_without_cat]
    price_without_cat_mean = np.mean(price_without_cat)
    price_without_cat_stderr = sem(price_without_cat, ddof=1)
    # 97.5% quantile of a t-distribution with n−2 degrees of freedom correction
    correction_without_cat = t.ppf(0.975, len(price_without_cat)-2)
    # print(f'With category: mean ± std: '
    #       f'{np.round(price_with_cat_mean, 3)} ± '
    #       f'{np.round(price_with_cat_stderr, 3):<10}'
    #       f'Without category: mean ± std: '
    #       f'{np.round(price_without_cat_mean, 3)} ± '
    #       f'{np.round(price_without_cat_stderr, 3)}')
    # t-test for the means of two independent samples of scores
    tt = ttest_ind(price_with_cat, price_without_cat)
    tt_st = np.round(tt.statistic, 1)
    tt_pval = np.round(tt.pvalue, 3)
    ax.bar(
        [label_with, label_without],
        [price_with_cat_mean, price_without_cat_mean],
        yerr=[
            correction_with_cat*price_with_cat_stderr,
            correction_without_cat*price_without_cat_stderr
            ],
        width=0.7, alpha=0.7, error_kw={'elinewidth': 1.0},
        label=(f'Statistic: {tt_st}\n$p$-value: {tt_pval}'))
    ax.set_ylabel('Price (£)', fontsize=8)
    plt.setp(ax.get_yticklabels(), fontsize=8)
    ax.tick_params(axis='y', which='major', pad=5)
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.grid(axis='y', which='minor', linestyle=':')
    ax.legend(loc='upper right', fontsize=5,
              handlelength=0, handletextpad=0, frameon=False,
              bbox_to_anchor=(1.0, 1.2))
    ax.set_ylim([0, 235])
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
fig.savefig(
    plot_dir / 'property_features_vs_average_price.png',
    bbox_inches='tight', dpi=288
    )
