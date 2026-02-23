#!/usr/bin/env python
# coding: utf-8

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd

# working directories
home_dir = Path.home()
data_dir = home_dir / 'Programming/data/inside-airbnb/london/2024-12-11/'
inside_airbnb_raw_data_dir = data_dir / 'raw/'
inside_airbnb_modified_data_dir = data_dir / 'modified/'

work_dir = home_dir / 'Programming/Python/machine-learning-exercises/'
inside_airbnb_work_dir = work_dir / 'short-term-rents-in-london'

# output direcotry for plots
plot_dir = inside_airbnb_work_dir / 'plots' / 'maps'
plot_dir.mkdir(exist_ok=True, parents=True)

inside_airbnb_data_file = (
    inside_airbnb_modified_data_dir /
    'selected_short_term_rentals_for_modeling.csv'
    )
inside_airbnb_df = pd.read_csv(inside_airbnb_data_file,
                               keep_default_na=False, thousands=',')

# dataframe for median prices of airbnb listings
median_price_df = inside_airbnb_df['price'].groupby(
    inside_airbnb_df['borough']).median()

# dataframe for number of airbnb listings
number_listings_df = pd.DataFrame(inside_airbnb_df.groupby(
    inside_airbnb_df['borough']).size())
number_listings_df.rename(columns={0: 'number_listings'}, inplace=True)

neighbourhoods_file = (
    inside_airbnb_raw_data_dir /
    'neighbourhoods.geojson'
    )
map_df = gpd.read_file(neighbourhoods_file)
map_df.drop('neighbourhood_group', axis=1, inplace=True)
map_df.rename(columns={'neighbourhood': 'borough'}, inplace=True)

# median price and number of listings joined dataframe
final_df = map_df.set_index('borough').join(
    [median_price_df, number_listings_df])

# matplotlib style file
mplstyle_file = inside_airbnb_work_dir / 'barplot-style.mplstyle'
plt.style.use(mplstyle_file)


# Plotting the number of listings in each borough
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
final_df.plot(
    column='number_listings', cmap='coolwarm', ax=ax
    )
ax.set_title('Number of Airbnb rentals for each London borough', fontsize=14)
sm = plt.cm.ScalarMappable(
    cmap='coolwarm', norm=plt.Normalize(
        vmin=min(final_df.number_listings),
        vmax=max(final_df.number_listings)
        )
    )
ax.axis('off')
cbar = fig.colorbar(sm, ax=ax)
cbar.ax.yaxis.set_tick_params(pad=5)

# set source text
ax.text(x=0.08, y=-0.01,
        s='''Source: "Data from Inside Airbnb."''',
        transform=fig.transFigure,
        ha='left', fontsize=8, alpha=0.7)

plot_filename = plot_dir / 'number-rentals-per-borough.png'
plt.savefig(plot_filename, dpi=144, bbox_inches='tight')


# Plotting the median price of listings in each borough in GBP
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
final_df.plot(column='price', cmap='coolwarm', ax=ax)
ax.set_title('Median price of Airbnb rentals for each London borough',
             fontsize=14)
sm = plt.cm.ScalarMappable(
    cmap='coolwarm', norm=plt.Normalize(
        vmin=min(final_df.price),
        vmax=max(final_df.price)
        )
    )
ax.axis('off')
cbar = fig.colorbar(sm, ax=ax)
cbar.ax.yaxis.set_tick_params(pad=5)
cbar.ax.set_ylabel('GBP', fontsize=10)

# set source text
ax.text(x=0.08, y=-0.01,
        s='''Source: "Data from Inside Airbnb."''',
        transform=fig.transFigure,
        ha='left', fontsize=8, alpha=0.7)

plot_filename = plot_dir / 'median-price-per-borough.png'
plt.savefig(plot_filename, dpi=288, bbox_inches='tight')
