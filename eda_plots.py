from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# working directories
home_dir = Path.home()
inside_airbnb_data_dir = home_dir / 'Programming/data/inside-airbnb/london'
inside_airbnb_work_dir = (home_dir / 'Programming/Python/'
                          'machine-learning-exercises/'
                          'short-term-rents-in-london')
inside_airbnb_data_file = (inside_airbnb_data_dir /
                           'selected_short_term_rentals_with_distances.csv')
inside_airbnb_df = pd.read_csv(inside_airbnb_data_file,
                               keep_default_na=False, thousands=',')

# output direcotry for plots
path_plots = inside_airbnb_work_dir / 'plots'
path_plots.mkdir(exist_ok=True, parents=True)

# matplotlib style file
mplstyle_file = inside_airbnb_work_dir / 'barplot-style.mplstyle'
plt.style.use(mplstyle_file)

# drop unnecessary features
inside_airbnb_df.drop(['room_type', 'nearest_station', 'minimum_nights'],
                      axis=1, inplace=True)


# short term rentals by borough
borough_sr = inside_airbnb_df.borough.value_counts()
fig, ax = plt.subplots(nrows=1, ncols=1)
ax.bar(borough_sr.index, borough_sr.values)
ax.set_ylabel('Number of short term rentals')
ax.set_title('Short term rentals by borough')
plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
plot_filename = path_plots / 'short-term-rentals.png'
if not plot_filename.exists():
    plt.savefig(plot_filename, dpi=144, bbox_inches='tight')


# histogram plot of price distribution
prices_data = inside_airbnb_df.price.values
prices_data = prices_data.astype('float')
fig, ax = plt.subplots(nrows=1, ncols=1)
ax.hist(prices_data, bins=50, range=[100, 1000])
ax.set_title('Histogram of price distribution')
ax.set_xlabel('Price per night (in GBP)')
plot_filename = path_plots / 'hist_price_distrib.png'
if not plot_filename.exists():
    plt.savefig(plot_filename, dpi=144, bbox_inches='tight')


# histogram plot of logarithmic price distribution
prices_data = np.log1p(inside_airbnb_df.price.values)
prices_data = prices_data.astype('float')
fig, ax = plt.subplots(nrows=1, ncols=1)
ax.hist(prices_data, bins=50)
ax.set_title('Histogram of logarithmic price distribution')
ax.set_xlabel('Price per night (in logarithm of GBP)')
plot_filename = path_plots / 'hist_log_price_distrib.png'
if not plot_filename.exists():
    plt.savefig(plot_filename, dpi=144, bbox_inches='tight')


# number of units by property types plot
property_type_sr = inside_airbnb_df.property_type.value_counts()
fig, ax = plt.subplots(nrows=1, ncols=1)
ax.bar(x=property_type_sr.index, height=property_type_sr.values)
ax.set_ylabel('Number of units')
ax.set_title('Number of units by property types')
plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
plot_filename = path_plots / 'num_units_property.png'
if not plot_filename.exists():
    plt.savefig(plot_filename, dpi=144, bbox_inches='tight')


# average price per night per borough plot
average_price_per_borough_sr = inside_airbnb_df.price.groupby(
    inside_airbnb_df.borough).mean()
average_price_per_borough_sr.sort_values(ascending=False, inplace=True)
fig, ax = plt.subplots(nrows=1, ncols=1)
ax.bar(average_price_per_borough_sr.index, average_price_per_borough_sr.values)
plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
ax.set_ylabel('Average price per night (in GBP)')
ax.set_title('Average price per night per borough')
plot_filename = path_plots / 'average_price_borough.png'
if not plot_filename.exists():
    plt.savefig(plot_filename, dpi=144, bbox_inches='tight')
