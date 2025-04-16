from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# working directories
HOME_PATH = Path.home()
INSIDE_AIRBNB_DATA_PATH = HOME_PATH / 'Programming/data/inside-airbnb/london'
INSIDE_AIRBNB_WORK_PATH = (HOME_PATH / 'Programming/Python/'
                           'machine-learning-exercises/'
                           'short-term-rents-in-london')
inside_airbnb_data_file = (INSIDE_AIRBNB_DATA_PATH /
                           'selected_short_term_rentals_with_distances.csv')
inside_airbnb_df = pd.read_csv(inside_airbnb_data_file,
                               keep_default_na=False, thousands=',')

# output direcotry for plots
path_plots = INSIDE_AIRBNB_WORK_PATH / 'plots' / 'histograms'
path_plots.mkdir(exist_ok=True, parents=True)

# matplotlib style file
mplstyle_file = INSIDE_AIRBNB_WORK_PATH / 'barplot-style.mplstyle'
plt.style.use(mplstyle_file)


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


# minimum number of nights data distribution
min_nights_data = inside_airbnb_df.minimum_nights.values
min_nights_data = min_nights_data.astype('int')
fig, ax = plt.subplots(nrows=1, ncols=1)
ax.hist(min_nights_data, bins=50, range=[1, 400])
ax.set_title('Histogram of minimum number of nights data distribution')
ax.set_xlabel('Minimum number of nights')
plot_filename = path_plots / 'min_nights_data_distrib.png'
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


# histogram plot of attributes
inside_airbnb_num_df = inside_airbnb_df.select_dtypes(include=[np.number])
fig, axs = plt.subplots(nrows=4, ncols=2, figsize=(13, 10))
num_columns = inside_airbnb_num_df.columns
for ax, col in zip(axs.flatten(), num_columns):
    ax.hist(inside_airbnb_num_df[col], bins=50)
    ax.set_ylabel(col, fontsize=10)
    ax.set_title(f'Histogram of {col}', fontsize=12)
    ax.yaxis.set_tick_params(pad=3)
    plt.tight_layout(pad=1)
plot_filename = path_plots / 'attribute_histogram_plots.png'
if not plot_filename.exists():
    plt.savefig(plot_filename, dpi=144, bbox_inches='tight')
