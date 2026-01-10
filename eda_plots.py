from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# working directories
home_dir = Path.home()
data_dir = home_dir / 'Programming/data/inside-airbnb/london'
inside_airbnb_data_dir = data_dir / '2024-12-11'

work_dir = home_dir / 'Programming/Python/machine-learning-exercises/'
inside_airbnb_work_dir = work_dir / 'short-term-rents-in-london'

inside_airbnb_data_file = (
    inside_airbnb_data_dir /
    'selected_short_term_rentals_for_modeling.csv')

inside_airbnb_df = pd.read_csv(inside_airbnb_data_file,
                               keep_default_na=False, thousands=',')

# output direcotry for plots
path_plots = inside_airbnb_work_dir / 'plots' / 'histograms'
path_plots.mkdir(exist_ok=True, parents=True)

# matplotlib style file
mplstyle_file = inside_airbnb_work_dir / 'barplot-style.mplstyle'
plt.style.use(mplstyle_file)


# short term rentals by borough
borough_sr = inside_airbnb_df.borough.value_counts()
fig, ax = plt.subplots(nrows=1, ncols=1)
ax.bar(borough_sr.index, borough_sr.to_numpy())
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
ax.hist(prices_data, bins=50, range=(100, 1000))
ax.set_title('Histogram of price distribution')
ax.set_xlabel('Price per night (in GBP)')
plot_filename = path_plots / 'hist_price_distrib.png'
if not plot_filename.exists():
    plt.savefig(plot_filename, dpi=144, bbox_inches='tight')


# histogram plot of logarithmic price distribution
prices_data = np.log1p(inside_airbnb_df.price.to_numpy())
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
ax.bar(x=property_type_sr.index, height=property_type_sr.to_numpy())
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
ax.bar(average_price_per_borough_sr.index,
       average_price_per_borough_sr.to_numpy())
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
    if col == 'price':
        ax.hist(inside_airbnb_num_df[col], bins=50, range=(100, 1000))
    else:
        ax.hist(inside_airbnb_num_df[col], bins=50)
    ax.set_ylabel(col, fontsize=10)
    ax.set_title(f'Histogram of {col}', fontsize=12)
    ax.yaxis.set_tick_params(pad=3)
    plt.tight_layout(pad=1)
plot_filename = path_plots / 'attribute_histogram_plots.png'
if not plot_filename.exists():
    plt.savefig(plot_filename, dpi=144, bbox_inches='tight')
