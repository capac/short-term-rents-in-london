#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from pathlib import Path
import json
import re

home_dir = Path.home()
inside_airbnb_raw_data_dir = (
    home_dir / 'Programming/data/inside-airbnb/london/2024-12-11/raw/'
    )
inside_airbnb_raw_data_file = (
    inside_airbnb_raw_data_dir / 'listings.csv'
    )

inside_airbnb_modified_data_dir = (
    home_dir / 'Programming/data/inside-airbnb/london/2024-12-11/modified/'
    )
inside_airbnb_modified_data_file = (
    inside_airbnb_modified_data_dir /
    'selected_prices_with_property_amenities.csv'
    )

work_dir = home_dir / 'Programming/Python/machine-learning-exercises/'
inside_airbnb_work_dir = work_dir / 'short-term-rents-in-london/'

json_file = (
    inside_airbnb_work_dir /
    'data_preparation/property_amenities_categories.json'
    )

inside_raw_airbnb_df = pd.read_csv(
    inside_airbnb_raw_data_file, low_memory=False,
    keep_default_na=False, thousands=','
    )

# select just the two columns of price and property amenities
columns = ['price', 'amenities']
inside_raw_airbnb_df = inside_raw_airbnb_df[columns]

inside_raw_airbnb_df.price = (
    inside_raw_airbnb_df.price.str.replace(r'[$,]', '', regex=True)
    )
inside_raw_airbnb_df.price = (
    inside_raw_airbnb_df.price.str.replace('', '0')
    )
inside_raw_airbnb_df.price = inside_raw_airbnb_df.price.astype('float')

with open(json_file, 'r') as f:
    property_amenities = json.load(f)
    category_rules = property_amenities['categories']

compiled_rules = {
    category: [re.compile(pat, re.IGNORECASE) for pat in patterns['patterns']]
    for category, patterns in category_rules.items()
}


def get_general_category(item):
    for category, patterns in compiled_rules.items():
        for pattern in patterns:
            if pattern.search(item):
                return category
    return 'Miscellaneous'


# add zero for all category columns
categories = list(category_rules.keys())
for category in categories:
    inside_raw_airbnb_df[category] = 0

# add one where general category is present
amenity_df = inside_raw_airbnb_df[['amenities']]
for index, sr in amenity_df.iterrows():
    row = json.loads(sr.values[0])
    for item in row:
        category = get_general_category(item)
        if category:
            inside_raw_airbnb_df.loc[index, category] = 1

# drop 'amenities' column
inside_raw_airbnb_df.drop(['amenities'], axis=1, inplace=True)
# retain only row that have rent prices
inside_raw_airbnb_df = (
    inside_raw_airbnb_df.loc[inside_raw_airbnb_df.price != 0]
    )

# save to CSV file
inside_raw_airbnb_df.to_csv(inside_airbnb_modified_data_file, index=False)
