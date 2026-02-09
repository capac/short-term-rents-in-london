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

#  just select two columns: price and property amenities
columns = ['price', 'amenities']
inside_raw_airbnb_df = inside_raw_airbnb_df[columns]

# removal of null values from the Inside Airbnb data
inside_raw_airbnb_df.price = (
    inside_raw_airbnb_df.price.str.replace(r'[$,]', '', regex=True)
    )
inside_raw_airbnb_df = inside_raw_airbnb_df.loc[
     inside_raw_airbnb_df.price.notna()
     ]

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
    return 'None'


amenity_df = inside_raw_airbnb_df[['amenities']]
for index, row in amenity_df.iterrows():
    for item in row:
        category = get_general_category(item)
        if category:
            inside_raw_airbnb_df.loc[index, category] = 1

inside_raw_airbnb_df.drop(['amenities'], axis=1, inplace=True)
inside_raw_airbnb_df.to_csv(inside_airbnb_modified_data_file, index=False)
