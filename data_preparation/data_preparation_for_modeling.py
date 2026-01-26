#!/usr/bin/env python
# coding: utf-8

from pathlib import Path
import pandas as pd
import numpy as np
import json

# working directories
home_dir = Path.home()
data_dir = home_dir / 'Programming/data/inside-airbnb/london/2024-12-11/'
inside_airbnb_raw_data_dir = data_dir / 'raw/'
inside_airbnb_modified_data_dir = data_dir / 'modified/'

work_dir = home_dir / 'Programming/Python/machine-learning-exercises/'
inside_airbnb_work_dir = work_dir / 'short-term-rents-in-london/'

selected_data_file = (
    inside_airbnb_modified_data_dir /
    'selected_short_term_rentals_with_distances_and_amenities.csv'
    )
output_data_file = (
    inside_airbnb_modified_data_dir /
    'selected_short_term_rentals_for_modeling.csv'
    )

json_file = (
    inside_airbnb_work_dir /
    'foursquare_categories/foursquare_categories.json'
    )

top_level_category_names_list = []
with open(json_file, 'r') as f:
    fsq_categories = json.load(f)
    categories = fsq_categories["response"]["categories"]


def get_top_level_category_from_label(target, categories):
    def search(current_categories, top_level_name=None):
        for cat in current_categories:
            name = cat['name']
            subcats = cat.get('categories', [])
            if name == target:
                return top_level_name or name
            result = search(subcats, top_level_name or name)
            if result:
                return result
        return None
    return search(categories)


inside_airbnb_df = pd.read_csv(selected_data_file)
amenity_cols = ['first_amenity', 'second_amenity', 'third_amenity']
inside_airbnb_df[amenity_cols] = (
    inside_airbnb_df['nearest_amenity']
    .str.split(', ', n=3, expand=True)
    .reindex(range(3), axis=1)
    )
inside_airbnb_df = inside_airbnb_df.drop('nearest_amenity', axis=1)

inside_airbnb_df[amenity_cols] = inside_airbnb_df[amenity_cols].map(
    lambda label: get_top_level_category_from_label(label, categories)
)
inside_airbnb_df = inside_airbnb_df.replace(np.nan, 'None')

if not output_data_file.exists():
    inside_airbnb_df.to_csv(output_data_file, index=False)
