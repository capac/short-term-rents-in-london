from pathlib import Path
import pandas as pd
# import numpy as np
import json

home_dir = Path.home()
inside_airbnb_data_dir = (
    home_dir / 'Programming/data/inside-airbnb/london')
inside_airbnb_work_dir = (
    home_dir /
    'Programming/Python/machine-learning-exercises/short-term-rents-in-london')

json_file = (
    inside_airbnb_work_dir /
    'foursquare_categories/foursquare_categories.json')

selected_data_file = (
    inside_airbnb_data_dir /
    'selected_short_term_rentals_with_distances_and_amenities.csv')
inside_airbnb_df = pd.read_csv(selected_data_file)

top_level_category_names_list = []
with open(json_file, 'r') as f:
    json_categories = json.load(f)


def extract_nonempty_nested_categories(data):
    def build_nested_dict(categories):
        result = {}
        for cat in categories:
            name = cat["name"]
            sub_dict = build_nested_dict(cat.get("categories", []))
            if sub_dict:
                result[name] = sub_dict
            elif categories:
                result[name] = None
        return result

    return build_nested_dict(data["response"]["categories"])


nested_dict = extract_nonempty_nested_categories(json_categories)
categories = json.dumps(nested_dict, indent=2)


def get_top_level_category_from_label(label):
    pass


# inside_airbnb_df[['amenity_1', 'amenity_2', 'amenity_3']] = (
#     inside_airbnb_df['nearest_amenity']
#     .str.split(', ', n=3, expand=True)
#     .reindex(range(3), axis=1)
#     )
# inside_airbnb_df = inside_airbnb_df.replace(np.nan, 'None')
# inside_airbnb_df = inside_airbnb_df.drop('nearest_amenity', axis=1)

# # saving results to CSV file with separate amenity columns
# print('Saving CSV file with separate amenity columns.')
# selected_data_file = (
#     inside_airbnb_data_dir /
#     'selected_short_term_rentals_for_modeling.csv')
# if not selected_data_file.exists():
#     inside_airbnb_df.to_csv(selected_data_file, index=False)
