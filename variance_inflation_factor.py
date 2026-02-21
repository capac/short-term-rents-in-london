#!/usr/bin/env python
# coding: utf-8

from pathlib import Path
import time
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Start of model analysis
start = time.perf_counter()

# working directories
home_dir = Path.home()
data_dir = home_dir / 'Programming/data/inside-airbnb/london/2024-12-11/'
inside_airbnb_modified_data_dir = data_dir / 'modified/'

work_dir = home_dir / 'Programming/Python/machine-learning-exercises/'
inside_airbnb_work_dir = work_dir / 'short-term-rents-in-london'

inside_airbnb_data_file = (
    inside_airbnb_modified_data_dir /
    'selected_short_term_rentals_for_modeling.csv'
    )

inside_airbnb_work_file = (
    inside_airbnb_work_dir / 'model_results' /
    'vif_num_output.txt'
    )

inside_airbnb_df = pd.read_csv(
    inside_airbnb_data_file,
    keep_default_na=False, thousands=','
    )

# Limit rent prices to a maximum of £1000 per night
max_limit = 1000
inside_airbnb_df = inside_airbnb_df[inside_airbnb_df.price < max_limit]

# Data pipeline
num_attribs = ['bathrooms', 'bedrooms', 'accommodates', 'availability_365',
               'crime_rate', 'distance_to_nearest_tube_station', 'price',
               'days_from_last_review', 'latitude', 'longitude']

num_pipeline = make_pipeline(
    SimpleImputer(strategy='median'),
    StandardScaler(),
)

preprocessing = ColumnTransformer([
    ('num', num_pipeline, num_attribs),
])

inside_airbnb_transformed = preprocessing.fit_transform(inside_airbnb_df)

prepared_df = pd.DataFrame(
    data=inside_airbnb_transformed,
    columns=preprocessing.get_feature_names_out(),
    index=inside_airbnb_df.index,
)

print(f'Writing variance inflation factor for '
      f'numerical features output to {inside_airbnb_work_file.name}')
with open(inside_airbnb_work_file, 'w') as f_output:
    for index, colname in enumerate(prepared_df.columns):
        modified_colname = colname.split('__')[1]
        f_output.writelines(
            f'{modified_colname:<36}'
            f'{round(variance_inflation_factor(prepared_df, index), 2)}\n'
        )

end = time.perf_counter()
print(f"Total time: {round((end - start), 2)} seconds")
