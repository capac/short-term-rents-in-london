from pathlib import Path
import pandas as pd
import numpy as np

home_dir = Path.home()
inside_airbnb_data_dir = home_dir / 'Programming/data/inside-airbnb/london'

selected_data_file = (inside_airbnb_data_dir /
                      'selected_short_term_rentals_with_distances.csv')
inside_airbnb_df = pd.read_csv(selected_data_file)

inside_airbnb_df['borough'] = \
    inside_airbnb_df['borough'].replace({r'\s': r'_'}, regex=True)
inside_airbnb_df[['amenity_1', 'amenity_2', 'amenity_3']] = \
    inside_airbnb_df['amenities'].str.split(', ', expand=True)
inside_airbnb_df = inside_airbnb_df.replace(np.nan, 'None')
inside_airbnb_df = inside_airbnb_df.drop('amenities', axis=1)

# saving results to CSV file with separate amenity
# columns and underscores in borough names
print('Saving CSV file with separate amenity columns '
      'and underscores in borough names.')
selected_data_file = (
    inside_airbnb_data_dir /
    'selected_short_term_rentals_for_modeling.csv')
if not selected_data_file.exists():
    inside_airbnb_df.to_csv(selected_data_file, index=False)
