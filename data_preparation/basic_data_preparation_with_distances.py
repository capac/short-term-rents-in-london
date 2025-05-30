from pathlib import Path
import pandas as pd
import numpy as np
import requests
from geopy.distance import geodesic
from scipy.spatial import KDTree

TFL_API_URL = 'https://api.tfl.gov.uk/StopPoint/Mode/tube'

home_dir = Path.home()
inside_airbnb_data_dir = (
    home_dir / 'Programming/data/inside-airbnb/london')
crime_rate_dir = (
    home_dir / 'Programming/data/crime-rate/')

inside_airbnb_data_file = (
    inside_airbnb_data_dir / 'listings.csv')
crime_rate_data_file = (
    crime_rate_dir / 'crimerate-pro-data-table-rmp-region-towns-cities.csv')

crime_rate_df = pd.read_csv(
    crime_rate_data_file, usecols=['Borough', 'Crime Rate'])
crime_rate_df.rename(
    columns={'Borough': 'borough', 'Crime Rate': 'crime_rate'},
    inplace=True)
crime_rate_df = crime_rate_df[crime_rate_df.borough != 'DownloadCSVExcelTSV']

columns_list = [
    'neighbourhood_cleansed', 'latitude', 'longitude', 'accommodates',
    'bedrooms', 'bathrooms', 'property_type', 'room_type', 'availability_365',
    'calendar_last_scraped', 'last_review', 'price']
inside_airbnb_df = pd.read_csv(
    inside_airbnb_data_file, usecols=columns_list,
    parse_dates=['calendar_last_scraped', 'last_review'],
    date_format="%d/%m/%Y")
inside_airbnb_df.rename(
    columns={'neighbourhood_cleansed': 'borough'},
    inplace=True)
inside_airbnb_df.price = inside_airbnb_df.price.str.replace('$', '')

print('Removing not-a-number values')
inside_airbnb_df = inside_airbnb_df.loc[
    (inside_airbnb_df.bathrooms.notna() &
     inside_airbnb_df.bedrooms.notna() &
     inside_airbnb_df.price.notna() &
     inside_airbnb_df.last_review.notna())]

inside_airbnb_df['days_from_last_review'] = (
    inside_airbnb_df.calendar_last_scraped -
    inside_airbnb_df.last_review).dt.days
inside_airbnb_df.drop(['calendar_last_scraped', 'last_review'],
                      axis=1, inplace=True)

print('Retaining only properties reviewed within the last six months')
six_months_in_days = 182
inside_airbnb_df = inside_airbnb_df[
    (inside_airbnb_df['days_from_last_review'] <
     six_months_in_days)]

year_minus_three_months_in_days = 275
print(f'Retaining properties that have been occupied at '
      f'least {365 - year_minus_three_months_in_days} days in the past year')
inside_airbnb_df = inside_airbnb_df[
    (inside_airbnb_df.availability_365 < year_minus_three_months_in_days)]

limit_num_categories = 30
print(f'Retaining only properties types present '
      f'{limit_num_categories} times or more')
inside_airbnb_sr = (inside_airbnb_df
                    .groupby('property_type')['property_type']
                    .count()
                    .sort_values(ascending=False))
inside_airbnb_sr_30 = inside_airbnb_sr[
    inside_airbnb_sr.values > limit_num_categories
    ]
inside_airbnb_property_type_list = list(inside_airbnb_sr_30.index)

inside_airbnb_df = inside_airbnb_df.loc[
    (inside_airbnb_df['property_type']
     .isin(inside_airbnb_property_type_list))]

print('Rounding latitude and longitude coordinates to six digits')
inside_airbnb_df[['latitude', 'longitude']] = \
    inside_airbnb_df[['latitude', 'longitude']].apply(
        lambda col: round(col, 6)
        )

print('Added crime data per borough')
inside_airbnb_df = inside_airbnb_df.merge(
    crime_rate_df, on='borough', how='left'
    )

print('Adding distance to nearest Underground station using TfL API')
response = requests.get(TFL_API_URL)
if response.status_code == 200:
    data = response.json()
else:
    raise Exception(f"API error: {response.status_code}")

tube_stations = []
for stop_point in data["stopPoints"]:
    lat, lon = stop_point["lat"], stop_point["lon"]
    tube_stations.append((lat, lon))

tube_df = pd.DataFrame(
    tube_stations, columns=["Latitude", "Longitude"]
    )
tube_coords = np.array(tube_df[['Latitude', 'Longitude']])
tree = KDTree(tube_coords)


def find_nearest_station(lat, lon):
    _, index = tree.query([lat, lon])
    nearest_station = tube_df.iloc[index]
    distance = geodesic(
        (lat, lon),
        (nearest_station.Latitude, nearest_station.Longitude)
        ).kilometers
    return round(distance, 6)


inside_airbnb_df['distance_to_nearest_tube_station'] = \
    inside_airbnb_df.apply(
        lambda row: find_nearest_station(row['latitude'], row['longitude']),
        axis=1, result_type='expand'
        )

selected_data_file = (inside_airbnb_data_dir /
                      'selected_short_term_rentals_with_distances.csv')
if not selected_data_file.exists():
    inside_airbnb_df.to_csv(selected_data_file, index=False)
