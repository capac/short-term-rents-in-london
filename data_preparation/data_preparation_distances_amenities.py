# /usr/bin/env python

import os
from pathlib import Path
import pandas as pd
import requests
import time
from geopy.distance import geodesic

FOURSQUARE_API_KEY = os.environ['FOURSQUARE_API_KEY']
FOURSQUARE_URL = 'https://api.foursquare.com/v3/places/search'

home_dir = Path.home()

data_dir = home_dir / 'Programming/data/inside-airbnb/london'
inside_airbnb_data_dir = data_dir / '2024-12-11'

work_dir = home_dir / 'Programming/Python/machine-learning-exercises/'
inside_airbnb_work_dir = work_dir / 'short-term-rents-in-london'

inside_airbnb_data_file = (
    inside_airbnb_data_dir / 'selected_short_term_rentals_with_distances.csv'
    )
cache_file = inside_airbnb_data_dir / 'amenities_cache.csv'
output_file = (
    inside_airbnb_data_dir /
    'selected_short_term_rentals_with_distances_and_amenities.csv'
    )

if os.path.exists(output_file):
    inside_airbnb_df = pd.read_csv(output_file)
else:
    inside_airbnb_df = pd.read_csv(inside_airbnb_data_file)

resolution = 3  # ~111m at equator per 0.001 deg
if os.path.exists(cache_file):
    cache_df = pd.read_csv(cache_file)
    cache = {
        (round(lat, resolution), round(lon, resolution)): amenity
        for lat, lon, amenity in zip(cache_df['latitude'],
                                     cache_df['longitude'],
                                     cache_df['nearest_amenity'])
        }
else:
    cache_df = pd.DataFrame(
        columns=['latitude', 'longitude', 'nearest_amenity'])
    cache = {}


def is_within_radius(lat1, lon1, lat2, lon2, radius_meters):
    return geodesic((lat1, lon1), (lat2, lon2)).meters <= radius_meters


def find_cached_category(lat, lon, cache_radius_meters=100):
    for (cached_lat, cached_lon), amenity in cache.items():
        if is_within_radius(lat, lon, cached_lat,
                            cached_lon, cache_radius_meters):
            return amenity
    return None


def get_nearby_categories(lat, lon, limit=1, radius=100):

    HEADERS = {
        'Authorization': FOURSQUARE_API_KEY,
        'Accept': 'application/json'
        }

    params = {
        'll': f'{lat},{lon}',
        'limit': limit,
        'radius': radius,
        'sort': 'DISTANCE'
    }

    response = requests.get(FOURSQUARE_URL, headers=HEADERS, params=params)
    response.raise_for_status()
    data = response.json()

    categories = set()
    for place in data.get('results', []):
        category_list = place.get('categories', [])
        if category_list:
            top_category = category_list[0].get('name')
            categories.add(top_category)

    result = ', '.join(categories) if categories else 'None'
    return result


def get_category_with_retry(lat, lon, retries=5):
    for attempt in range(retries):
        try:
            return get_nearby_categories(lat, lon, limit=3, radius=100)
        except requests.RequestException as e:
            print(f'Network error at ({lat}, {lon}): '
                  f'{e} - retry {attempt + 1}')
            time.sleep(2 ** attempt)
    print(f'Failed after {retries} attempts: ({lat}, {lon})')
    return 'None'


def process_dataframe(df):
    counter_cache = 0
    counter_saved_data = 0
    counter_using_category_from_cache = 0
    if 'nearest_amenity' not in df.columns:
        df['nearest_amenity'] = ''

    for idx, row in df.iterrows():
        if pd.notna(row['nearest_amenity']) and row['nearest_amenity'] != '':
            continue

        lat, lon = row['latitude'], row['longitude']
        cached = find_cached_category(lat, lon)
        if cached:
            counter_using_category_from_cache += 1
            category = cached
            if counter_using_category_from_cache % 10 == 0:
                print(f'Used {counter_using_category_from_cache} '
                      f'results from {cache_file.name}')
        else:
            category = get_category_with_retry(lat, lon)
            cache[(round(lat, resolution), round(lon, resolution))] = category
            cache_df.loc[len(cache_df)] = [lat, lon, category]
            cache_df.to_csv(cache_file, index=False)
            counter_cache += 1
            if counter_cache % 100 == 0:
                print(f'Saved {counter_cache} results to {cache_file.name}')

        df.at[idx, 'nearest_amenity'] = category
        df.to_csv(output_file, index=False)
        counter_saved_data += 1
        if counter_saved_data % 100 == 0:
            print(f'Saved {counter_saved_data} results to {output_file.name}')


if __name__ == '__main__':
    process_dataframe(inside_airbnb_df)
