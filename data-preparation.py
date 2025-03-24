from pathlib import Path
import requests
import os
import re
import pandas as pd
import numpy as np
from geopy.distance import geodesic
from scipy.spatial import KDTree

home_dir = Path.home()
inside_airbnb_data_dir = home_dir / 'Programming/data/inside-airbnb/london'
crime_rate_dir = home_dir / 'Programming/data/crime-rate/'

FOURSQUARE_API_KEY = os.environ['FOURSQUARE_API_KEY']
FOURSQUARE_URL = 'https://api.foursquare.com/v3/places/search'
TFL_API_URL = 'https://api.tfl.gov.uk/StopPoint/Mode/tube'

inside_airbnb_data_file = inside_airbnb_data_dir / 'listings.csv'
crime_rate_data_file = (crime_rate_dir /
                        'crimerate-pro-data-table-rmp-region-towns-cities.csv')

crime_rate_df = pd.read_csv(crime_rate_data_file,
                            usecols=['Borough', 'Crime Rate'])
crime_rate_df.rename(columns={'Borough': 'borough',
                              'Crime Rate': 'crime_rate'}, inplace=True)
crime_rate_df = crime_rate_df[crime_rate_df.borough != 'DownloadCSVExcelTSV']

columns_list = ['neighbourhood_cleansed', 'bathrooms', 'bedrooms',
                'latitude', 'longitude', 'room_type', 'latitude',
                'longitude', 'property_type', 'price', 'minimum_nights']
inside_airbnb_df = pd.read_csv(inside_airbnb_data_file, usecols=columns_list)
inside_airbnb_df.rename(columns={'neighbourhood_cleansed': 'borough'},
                        inplace=True)
inside_airbnb_df.price = inside_airbnb_df.price.str.replace('$', '')

inside_airbnb_df = inside_airbnb_df.loc[
    inside_airbnb_df.room_type == 'Entire home/apt'
    ]
inside_airbnb_df = inside_airbnb_df.loc[
    inside_airbnb_df.minimum_nights >= 30
    ]
inside_airbnb_df = inside_airbnb_df.merge(
    crime_rate_df, on='borough', how='left'
    )
inside_airbnb_df = inside_airbnb_df.loc[
    (inside_airbnb_df.bathrooms.notna() &
     inside_airbnb_df.bedrooms.notna() &
     inside_airbnb_df.price.notna())
     ]

BROAD_CATEGORIES = [
    ("Grocery Store", ["supermarket", "grocery", "convenience store",
                       "gourmet", "butcher"]),
    ("Restaurant", ["restaurant", "bbq", "steakhouse", "diner", "sushi",
                    "cuisine", "brasserie", "joint", "buffet",
                    "pizzeria", "parlor", "fish", "chips", "bistro",
                    "dining", "buffet", "deli"]),
    ("Cafe", ["coffee", "cafe", "tea", "bakery", "dessert",
              "café", "drinking", "breakfast", "gelato shop",
              "bagel", "sandwich", "snack", "cupcake", "pastry"]),
    ("Nightlife", ["bar", "pub", "club", "lounge", "casino", "speakeasy",
                   "brewery", "roof deck"]),
    ("Retail", ["shopping", "store", "mall", "market", "food", "beverage",
                "boutique", "office", "plaza"]),
    ("Fitness", ["gym", "fitness", "yoga", "crossfit", "martial arts",
                 "tennis", "sports", "football", "cricket", "stable",
                 "swimming", "bowling", "skating", "sporting", "sport",
                 "soccer"]),
    ("Wellness", ["spa", "massage", "therapy", "sauna", "escape room",
                  "psychic", "astrologer"]),
    ("Entertainment", ["theater", "cinema", "concert", "comedy",
                       "recreation", "bingo", "music", "auditorium", "jazz",
                       "blues", "stadium", "gun", "race", "track"]),
    ("Cultural", ["museum", "art", "gallery", "library", "historic",
                  "landmarks", "monument", "tour", "opera", "exhibit",
                  "memorial"]),
    ("Outdoor", ["park", "trail", "beach", "zoo", "hiking", "playground",
                 "outdoors", "tunnel", "fountain", "scenic", "nature",
                 "aquarium", "campground", "camp", "farm", "canal"]),
    ("Transport", ["train", "bus", "subway", "parking", "taxi", "tube",
                   "dealership", "automotive", "car rental",
                   "shipping", "motorcycle", "fuel station",
                   "harbor", "marina"]),
    ("Healthcare", ["hospital", "clinic", "pharmacy", "dentist", "veterinary",
                    "medicine", "doctor", "surgeon", "surgery", "healthcare",
                    "physiotherapist", "physician", "psycho", "medical",
                    "nutritionist", "ambulance", "assisted living"]),
    ("Services", ["bank", "atm", "post", "salon", "barber", "laundry",
                  "child care", "agency", "photographer", "chimney",
                  "veterinarian", "telecommunication", "pet", "wedding",
                  "architecture", "upholstery", "cleaning", "computer",
                  "photography", "audiovisual", "manufacturer", "auction",
                  "designer", "event", "renewable energy", "hotel",
                  "wholesaler"]),
    ("Organization", ["community", "government", "assistance", "legal",
                      "environmental", "non-profit", "charity", "youth",
                      "city hall", "disabled", "military", "embassy",
                      "consulate", "agriculture", "forestry", "courthouse",
                      "police", "fire", "station"]),
    ("Education", ["school", "learning", "tutoring", "preschool",
                   "kindergarten", "university", "college", "education"]),
    ("Religion", ["church", "cathedral", "seminary", "mosque", "temple",
                  "synagogue", "faith", "monastery", "cemetery", "spiritual",
                  "kingdom hall"]),
    ("Home Improvement", ["hvac", "heating ventilating air conditioning",
                          "landscape", "garden", "smith", "contractor",
                          "home", "construction", "carpenter", "builder",
                          "plumber", "housing", "electrician", "locksmith",
                          "real estate"]),
]


# from collections import Counter
# flatten_arr = [category for item in inside_airbnb_df.amenities.to_list()
#                for category in item.split(', ')]
# counts = Counter(flatten_arr)
# print(counts)


def classify_category(category_name):
    category_name_lower = category_name.lower()
    for broad_category, keywords in BROAD_CATEGORIES:
        if any(keyword in category_name_lower for keyword in keywords):
            return broad_category
    return category_name


def get_nearby_categories(lat, lon, radius=100, limit=3):
    if not FOURSQUARE_API_KEY:
        return "API key missing"

    headers = {
        "Authorization": FOURSQUARE_API_KEY,
        "Accept": "application/json"
    }
    params = {
        "ll": f"{lat},{lon}",
        "radius": radius,
        "limit": limit
    }

    response = requests.get(FOURSQUARE_URL, headers=headers, params=params)

    if response.status_code == 200:
        data = response.json()
        categories = set()
        for place in data.get('results', []):
            category_list = place.get('categories', [])
            if category_list:
                category_name = category_list[0]['name']
                broad_category = classify_category(category_name)
                categories.add(broad_category)

        return ', '.join(categories) if categories else "None"
    else:
        return f"API error: {response.status_code}"


print('Applying Foursquare API results')
inside_airbnb_df['amenities'] = inside_airbnb_df.apply(
    lambda row: get_nearby_categories(row['latitude'],
                                      row['longitude']), axis=1
    )

inside_airbnb_df.reset_index(inplace=True, drop=True)
inside_airbnb_df.drop(inside_airbnb_df.index[-1], inplace=True)


def map_categories(amenities):
    categories = set()

    for amenity in amenities.split(', '):
        found = False
        for broad_category, keywords in BROAD_CATEGORIES:
            if any(re.search(rf"\b{keyword}\b", amenity, re.IGNORECASE)
                   for keyword in keywords):
                categories.add(broad_category)
                found = True
                break
        if not found:
            categories.add(amenity)

    return ', '.join(sorted(categories))


inside_airbnb_df['amenities'] = inside_airbnb_df['amenities'].apply(
    map_categories
    )

response = requests.get(TFL_API_URL)
if response.status_code == 200:
    data = response.json()
else:
    raise Exception(f"API error: {response.status_code}")

tube_stations = []
for stop_point in data["stopPoints"]:
    lat, lon = stop_point["lat"], stop_point["lon"]
    station_name = stop_point["commonName"]
    tube_stations.append((station_name, lat, lon))

tube_df = pd.DataFrame(tube_stations,
                       columns=["Station", "Latitude", "Longitude"]
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
    return nearest_station.Station.replace(
        " Underground Station", ""
        ).replace(" Station", ""), distance


print('Applying TfL API results')
inside_airbnb_df[['nearest_station', 'distance_to_station']] = \
    inside_airbnb_df.apply(
        lambda row: find_nearest_station(row['latitude'], row['longitude']),
        axis=1, result_type='expand'
        )

selected_data_file = (inside_airbnb_data_dir /
                      'selected_short_term_rentals_with_distances.csv')
if not selected_data_file.exists():
    inside_airbnb_df.to_csv(selected_data_file, index=False)

print('Done!')
