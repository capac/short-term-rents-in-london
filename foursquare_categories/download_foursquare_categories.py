import requests
import json
import os

CLIENT_ID = os.environ['FOURSQUARE_CLIENT_ID']
CLIENT_SECRET = os.environ['FOURSQUARE_CLIENT_SECRET']
VERSION = '20250528'

url = (f'https://api.foursquare.com/v2/venues/categories?'
       f'client_id={CLIENT_ID}&client_secret={CLIENT_SECRET}&v={VERSION}')


response = requests.get(url)

if response.status_code == 200:
    categories = response.json()
    with open('foursquare_categories.json', 'w') as f:
        json.dump(categories, f, indent=2)
    print("Categories saved to 'foursquare_categories.json'")
else:
    print(f"Failed to retrieve categories: {response.status_code}")
