# Project on short term rentals in London, UK

## Summary

The intent of this data project is to predict prices of short-term flat rentals in London in 2024. Due to the terms of service of major UK home realtors whihc don't allow against web scraping, I decided to use the [Inside AirBNB](https://insideairbnb.com/london/ "https://insideairbnb.com/london/") website and filter the data on short-term rentals of entire flats or buildings for London, UK. For the purposes of this study short-term flat rentals are those greater than 30 days, of which there are 2059 listed in London currently as of 12th March 2025.

## Data sources

I decided to employ just a few of the features of the data set, notably the number of bedrooms and bathrooms, price per night, borough of the property location, and latitude and longitude of the property, which for purposes of anonymity are randomly offset by 150 meters. The data was also enriched by adding crime rate per borough, local amenities in the vicinity of the property and distance from the property to the nearest Tube station. Beyond the main source of data, there are several, other sources used for data enrichment. Here follows a list of them.

### Short-term housing data

The main bulk of the data comes from the [Inside AirBNB](https://insideairbnb.com/ "https://insideairbnb.com/") website, which is generated from data anonymously scraped from AirBNB host profiles in a number of major international cities. The data for the city of London can be found at the [specific webpage for London](https://insideairbnb.com/london/ "https://insideairbnb.com/london/").

The model was created using the data from 11 December 2024. The time series data was generated using additional data from 19 March 2024, 14 June 2024 and 6 September 2024.

### Crime data

The crime rate data by London borough is retrieved from the webpage on the [Greater London Crime Statistics](https://crimerate.co.uk/london "https://crimerate.co.uk/london") on [CrimeRate](https://crimerate.co.uk/ "https://crimerate.co.uk/"). It regards crime in each borough over the period from October 2023 to September 2024.

### Amenities data

I decided to retrieve data on the amenities located in the vicinities of rentals using the `Places/Search` endpoint of the Foursquare API. The details of the Foursquare API Developers documentation on the Place Search endpoint can be found [here](https://api.foursquare.com/v3/places/search "https://api.foursquare.com/v3/places/search").

### Transport data

I also added the distances from each rental unit to the closest Tube station using the `StopPoint` [endpoint](https://api.tfl.gov.uk/StopPoint/Mode/tube "https://api.tfl.gov.uk/StopPoint/Mode/tube") from the Transport for London (TfL) developer API, from which I extracted the geographical coordinates of each Tube station and calculated the distance to the rental unit using the [GeoPy](https://github.com/geopy/geopy "https://github.com/geopy/geopy") package.

## Exploratory data analysis

The price distribution of short-term rentals is heavily skewed towards the positive end of the _x_ axis, so it is a good idea to replace the price feature with its logarithm. The logarithm of the price feature is much more normally distrubuted as a consequence of the transformation.

The great majority short-term rentals are listed with a minimum stay of thirty days, with only a few exceptions for longer stays.

The price of most short-term rentals is less than £200 per night, but the distribution is heavily skewed towards the positive _x_ axis. I chose to limit the _x_ axis to £1000 as the upper limit, but there are outliers that are even further up in price.

Most short-term rentals are in the Westminister borough, with Kensington & Chelsea and Tower Hamlets listed in second and third places.

## Model generation

## Conclusions
