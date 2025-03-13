# Project on short term rentals in London, UK

This is a data project with the intent of predicting prices of short-term rents of flats in London. The source of the data for this projects comes from the London page of the Inside AirBNB website. Due to the terms of service of major UK home realtors that don't permit web scraping, I decided to use Inside AirBNB limiting myself on short-term rentals of entire flats. Short term flat rental are those greater than 30 days, of which there are 2059 listed in London currently as of 12th March 2025.

## Data sources

There are different data sources that are used for this data analysis. Here is a list of them.

### Short-term housing data

The main bulk of the data comes from the [Inside AirBNB](https://insideairbnb.com/ "https://insideairbnb.com/") website, which is generated from data anonymously scraped from AirBNB host profiles in a number of major international cities. The data for the city of London can be found at the [specific webpage for London](https://insideairbnb.com/london/ "https://insideairbnb.com/london/").

The model was created using the data from 11 December 2024. The time series data was generated using additional data from 19 March 2024, 14 June 2024 and 6 September 2024.

### Crime data

The crime data are for the number of crimes committed at the geographic level of the London boroughs per month according to crime type, as recorded by the London Metropolitan Police Service (MPS). The data can be downloaded from the London Data Store at [MPS Recorded Crime: Geographic Breakdown](https://data.london.gov.uk/dataset/recorded_crime_summary "https://data.london.gov.uk/dataset/recorded_crime_summary").

### Amenities data

I decided to retrieve data on the amenities located in the vicinities of rentals using the Place Match endpoint of the Foursquare API. The details of the Foursquare API Developers documentation on the Place Match endpoint can be found [here](https://api.foursquare.com/v3/places/match "https://api.foursquare.com/v3/places/match").


## Model generation

