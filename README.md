# Project on short term rentals in London UK

## Summary

The intent of this data project is to predict prices of short-term flat rentals in London in 2024. Due to the terms of service of major UK home realtors which don't allow web scraping, I decided to use the [Inside AirBNB](https://insideairbnb.com/london/ "https://insideairbnb.com/london/") website and filter the data on short-term rentals of entire flats or buildings for London, UK. For the purposes of this study, short-term rentals are those where the overnight stay is from 1 to 999 days.

## Data sources

I decided to employ just a few of the features of the data set, notably the borough of the property location, the property and room types, the amount of people the property can accomodate, the number of bedrooms and bathrooms, price per night, the availability of the property over the last year, the number of days from the last review, and latitude and longitude of the property, which for purposes of anonymity are randomly offset by 150 meters. The data was also enriched by adding crime rate per borough, distance of the property to the nearest Tube Underground station, and local amenities in the vicinity of the property. 

Beyond the main source of data, there are several, other sources used for data enrichment. Here follows a list of them.

### Short-term housing data

The main bulk of the data comes from the [Inside AirBNB](https://insideairbnb.com/ "https://insideairbnb.com/") website, which is generated from data anonymously scraped from AirBNB host profiles in a number of major international cities. The data for the city of London can be found at the [webpage for London](https://insideairbnb.com/london/ "https://insideairbnb.com/london/"). The specific data used in this analysis was scraped 11 December 2024.

### Crime data

The crime rate data by London borough is retrieved from the [CrimeRate](https://crimerate.co.uk/ "https://crimerate.co.uk/") webpage for the [Greater London Crime Statistics](https://crimerate.co.uk/london "https://crimerate.co.uk/london"). It regards the crime rate in each borough over the period from October 2023 to September 2024.

### Transport data

I added the distances from each rental unit to the closest Tube Underground station using the `StopPoint` [endpoint](https://api.tfl.gov.uk/StopPoint/Mode/tube "https://api.tfl.gov.uk/StopPoint/Mode/tube") from the Transport for London (TfL) developer API, from which I extracted the geographical coordinates of each Tube station. Afterwards, I calculated the distance from the nearest Tube station to the rental unit using the [GeoPy](https://github.com/geopy/geopy "https://github.com/geopy/geopy") Python package.

### Amenities data

I also decided to retrieve data on the amenities located in the vicinities of the property rentals using [Foursquare](https://foursquare.com/ "https://foursquare.com/"). Specifically, I used the `Place/Search` endpoint, the details of which can be found [here](https://api.foursquare.com/v3/places/search "https://api.foursquare.com/v3/places/search").

## Exploratory data analysis

The histogram of the price distribution of short-term rentals is heavily skewed towards the positive end of the _x_ axis, so it is a good idea to replace the price feature plus 1 with its logarithm. The logarithm of the price feature plus 1 is much more normally distributed as a consequence of the transformation.

The price of most short-term rentals is less than £200 per night, but the distribution is heavily skewed towards the positive _x_ axis. I chose to limit the _x_ axis to £1000 as the upper limit, but there are several outliers that are even further up in price. The outliers however are still present in the model analysis.

Most short-term rentals are in the Westminister borough, with Kensington & Chelsea, Camden, and Tower Hamlets listed in second, third, and fourth places. The borough with the least number of rentals is Sutton.

![](plots/histograms/attribute_histogram_plots.png "attribute_histogram_plots.png")

### Cluster analysis

An analysis of possible clusters of rental properties in London was determined by finding the number of clusters that maximized the [Silhouette coefficient](https://en.wikipedia.org/wiki/Silhouette_(clustering) "https://en.wikipedia.org/wiki/Silhouette_(clustering)"). The maximum Silhouette score (0.524) is achieved with just one big cluster of properties that covers the entire city, with no other discernible subclusters visible. The Silhouette score was calculated using [DBSCAN](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html#sklearn.cluster.DBSCAN "https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html#sklearn.cluster.DBSCAN") in Scikit-Learn, using `eps=0.03` and `min_sample=400`. Fine-tuning the `eps` and `min_sample` parameters, which are the most important parameters for DBSCAN, doesn't offer more than one cluster even at the expense of lower Silhouette scores.

## Model generation

A few regression algorithms from [Scikit-Learn](https://scikit-learn.org/stable/ "https://scikit-learn.org/stable/") were used to model the data. These were [linear regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html "https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html"), [random forest regressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html#sklearn.ensemble.RandomForestRegressor "https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html#sklearn.ensemble.RandomForestRegressor"), [stocastic gradient descent regressor](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDRegressor.html#sklearn.linear_model.SGDRegressor "https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDRegressor.html#sklearn.linear_model.SGDRegressor"), [support vector regressor](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html#sklearn.svm.SVR "https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html#sklearn.svm.SVR") and [XGBoost regressor](https://xgboost.readthedocs.io/en/stable/python/python_api.html#xgboost.XGBRegressor "https://xgboost.readthedocs.io/en/stable/python/python_api.html#xgboost.XGBRegressor"). The performance of the algorithms was determined based on the best (lowest) root mean squared error (RMSE). The support vector regressor algorithm achieved the best RSME of 0.37884, which was determined using 10-fold cross validation. The RMSE for the support vector regressor using the test data set was slightly better at 0.36268.

A grid search analysis on the support vector regressor produced the best RMSE value with `C=1.0` and `epsilon=0.1`, which are the default values for the support vector regressor.

[Ordinary least squares (OLS)](https://www.statsmodels.org/stable/index.html/generated/statsmodels.regression.linear_model.OLS.html#statsmodels.regression.linear_model.OLS "https://www.statsmodels.org/stable/index.html/generated/statsmodels.regression.linear_model.OLS.html#statsmodels.regression.linear_model.OLS") from [Statsmodels](https://www.statsmodels.org/stable/index.html "https://www.statsmodels.org/stable/index.html") allows us to calculate the F-statistic to determine the likelihood of association between the predictors and the outcome. In the regression results the F-statistics returns a value of 741.5 >> 1, and points to a very high association between at least one predictor and the outcome.

The mean and residual standard error of the price (in GBP) is 154.24 ± 55.87 (lower end 98.37, upper end 210.11). The error percentage of the residual standard error to the mean is 36.2%. This is the expected average variation of the price compared to the mean.

## Conclusions

The project was really enjoyable, and the part I liked the most was creating a new data set by data enrichment from other data sources. Once the model was generated, I set up an interactive web app with [Streamlit](https://streamlit.io/cloud "https://streamlit.io/cloud") that allows users to determine the nightly price for their short-term rental properties according to the features described above. Check it out at:

[Short-Term Rental Price Estimator &bull; Streamlit](https://rental-pricing-app.streamlit.app/ "https://rental-pricing-app.streamlit.app/")

Enjoy!

For the data analysis, the following software packages were used: [Scikit-Learn](https://scikit-learn.org/stable/ "https://scikit-learn.org/stable/") (version 1.6.1), [Matplotlib](https://matplotlib.org/ "https://matplotlib.org/") (version 3.10.0), [Statsmodels](https://www.statsmodels.org/stable/index.html "https://www.statsmodels.org/stable/index.html") (version 0.14.4), [XGBoost](https://xgboost.readthedocs.io/en/stable/ "https://xgboost.readthedocs.io/en/stable/") (version 3.0.1), [contextily](https://github.com/geopandas/contextily "https://github.com/geopandas/contextily") (version 1.6.2), [GeoPy](https://github.com/geopy/geopy "https://github.com/geopy/geopy") (version 2.4.1), [GeoPandas](https://geopandas.org/en/stable/ "https://geopandas.org/en/stable/") (version 1.0.1), [Shapely](https://github.com/shapely/shapely "https://github.com/shapely/shapely") (2.0.6), [Streamlit](https://streamlit.io/cloud "https://streamlit.io/cloud") (version 1.45.0), [Pandas](https://pandas.pydata.org/ "https://pandas.pydata.org/") (version 2.2.3), [joblib](https://joblib.readthedocs.io/en/stable/ "https://joblib.readthedocs.io/en/stable/") (version 1.4.2) and [NumPy](https://numpy.org/ "https://numpy.org/") (version 1.26.4).
