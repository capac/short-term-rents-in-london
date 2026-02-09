# Project on short term rentals in London UK

## Summary

One of the major challenges for letting agencies and private citizens alike is to determine the correct market price for their short-term property listings. Prices that are too low may result in a lost of potential revenue, while prices that are too high may result in too few bookings. Moreover, there are no free services that provide an accurate price estimation for property listings. Using data from [Inside Airbnb](https://insideairbnb.com/ "https://insideairbnb.com/"), a data and advocacy website about Airbnb's impact on residential communities, and from other public data sources, I've built a [Streamlit](https://streamlit.io/ "https://streamlit.io/") web app to estimate the short-term rental prices for property listings based on their characteristics and location in London UK for December 2024. On average the web app estimates a percentage difference of 25.2% between actual and predicted rent price. If you're curious, take a look at the web app at the link below.

[Short-Term Rental Price Estimator &bull; Streamlit](https://rental-pricing-app.streamlit.app/ "https://rental-pricing-app.streamlit.app/")

## Data sources

### Short-term housing data

Due to the terms of service of major UK home realtors which don't permit web scraping, I decided to use the [Inside Airbnb](https://insideairbnb.com/london/ "https://insideairbnb.com/london/") website and filter the data on short-term rentals of entire flats or buildings for London, UK. The data set is anonymously scraped from Airbnb host profiles in a number of major international cities. The data for London itself can be found here at the [webpage for London](https://insideairbnb.com/london/ "https://insideairbnb.com/london/"). The specific data set used in this analysis was scraped 11 December 2024. Because of [planning regulations in the Greater London area](https://www.gov.uk/government/news/short-term-lets-rules-to-protect-communities-and-keep-homes-available "https://www.gov.uk/government/news/short-term-lets-rules-to-protect-communities-and-keep-homes-available") made to protect communities and keep homes available, short-term rentals are limited to 90 nights per year.

For the analysis I employed several features of the Inside Airbnb data set, notably the borough of the property location, the property and room types, the amount of people the property can accomodate, the number of bedrooms and bathrooms, the price per night, the availability of the property over the last year, the number of days from the last review (if any), and latitude and longitude of the property. For the purposes of anonymity, these geographic coordinates are randomly offset by 150 meters.

Beyond the main data source from Inside Airbnb, the data were also enriched by adding crime rate per borough, distance of the property to the nearest Tube Underground station, and local amenities in the vicinity of the property. Here follows a more detailed list of them.

### Crime data

The crime rate data by London borough are retrieved from the [CrimeRate](https://crimerate.co.uk/ "https://crimerate.co.uk/") webpage for the [Greater London Crime Statistics](https://crimerate.co.uk/london "https://crimerate.co.uk/london") website. It regards the crime rate in each borough over the period from October 2023 to September 2024.

### Transport data

I added the distances from each rental unit to the closest Tube Underground station using the `StopPoint` [endpoint](https://api.tfl.gov.uk/StopPoint/Mode/tube "https://api.tfl.gov.uk/StopPoint/Mode/tube") from the Transport for London (TfL) developer API, from which I extracted the geographic coordinates of each Tube station. Afterwards, I calculated the distance from the nearest Tube station to the rental unit using the [GeoPy](https://github.com/geopy/geopy "https://github.com/geopy/geopy") Python package.

### Amenities data

I also retrieved data on the amenities located in the vicinities of the property rentals using [Foursquare](https://foursquare.com/ "https://foursquare.com/"). Specifically, I used the `Place/Search` endpoint, the details of which can be found [at the Foursquare API](https://api.foursquare.com/v3/places/search "https://api.foursquare.com/v3/places/search"). At most three amenity categories for each property location are retrieved using the Foursquare API, which are then set to one of ten broad category types. These types can be easily viewed in the web app under one of the `Nearby amenity category` drop-drop menus.

## Data preparation

For the purposes of the data preparation, null values were removed from the data set and only properties reviewed within the last six months were retained in the data set. Also just the most frequent property types, present at least 30 times, were kept for the analysis. These types can be easily selected and viewed in the web app under the `Property Type` drop-drop menu.

As one can see from Figure 1, the histogram of the price distribution of short-term rentals is heavily skewed towards the positive end of the _x_ axis, so for the machine learning model generation the price feature was transformed into the logarithm of the price feature plus 1. This new feature is much more normally distributed compared to the previous feature, and helps to produce machine learning models with more accurate predictions.

## Exploratory data analysis

A few initial observations can already be gleaned from the series of histograms in Figure 1. The top two histograms of latitude and logitude show a bimodal distribution that can be ascribed primarily to the Thames river for the first histogram, but is harder to ascertain for the second. This could be due to the presence of more property listings around the major London parks, mostly present in the east and west of the city.

From the price histogram we observe a sharp drop in short-term rental prices per night, with a distribution heavily skewed towards the positive _x_ axis. I chose to limit the _x_ axis to £1000 as the upper limit, but there are several outliers that are even further up in price. These outliers however are still present in the model analysis. A similar distribution behavior is visible in the histogram for the number of days from the last review.

![attribute_histogram_plots](plots/histograms/attribute_histogram_plots.png "attribute_histogram_plots.png")

### Borough plots

From Figure 2 one sees that most short-term rentals are present in the borough of Westminister, with Kensington & Chelsea, Camden, and Tower Hamlets listed in second, third, and fourth places respectively. The borough with the least number of rentals is Sutton.

![number-rentals-per-borough](plots/maps/number-rentals-per-borough.png "number-rentals-per-borough.png")

As for the median price, the borough with the highest median price per rental belongs unsurprisingly to Kensington & Chelsea, which is the borough with the most exclusive and expensive properties of the city, followed closely by the boroughs of Westminster, Camden and Lambeth.

![median-price-per-borough](plots/maps/median-price-per-borough.png "median-price-per-borough.png")

### Cluster analysis

An analysis of possible clusters of rental properties in London was undertaken by finding the number of clusters that maximized the [Silhouette coefficient](https://en.wikipedia.org/wiki/Silhouette_(clustering) "https://en.wikipedia.org/wiki/Silhouette_(clustering)"). The maximum Silhouette score (0.524) is achieved with just one big cluster of properties that covers the entire city, with no other discernible subclusters visible. The maximum Silhouette score was calculated using [DBSCAN](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html#sklearn.cluster.DBSCAN "https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html#sklearn.cluster.DBSCAN") in Scikit-Learn and `eps=0.03` and `min_sample=400`. Fine-tuning the `eps` and `min_sample` parameters, which are the most important parameters for DBSCAN, doesn't offer more than one cluster even at the expense of lower Silhouette scores.

## Model generation

A few regression algorithms from [Scikit-Learn](https://scikit-learn.org/stable/ "https://scikit-learn.org/stable/") were used to model the data. These were [linear regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html "https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html"), [random forest regressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html#sklearn.ensemble.RandomForestRegressor "https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html#sklearn.ensemble.RandomForestRegressor"), [stocastic gradient descent regressor](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDRegressor.html#sklearn.linear_model.SGDRegressor "https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDRegressor.html#sklearn.linear_model.SGDRegressor"), [support vector regressor](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html#sklearn.svm.SVR "https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html#sklearn.svm.SVR") and [XGBoost regressor](https://xgboost.readthedocs.io/en/stable/python/python_api.html#xgboost.XGBRegressor "https://xgboost.readthedocs.io/en/stable/python/python_api.html#xgboost.XGBRegressor"). The performance of the algorithms was determined based on the best (lowest) root mean squared error (RMSE). The support vector regressor algorithm achieved the best RSME of 0.37884, which was determined using 10-fold cross validation. The RMSE for the support vector regressor using the test data set was slightly better at 0.36268. A grid search analysis on the support vector regressor produced the best RMSE value with `C=1.0` and `epsilon=0.1`, which are the default values for the support vector regressor.

### RMSE and $R^2$ validation performance for all models

|                 Model                |   RMSE  |  $R^2$  |
|--------------------------------------|---------|---------|
| Linear regression                    | 0.38738 | 0.69988 |
| Random forest regressor              | 0.37294 | 0.72184 |
| Stocastic gradient descent regressor | 0.3911  | 0.69409 |
| Support vector regressor             | 0.37177 | 0.72358 |
| XGBoost regressor                    | 0.39077 | 0.69461 |

### Cross-Validation RMSE Results

|                 Model        |  RMSE Mean | RMSE Std. Dev |
|------------------------------|------------|---------------|
| Linear Regression            |   0.39468  |    0.03332    |
| Random Forest Regressor      |   0.38261  |    0.03354    |
| Stochastic Gradient Descent  |   0.39836  |    0.03358    |
| Support Vector Regressor     |   0.37884  |    0.03443    |
| XGBoost Regressor            |   0.39484  |    0.02909    |

### Support vector regressor using test dataset

| Metric   | Value   |
|----------|---------|
| RMSE     | 0.36268 |
| $R^2$    | 0.73458 |

The mean and residual standard error of the price (in GBP) is 154.24 ± 55.87 (lower end 98.37, upper end 210.11). The error percentage of the residual standard error to the mean is 36.2%. This is the expected average variation of the price compared to the mean.

### Residual statistics (price & log-price)

| Metric                                  | Value          |
|-----------------------------------------|----------------|
| Mean Log Price                          | 4.85           |
| Residual Std. Error (Log Price)         | 0.39           |
| Mean Price (£)                          | 154.24         |
| Residual Std. Error (£)                 | 55.87          |
| Price Interval (£)                      | 98.37, 210.11  |
| Residual Std. Error as % of Mean Price  | 36.2%          |

[Ordinary least squares (OLS)](https://www.statsmodels.org/stable/index.html/generated/statsmodels.regression.linear_model.OLS.html#statsmodels.regression.linear_model.OLS "https://www.statsmodels.org/stable/index.html/generated/statsmodels.regression.linear_model.OLS.html#statsmodels.regression.linear_model.OLS") from [Statsmodels](https://www.statsmodels.org/stable/index.html "https://www.statsmodels.org/stable/index.html") allows us to calculate the F-statistic to determine the likelihood of association between the predictors and the outcome. In the regression results the F-statistics returns a value of 741.5 >> 1, and points to a very high association between at least one predictor and the outcome.

### OLS Regression Summary

| Statistic                      | Value        |
|--------------------------------|--------------|
| Dependent Variable             | log_price    |
| R-squared                      | 0.701        |
| Adjusted R-squared             | 0.700        |
| F-statistic                    | 741.5        |
| Prob (F-statistic)             | 0.00         |
| Observations                   | 26,665       |
| Degrees of Freedom (Model)     | 84           |
| Degrees of Freedom (Residuals) | 26,580       |
| Log-Likelihood                 | -12,702      |
| AIC                            | 25,570       |
| BIC                            | 26,270       |
| Covariance Type                | Nonrobust    |

## Conclusions

The project was really enjoyable, and the part I liked the most was creating a new data set by data enrichment from other data sources. Once the model was generated, I set up an interactive web app with [Streamlit](https://streamlit.io/cloud "https://streamlit.io/cloud") that allows users to determine the nightly price for their short-term rental properties according to the features described above. Check it out at:

[Short-Term Rental Price Estimator &bull; Streamlit](https://rental-pricing-app.streamlit.app/ "https://rental-pricing-app.streamlit.app/")

Enjoy!

For the data analysis, the following software packages were used: [Scikit-Learn](https://scikit-learn.org/stable/ "https://scikit-learn.org/stable/") (version 1.6.1), [Matplotlib](https://matplotlib.org/ "https://matplotlib.org/") (version 3.10.0), [Statsmodels](https://www.statsmodels.org/stable/index.html "https://www.statsmodels.org/stable/index.html") (version 0.14.4), [XGBoost](https://xgboost.readthedocs.io/en/stable/ "https://xgboost.readthedocs.io/en/stable/") (version 3.0.1), [contextily](https://github.com/geopandas/contextily "https://github.com/geopandas/contextily") (version 1.6.2), [GeoPy](https://github.com/geopy/geopy "https://github.com/geopy/geopy") (version 2.4.1), [GeoPandas](https://geopandas.org/en/stable/ "https://geopandas.org/en/stable/") (version 1.0.1), [Shapely](https://github.com/shapely/shapely "https://github.com/shapely/shapely") (2.0.6), [Streamlit](https://streamlit.io/cloud "https://streamlit.io/cloud") (version 1.45.0), [Pandas](https://pandas.pydata.org/ "https://pandas.pydata.org/") (version 2.2.3), [joblib](https://joblib.readthedocs.io/en/stable/ "https://joblib.readthedocs.io/en/stable/") (version 1.4.2) and [NumPy](https://numpy.org/ "https://numpy.org/") (version 1.26.4).
