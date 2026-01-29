#!/usr/bin/env python
# coding: utf-8

from pathlib import Path
import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import xgboost as xgb
from sklearn.base import clone
from sklearn.metrics import (
    root_mean_squared_error, mean_absolute_error,
    mean_absolute_percentage_error, r2_score,
    )
from sklearn.model_selection import cross_val_score
from statsmodels.formula.api import ols
import joblib

# Working directories
home_dir = Path.home()
data_dir = home_dir / 'Programming/data/inside-airbnb/london/2024-12-11'
inside_airbnb_modified_data_dir = data_dir / 'modified/'

work_dir = home_dir / 'Programming/Python/machine-learning-exercises/'
inside_airbnb_work_dir = work_dir / 'short-term-rents-in-london/'

# Start of model analysis
start = time.perf_counter()

# Data preparation
inside_airbnb_data_file = (
    inside_airbnb_modified_data_dir /
    'selected_short_term_rentals_for_modeling.csv'
    )
inside_airbnb_df = pd.read_csv(
    inside_airbnb_data_file,
    keep_default_na=False, thousands=','
    )

# Dropping latitude and longitude due to lack of correlation with price
inside_airbnb_df = inside_airbnb_df.drop(['latitude', 'longitude'], axis=1)

# Limit rent prices to a maximum of £1000 per night
max_limit = 1000
inside_airbnb_df = inside_airbnb_df[inside_airbnb_df.price < max_limit]

df_full_train, df_test = train_test_split(inside_airbnb_df, test_size=0.2,
                                          random_state=33,
                                          stratify=inside_airbnb_df['borough'])
df_train, df_val = train_test_split(df_full_train, test_size=0.25,
                                    random_state=33,
                                    stratify=df_full_train['borough'])

# Transformation of price to log(price+1) in the training set
df_train['log_price'] = np.log1p(df_train['price'])
X_train = df_train.drop(['log_price', 'price'], axis=1)
y_train = df_train['log_price'].copy()

X_val = df_val.drop(['price'], axis=1)
y_val = df_val['price'].copy()
X_test = df_test.drop(['price'], axis=1)
y_test = df_test['price'].copy()

X_train = X_train.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)
X_val = X_val.reset_index(drop=True)
y_val = y_val.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)


# Data pipeline
cat_attribs = ['borough', 'property_type', 'room_type', 'first_amenity',
               'second_amenity', 'third_amenity']
num_attribs = ['bathrooms', 'bedrooms', 'accommodates', 'availability_365',
               'crime_rate', 'distance_to_nearest_tube_station',
               'days_from_last_review']

num_pipeline = make_pipeline(
    SimpleImputer(strategy='median'),
    StandardScaler(),
)

cat_pipeline = make_pipeline(
    SimpleImputer(strategy='most_frequent'),
    OneHotEncoder(handle_unknown='ignore', sparse_output=False),
)

preprocessing = ColumnTransformer([
    ('num', num_pipeline, num_attribs),
    ('cat', cat_pipeline, cat_attribs),
])

# fitting only occurs here!
X_train_prepared = preprocessing.fit_transform(X_train)
# using 'preprocessing' object to transform data frame
X_val_prepared = preprocessing.transform(X_val)
# using 'preprocessing' object to transform data frame
X_test_prepared = preprocessing.transform(X_test)


X_train_prepared_df = pd.DataFrame(
    data=X_train_prepared,
    columns=preprocessing.get_feature_names_out(),
    index=X_train.index,
)

X_val_prepared_df = pd.DataFrame(
    data=X_val_prepared,
    columns=preprocessing.get_feature_names_out(),
    index=X_val.index,
)

X_test_prepared_df = pd.DataFrame(
    data=X_test_prepared,
    columns=preprocessing.get_feature_names_out(),
    index=X_test.index,
)

# Training, validation and testing dataframe sizes
len_df = inside_airbnb_df.shape[0]
print(f'Size of dataframe: {inside_airbnb_df.shape}')
print(f'Training size: '
      f'{round(len(X_train_prepared_df)/len_df, 5)}')
print(f'Validation size: '
      f'{round(len(X_val_prepared_df)/len_df, 5)}')
print(f'Testing size: '
      f'{round(len(X_test_prepared_df)/len_df, 5)}\n')

data_algorithms = {
    'linear regression': LinearRegression(),
    'random forest regressor': RandomForestRegressor(random_state=42),
    'stocastic gradient descent regressor': SGDRegressor(random_state=42),
    'support vector regressor': SVR(),
    'XGBoost regressor': xgb.XGBRegressor(
        tree_method="hist",
        eval_metric=root_mean_squared_error,
        max_depth=10,
        n_estimators=100,
        verbosity=0,
        ),
    }

for name, model in data_algorithms.items():
    print(name.capitalize())
    model.fit(X_train_prepared_df, y_train)
    y_pred_log = model.predict(X_val_prepared_df)
    y_pred = np.expm1(y_pred_log)
    mae = mean_absolute_error(y_val, y_pred)
    print(f'Validation MAE for {name}: {round(mae, 5)}')
    mape = mean_absolute_percentage_error(y_val, y_pred)
    print(f'Validation MAPE for {name}: {round(mape*100, 1)}%')
    rmse = root_mean_squared_error(y_val, y_pred)
    print(f'Validation RMSE for {name}: {round(rmse, 5)}')
    r2 = r2_score(y_val, y_pred)
    print(f'Validation R2 for {name}: {round(r2, 5)}\n')
print('\n')


# Custom scorer
def price_space_scorer(metric):
    def scorer(estimator, X, y_log_true):
        # Predict in log space
        y_log_pred = estimator.predict(X)

        # Back-transform
        y_true = np.expm1(y_log_true)
        y_pred = np.expm1(y_log_pred)

        return metric(y_true, y_pred)

    return scorer


# Cross validation
scoring_methods = {
    'MAE': mean_absolute_error,
    'MAPE': mean_absolute_percentage_error,
    'RMSE': root_mean_squared_error,
    }
for name, model in data_algorithms.items():
    print(f'Cross validation for {name}')
    cloned_model = clone(model)
    for scorer_name, scorer in scoring_methods.items():
        custom_scorer = price_space_scorer(scorer)
        cv_scores = cross_val_score(
            cloned_model, X_train_prepared_df, y_train,
            scoring=custom_scorer, cv=5,
            )
        cv_sr = pd.Series(cv_scores).describe()
        if scorer_name == 'MAPE':
            print(f"Cross-validation {scorer_name} mean and std dev: "
                  f"{100*cv_sr.loc['mean']:.1f} ± "
                  f"{100*cv_sr.loc['std']:.1f} (%)")
        print(f"Cross-validation {scorer_name} mean and std dev: "
              f"{cv_sr.loc['mean']:.5f} ± {cv_sr.loc['std']:.5f}")
    print('\n')


# Results for test data set
print('Support vector regressor using test dataset')
svr = SVR()
svr.fit(X_train_prepared_df, y_train)
y_test_pred_log_svr = svr.predict(X_test_prepared_df)
y_test_pred_svr = np.expm1(y_test_pred_log_svr)
svr_mae = mean_absolute_error(y_test, y_test_pred_svr)
print(f'Test MAE for support vector regressor: {round(svr_mae, 5)}')
svr_mape = mean_absolute_percentage_error(y_test, y_test_pred_svr)
print(f'Test MAPE for support vector regressor: {round(svr_mape*100, 1)}%')
svr_rmse = root_mean_squared_error(y_test, y_test_pred_svr)
print(f'Test RMSE for support vector regressor: {round(svr_rmse, 5)}')
svr_r2_score = r2_score(y_test, y_test_pred_svr)
print(f'Test R2 for support vector regressor: {round(svr_r2_score, 5)}\n\n')


# OLS regression results from statsmodels
inside_airbnb_df['log_price'] = np.log1p(inside_airbnb_df['price'])
lm = ols(
    'log_price ~ borough + property_type + room_type + first_amenity '
    '+ second_amenity + third_amenity + bathrooms + bedrooms + accommodates '
    '+ availability_365 + crime_rate + distance_to_nearest_tube_station '
    '+ days_from_last_review',
    data=inside_airbnb_df
    )
model = lm.fit()
mean_log_price = np.mean(y_train)
mean_log_price = mean_log_price.round(2)
log_rse = np.sqrt(model.mse_resid)
log_rse = log_rse.round(2)
print(f'Mean and residual standard error of the logarithmic price: '
      f'{mean_log_price} ± '
      f'{log_rse}')

mean_price = (
    np.exp(mean_log_price) * (np.exp(log_rse/2)) - 1
    ).round(2)
rse = np.sqrt(
    (np.exp(log_rse**2) - 1) *
    np.exp(2*mean_log_price + log_rse**2)
    ).round(2)
print(f'Mean and residual standard error of the price: '
      f'{mean_price} ± '
      f'{rse} ({(mean_price-rse).round(2)}, '
      f'{(mean_price+rse).round(2)}) (£)')

perc_error = (100*rse/mean_price).round(1)
print(f'Error percentage of the residual '
      f'standard error to the mean: {perc_error}%\n\n')

print(model.summary().tables[0], end='\n\n')


# Calculating support vector regressor model on entire
# data set and saving model to a pickle file
print('Calculating support vector regressor model on entire data set')
full_pipeline = Pipeline([
    ('preprocessing', preprocessing),
    ('svm_regressor', SVR(C=1.0, epsilon=0.1)),
])
y_full = inside_airbnb_df['log_price'].copy()
X_full = inside_airbnb_df.drop(['log_price'], axis=1)
full_pipeline.fit(X_full, y_full)
model_file = inside_airbnb_work_dir / 'model.pkl'
print(f'Saving model file to {model_file}')
joblib.dump(full_pipeline, model_file)
end = time.perf_counter()
print(f"Total time: {round((end - start)/60, 2)} minutes")
