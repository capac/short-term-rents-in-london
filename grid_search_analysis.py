#!/usr/bin/env python
# coding: utf-8

from pathlib import Path
import time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import (
    root_mean_squared_error, mean_absolute_error,
    mean_absolute_percentage_error, r2_score,
    )
import joblib

# working directories
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
inside_airbnb_df = pd.read_csv(inside_airbnb_data_file,
                               keep_default_na=False, thousands=',')

# Dropping latitude and longitude due to lack of correlation with price
inside_airbnb_df = inside_airbnb_df.drop(['latitude', 'longitude'], axis=1)

# Limit rent prices to a maximum of £1000 per night
max_limit = 1000
inside_airbnb_df = inside_airbnb_df[inside_airbnb_df.price < max_limit]

# Split dataset in a 80% training set and 20% testing set
df_train, df_test = train_test_split(
    inside_airbnb_df, test_size=0.2, random_state=33,
    stratify=inside_airbnb_df['borough']
    )

# Transformation of price to log(price+1) in the training set
df_train['log_price'] = np.log1p(df_train['price'])
X_train = df_train.drop(['log_price', 'price'], axis=1)
y_train = df_train['log_price'].copy()

X_test = df_test.drop(['price'], axis=1)
y_test = df_test['price'].copy()

X_train = X_train.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)
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
X_test_prepared = preprocessing.transform(X_test)

X_train_prepared_df = pd.DataFrame(
    data=X_train_prepared,
    columns=preprocessing.get_feature_names_out(),
    index=X_train.index,
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
print(f'Testing size: '
      f'{round(len(X_test_prepared_df)/len_df, 5)}\n')

# Grid search for support vector regressor
param_grid = [
    {'C': [0.001, 0.01, 0.1],
     'epsilon': [0.001, 0.01, 0.1]},
    {'C': [0.1, 1, 10],
     'epsilon': [0.1, 1, 10]},
    {'C': [10, 100, 1000],
     'epsilon': [10, 100, 1000]},
    ]


# Custom scorer for grid search cross validation
def price_space_scorer(metric):
    def scorer(estimator, X, y_log_true):
        # Predict in log price space
        y_log_pred = estimator.predict(X)
        # Back-transform to price
        y_true = np.expm1(y_log_true)
        y_pred = np.expm1(y_log_pred)
        return metric(y_true, y_pred)
    return scorer


custom_scorer = price_space_scorer(r2_score)
grid_search = GridSearchCV(
    SVR(), param_grid,
    cv=5, n_jobs=-1, error_score='raise',
    scoring=custom_scorer)
grid_search.fit(X_train_prepared_df, y_train)

print(f'Best parameters: {grid_search.best_params_}\n')

cv_res = pd.DataFrame(grid_search.cv_results_)
cv_res.sort_values(by='mean_test_score', ascending=False, inplace=True)
cols = [
    'param_C', 'param_epsilon', 'rank_test_score',
    'mean_test_score', 'std_test_score'
    ]
cv_res[cols] = cv_res[cols].round(5).astype(np.float64)
print('Best grid search values for support vector regressor\n')
cv_res = cv_res.reset_index(drop=True)
print(cv_res[cols])
print('\n')

# Results for test data set
print('Support vector regressor using test dataset')
svr = SVR(C=cv_res.at[0, 'param_C'], epsilon=cv_res.at[0, 'param_epsilon'])
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


# Calculating support vector regressor model on entire
# data set and saving model to a pickle file
print('Calculating support vector regressor model on entire data set')
full_pipeline = Pipeline([
    ('preprocessing', preprocessing),
    ('svm_regressor',
     SVR(C=cv_res.at[0, 'param_C'], epsilon=cv_res.at[0, 'param_epsilon'])),
    ])
inside_airbnb_df['log_price'] = np.log1p(inside_airbnb_df['price'])
y_full = inside_airbnb_df['log_price'].copy()
X_full = inside_airbnb_df.drop(['log_price'], axis=1)
full_pipeline.fit(X_full, y_full)
model_file = inside_airbnb_work_dir / 'model.pkl'
print(f'Saving model file to {model_file}')
joblib.dump(full_pipeline, model_file)
end = time.perf_counter()
print(f"\nTotal time: {round((end - start)/60, 2)} minutes")
