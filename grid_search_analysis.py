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
from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score

home_dir = Path.home()
inside_airbnb_data_dir = (
    home_dir /
    'Programming/data/inside-airbnb/london'
)
inside_airbnb_work_dir = (
    home_dir /
    'Programming/Python/machine-learning-exercises/short-term-rents-in-london'
)

# Start of model analysis
start = time.perf_counter()

# Data preparation
inside_airbnb_data_file = (
    inside_airbnb_data_dir /
    'selected_short_term_rentals_for_modeling.csv'
)
inside_airbnb_df = pd.read_csv(inside_airbnb_data_file,
                               keep_default_na=False, thousands=',')

inside_airbnb_df['log_price'] = np.log1p(inside_airbnb_df['price'])
inside_airbnb_df = inside_airbnb_df.drop('price', axis=1)

inside_airbnb_df = inside_airbnb_df.drop(['latitude', 'longitude'], axis=1)

df_full_train, df_test = train_test_split(inside_airbnb_df,
                                          test_size=0.2, random_state=33,
                                          stratify=inside_airbnb_df['borough'])
df_train, df_val = train_test_split(df_full_train,
                                    test_size=0.25, random_state=33,
                                    stratify=df_full_train['borough'])


X_train = df_train.drop(['log_price'], axis=1)
y_train = df_train['log_price'].copy()
X_val = df_val.drop(['log_price'], axis=1)
y_val = df_val['log_price'].copy()
X_test = df_test.drop(['log_price'], axis=1)
y_test = df_test['log_price'].copy()

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

print(f'Training size: '
      f'{round(len(X_train_prepared_df)/len(inside_airbnb_df), 5)}')
print(f'Validation size: '
      f'{round(len(X_val_prepared_df)/len(inside_airbnb_df), 5)}')
print(f'Testing size: '
      f'{round(len(X_test_prepared_df)/len(inside_airbnb_df), 5)}\n')


# Predictive modeling
# Support vector regressor
svr = SVR()
svr.fit(X_train_prepared_df, y_train)
y_pred_svr = svr.predict(X_val_prepared_df)
svr_rmse = root_mean_squared_error(y_val, y_pred_svr)
print(f'RMSE for support vector regressor: {round(svr_rmse, 5)}')
svr_r2_score = r2_score(y_val, y_pred_svr)
print(f'R2 for support vector regressor: {round(svr_r2_score, 5)}\n')


# Cross validation
# Support vector regressor
svr_rmses = -cross_val_score(
    svr, X_train_prepared_df,
    y_train,
    scoring='neg_root_mean_squared_error',
    cv=10)
svr_rmse_sr = pd.Series(svr_rmses)
print(f"Cross validation mean and standard deviation: "
      f"{svr_rmse_sr.describe().loc['mean']:.5f} ± "
      f"{svr_rmse_sr.describe().loc['std']:.5f}\n")


# Grid search
# Support vector regressor
full_pipeline = Pipeline([
    ('svr', SVR()),
])

param_grid = [
    {'svr__C': [0.001, 0.01, 0.1],
     'svr__epsilon': [0.001, 0.01, 0.1]},
    {'svr__C': [1, 10, 100],
     'svr__epsilon': [0.1, 1, 10]},
]

grid_search = GridSearchCV(
    full_pipeline, param_grid,
    cv=5,
    error_score='raise',
    scoring='neg_root_mean_squared_error')
grid_search.fit(X_train_prepared_df, y_train)

grid_search.best_params_

cv_res = pd.DataFrame(grid_search.cv_results_)
cv_res.sort_values(by='mean_test_score', ascending=False, inplace=True)
cv_res = cv_res[['param_svr__C', 'param_svr__epsilon', 'split0_test_score',
                 'split1_test_score', 'split2_test_score', 'mean_test_score']]

score_cols = ['split0', 'split1', 'split2', 'mean_test_rmse']
cv_res.columns = ['C', 'epsilon'] + score_cols
cv_res[score_cols] = -cv_res[score_cols].round(5).astype(np.float64)
print('Best grid search values for support vector regressor\n')
cv_res = cv_res.reset_index(drop=True)
print(cv_res)
end = time.perf_counter()
print(f"\nTotal time: {round((end - start)/60, 2)} minutes")
