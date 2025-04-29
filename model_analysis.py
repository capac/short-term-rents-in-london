#!/usr/bin/env python
# coding: utf-8

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import xgboost as xgb
from sklearn.base import clone
from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
from statsmodels.formula.api import ols

home_dir = Path.home()
inside_airbnb_data_dir = (
    home_dir / 'Programming/data/inside-airbnb/london'
)
inside_airbnb_work_dir = (
    home_dir /
    'Programming/Python/machine-learning-exercises/short-term-rents-in-london'
)

plots_dir = inside_airbnb_work_dir / 'plots'
plots_dir.mkdir(parents=True, exist_ok=True)
hist_dir = plots_dir / 'histograms'
hist_dir.mkdir(parents=True, exist_ok=True)

# Data preparation
inside_airbnb_data_file = (
    inside_airbnb_data_dir /
    'selected_short_term_rentals_with_distances.csv'
)
inside_airbnb_df = pd.read_csv(
    inside_airbnb_data_file,
    keep_default_na=False, thousands=','
)

inside_airbnb_df.drop(['room_type', 'nearest_station'], axis=1, inplace=True)
inside_airbnb_df['borough'] = \
    inside_airbnb_df['borough'].replace({r'\s': r'_'}, regex=True)

inside_airbnb_df = \
    inside_airbnb_df.loc[inside_airbnb_df['borough'] != 'Sutton']

inside_airbnb_df[['amenity_1', 'amenity_2', 'amenity_3']] = \
    inside_airbnb_df['amenities'].str.split(',', expand=True)
inside_airbnb_df = inside_airbnb_df.drop('amenities', axis=1)

inside_airbnb_df['log_price'] = np.log1p(inside_airbnb_df['price'])
inside_airbnb_df = inside_airbnb_df.drop('price', axis=1)

inside_airbnb_df = inside_airbnb_df.drop(['latitude', 'longitude'], axis=1)


df_full_train, df_test = train_test_split(inside_airbnb_df, test_size=0.2,
                                          random_state=33,
                                          stratify=inside_airbnb_df['borough'])
df_train, df_val = train_test_split(df_full_train, test_size=0.25,
                                    random_state=33,
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
cat_attribs = ['borough', 'property_type', 'amenity_1',
               'amenity_2', 'amenity_3']
num_attribs = ['bathrooms', 'bedrooms', 'minimum_nights',
               'crime_rate', 'distance_to_station']

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

len_df = inside_airbnb_df.shape[0]
print(f'Training size: {round(len(X_train_prepared_df)/len_df, 5):>10}')
print(f'Validation size: {round(len(X_val_prepared_df)/len_df, 5):>8}')
print(f'Testing size: {round(len(X_test_prepared_df)/len_df, 5):>11}\n')


# Predictive modeling
print('Linear regression')
lr = LinearRegression()
lr.fit(X_train_prepared_df, y_train)
y_pred_lr = lr.predict(X_val_prepared_df)
lr_rmse = root_mean_squared_error(y_val, y_pred_lr)
print(f'Validation RMSE for linear regression: {round(lr_rmse, 5)}')
lr_r2_score = r2_score(y_val, y_pred_lr)
print(f'Validation R2 for linear regression: {round(lr_r2_score, 5)}\n')


print('Decision tree regressor')
dtr = DecisionTreeRegressor(random_state=42)
dtr.fit(X_train_prepared_df, y_train)
y_pred_dtr = dtr.predict(X_val_prepared_df)
dtr_rmse = root_mean_squared_error(y_val, y_pred_dtr)
print(f'Validation RMSE for decision tree: {round(dtr_rmse, 5)}')
dtr_r2_score = r2_score(y_val, y_pred_dtr)
print(f'Validation R2 for linear regression: {round(dtr_r2_score, 5)}\n')


print('Random forest regressor')
rfr = RandomForestRegressor(random_state=42)
rfr.fit(X_train_prepared_df, y_train)
y_pred_rfr = rfr.predict(X_val_prepared_df)
rfr_rmse = root_mean_squared_error(y_val, y_pred_rfr)
print(f'Validation RMSE for random forest regressor: {round(rfr_rmse, 5)}')
rfr_r2_score = r2_score(y_val, y_pred_rfr)
print(f'Validation R2 for random forest regressor: {round(rfr_r2_score, 5)}\n')


print('Stocastic gradient descent regressor')
sgdr = SGDRegressor(random_state=42)
sgdr.fit(X_train_prepared_df, y_train)
y_pred_sgdr = sgdr.predict(X_val_prepared_df)
sgdr_rmse = root_mean_squared_error(y_val, y_pred_sgdr)
print(f'Validation RMSE for stocastic gradient descent: '
      f'{round(sgdr_rmse, 5)}')
sgdr_r2_score = r2_score(y_val, y_pred_sgdr)
print(f'Validation R2 for stocastic gradient descent: '
      f'{round(sgdr_r2_score, 5)}\n')


print('Support vector regressor')
svr = SVR()
svr.fit(X_train_prepared_df, y_train)
y_pred_svr = svr.predict(X_val_prepared_df)
svr_rmse = root_mean_squared_error(y_val, y_pred_svr)
print(f'Validation RMSE for support vector regressor: '
      f'{round(svr_rmse, 5)}')
svr_r2_score = r2_score(y_val, y_pred_svr)
print(f'Validation R2 for support vector regressor: '
      f'{round(svr_r2_score, 5)}\n')


print('XGBoost regressor')
reg = xgb.XGBRegressor(
    tree_method="hist",
    eval_metric=root_mean_squared_error,
    max_depth=10,
    n_estimators=100,
    verbosity=0,
)
reg.fit(
    X_train_prepared_df, y_train, verbose=False,
    eval_set=[(X_train_prepared_df, y_train)]
)
y_pred_reg = reg.predict(X_val_prepared_df)
reg_rmse = root_mean_squared_error(y_val, y_pred_reg)
print(f'Validation RMSE for XGBoost regressor: {round(reg_rmse, 5)}')
reg_r2_score = r2_score(y_val, y_pred_reg)
print(f'Validation R2 for XGBoost regressor: {round(reg_r2_score, 5)}\n\n')


# Cross validation
print('Cross validation for linear regression')
cloned_lr = clone(lr)
lr_rmses = -cross_val_score(
    cloned_lr, X_train_prepared_df, y_train,
    scoring='neg_root_mean_squared_error',
    cv=10,
)
lr_cv_sr = pd.Series(lr_rmses).describe()
print(f"Cross-validation RMSE mean and std dev: {lr_cv_sr.loc['mean']:.5f} ± "
      f"{lr_cv_sr.loc['std']:.5f}\n")


print('Cross validation for decision tree regressor')
cloned_dtr = clone(dtr)
dtr_rmses = -cross_val_score(
    cloned_dtr, X_train_prepared_df, y_train,
    scoring='neg_root_mean_squared_error',
    cv=10,
)
dtr_cv_sr = pd.Series(dtr_rmses).describe()
print(f"Cross-validation RMSE mean and std dev: {dtr_cv_sr.loc['mean']:.5f} ± "
      f"{dtr_cv_sr.loc['std']:.5f}\n")


print('Cross validation for random forest regressor')
cloned_rfr = clone(rfr)
rfr_rmses = -cross_val_score(
    cloned_rfr, X_train_prepared_df, y_train,
    scoring='neg_root_mean_squared_error',
    cv=10,
)
rfr_cv_sr = pd.Series(rfr_rmses).describe()
print(f"Cross-validation RMSE mean and std dev: "
      f"{rfr_cv_sr.loc['mean']:.5f} ± "
      f"{rfr_cv_sr.loc['std']:.5f}\n")


print('Cross validation for stocastic gradient descent regressor')
cloned_sgdr = clone(sgdr)
sgdr_rmses = -cross_val_score(
    cloned_sgdr, X_train_prepared_df, y_train,
    scoring='neg_root_mean_squared_error',
    cv=10,
)
sgdr_cv_sr = pd.Series(sgdr_rmses).describe()
print(f"Cross-validation RMSE mean and std dev: "
      f"{sgdr_cv_sr.loc['mean']:.5f} ± "
      f"{sgdr_cv_sr.loc['std']:.5f}\n")


print('Cross validation for support vector machine regressor')
cloned_svr = clone(svr)
svr_rmses = -cross_val_score(
    cloned_svr, X_train_prepared_df, y_train,
    scoring='neg_root_mean_squared_error',
    cv=10,
)
svr_cv_sr = pd.Series(svr_rmses).describe()
print(f"Cross-validation RMSE mean and std dev: {svr_cv_sr.loc['mean']:.5f} ± "
      f"{svr_cv_sr.loc['std']:.5f}\n")


print('Cross validation for XGBoost regressor')
cloned_reg = clone(reg)
reg_rmses = -cross_val_score(
    cloned_reg, X_train_prepared_df, y_train,
    scoring='neg_root_mean_squared_error',
    cv=10,
)
reg_cv_sr = pd.Series(reg_rmses).describe()
print(f"Cross-validation RMSE mean and std dev: {reg_cv_sr.loc['mean']:.5f} ± "
      f"{reg_cv_sr.loc['std']:.5f}\n\n")


# Results for test data set
print('Support vector regressor using test dataset')
y_pred_svr = svr.predict(X_test_prepared_df)
svr_rmse = root_mean_squared_error(y_test, y_pred_svr)
print(f'Test RMSE for support vector regressor: {round(svr_rmse, 5)}')
svr_r2_score = r2_score(y_test, y_pred_svr)
print(f'Test R2 for support vector regressor: {round(svr_r2_score, 5)}\n\n')


# OLS regression results from statsmodels
lm = ols(
    'log_price ~ property_type + crime_rate + bedrooms + minimum_nights '
    '+ borough + distance_to_station + bathrooms + amenity_1 + amenity_3 '
    '+ amenity_2', data=inside_airbnb_df
    )
model = lm.fit()
print(model.summary().tables[0])
