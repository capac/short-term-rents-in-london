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
from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
from statsmodels.formula.api import ols
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
inside_airbnb_df = pd.read_csv(
    inside_airbnb_data_file,
    keep_default_na=False, thousands=','
    )

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

len_df = inside_airbnb_df.shape[0]
print(f'Training size: '
      f'{round(len(X_train_prepared_df)/len_df, 5)}')
print(f'Validation size: '
      f'{round(len(X_val_prepared_df)/len_df, 5)}')
print(f'Testing size: '
      f'{round(len(X_test_prepared_df)/len_df, 5)}\n')

# Predictive modeling
print('Linear regression')
lr = LinearRegression()
lr.fit(X_train_prepared_df, y_train)
y_pred_lr = lr.predict(X_val_prepared_df)
lr_rmse = root_mean_squared_error(y_val, y_pred_lr)
print(f'Validation RMSE for linear regression: {round(lr_rmse, 5)}')
lr_r2_score = r2_score(y_val, y_pred_lr)
print(f'Validation R2 for linear regression: {round(lr_r2_score, 5)}\n')


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


print('Cross validation for support vector regressor')
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
X_full = inside_airbnb_df.drop(['log_price'], axis=1)
y_full = inside_airbnb_df['log_price'].copy()
full_pipeline.fit(X_full, y_full)
model_file = inside_airbnb_work_dir / 'model.pkl'
print(f'Saving model file to {model_file}')
joblib.dump(full_pipeline, model_file)
end = time.perf_counter()
print(f"Total time: {round((end - start)/60, 2)} minutes")
