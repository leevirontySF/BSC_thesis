from prophet.python import fbprophet
from prophet.python.fbprophet import models
from prophet.python.fbprophet import plot
from prophet.python.fbprophet import diagnostics

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pkg_resources
import os
from pathlib import Path
import numpy as np
import scipy
from copy import deepcopy
import tqdm

import metrics_refactored

import json
from prophet.python.fbprophet.serialize import model_to_json, model_from_json

features = pd.read_csv('data/Features data set.csv')
features['Date'] = pd.to_datetime(features['Date'], dayfirst=True)
sales = pd.read_csv('data/sales data-set.csv')
sales['Date'] = pd.to_datetime(sales['Date'], dayfirst=True)
stores = pd.read_csv('data/stores data-set.csv')

sales_weekly = sales.groupby(['Store', 'Date']).sum().reset_index().merge(stores, on='Store')[['Store', 'Date', 'Weekly_Sales', 'Type', 'Size']]

for i in features['Store'].unique():
    # interpoloi työttömyys
    store_mask = features['Store'] == i
    store_features = features[store_mask]
    cp_flags = store_features['Unemployment'].diff() != 0
    store_features.loc[cp_flags, 'u_cp'] = store_features.loc[cp_flags, 'Unemployment']
    features.loc[store_mask, 'unemployment_interpolated'] = store_features['u_cp'].interpolate().values
    for diff_name, feature_name in [
        ('temp_diff', 'Temperature'), 
        ('fuel_diff', 'Fuel_Price'), 
        ('cpi_diff', 'CPI')]:
        features.loc[store_mask, diff_name] = store_features[feature_name].diff()

data = features.merge(sales_weekly, on=['Store', 'Date'])

def weighted_mean(chunk):
    fs = chunk[['Temperature', 'Fuel_Price', 'CPI', 'unemployment_interpolated', 'Weekly_Sales', 'temp_diff', 'fuel_diff', 'cpi_diff']]
    holidays = chunk['IsHoliday']
    is_holiday = holidays.sum() > 0
    scales = chunk['Size']
    weighted_features = (fs * scales.values[:, np.newaxis]).sum() / scales.sum()
    weighted_features['IsHoliday'] = is_holiday
    return weighted_features

n_weeks = (sales_weekly['Store']==1).sum()
noise = np.random.normal(loc=0, scale=0.02, size=n_weeks)
def format_for_prophet(df):
    df = df.reset_index()
    df = df.rename(columns={'Date':'ds', 'Weekly_Sales':'y', 'Temperature':'temperature', 'Fuel_Price':'fuel_price', 'CPI':'cpi'})
    df['overfit'] = (df['y'] / df['y'].max()) + noise
    df['IsHoliday'] = df['IsHoliday'].astype(bool)
    return df.iloc[1:, :]  # drop first na

datasets = {}
for t, chunk in data.groupby('Type'):
    wm = chunk.groupby('Date').apply(weighted_mean)
    datasets[t] = format_for_prophet(wm)

def dataset_generator():  # Toimii
    out = []
    for l in 'ABC':
        out.append((f'aggregated_{l}', datasets[l]))
        for i in data[data['Type']==l]['Store'].drop_duplicates().sample(n=3):
            out.append((f'single_store_{i}_of_{l}', format_for_prophet(data[data['Store']==i])))
    return out