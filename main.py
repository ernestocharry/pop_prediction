import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
import scipy
from scipy.optimize import curve_fit
import random
from scipy.stats.stats import pearsonr
from sklearn.svm import SVR


def fit_funtion(x, parameters):
    return np.exp(parameters[1]) * np.exp(x * parameters[0])


def fit_funtion_error(x, parameters):
    return parameters[0] * x * x + parameters[1] * x + parameters[2]


def fit_funtion_error_line(x, parameters):
    return parameters[0] * x + parameters[1]


df = pd.read_csv('data/df.csv')
df_1 = pd.read_csv('data/df_1.csv')
df_2 = pd.read_csv('data/df_2.csv')
df_3 = pd.read_csv('data/df_3.csv')
df_4 = pd.read_csv('data/df_4.csv')

df_group = pd.read_csv('data/df_group.csv', index_col = ['countryiso3code', 'date'])
df_group_1 = pd.read_csv('data/df_group_1.csv', index_col = ['countryiso3code', 'date'])
df_group_2 = pd.read_csv('data/df_group_2.csv', index_col = ['countryiso3code', 'date'])
df_group_3 = pd.read_csv('data/df_group_3.csv', index_col = ['countryiso3code', 'date'])
df_group_4 = pd.read_csv('data/df_group_4.csv', index_col = ['countryiso3code', 'date'])

df_group_min_mean_max = df_group.copy()
df_group_2_min_mean_max = df_group_2.copy()
df_group_3_min_mean_max = df_group_3.copy()
df_group_4_min_mean_max = df_group_4.copy()

df_group_min_mean_max = pd.read_csv('data/df_group_min_mean_max.csv')
df_group_2_min_mean_max = pd.read_csv('data/df_group_min_mean_max_2.csv')
df_group_3_min_mean_max = pd.read_csv('data/df_group_min_mean_max_3.csv')
df_group_4_min_mean_max = pd.read_csv('data/df_group_min_mean_max_4.csv')

df_group_min_mean_max.loc[df_group_min_mean_max['value_min'].isnull(), 'value_min'] = df_group_min_mean_max.loc[df_group_min_mean_max['value_min'].isnull(), 'value']
df_group_min_mean_max.loc[df_group_min_mean_max['value_max'].isnull(), 'value_max'] = df_group_min_mean_max.loc[df_group_min_mean_max['value_max'].isnull(), 'value']

df_group_2_min_mean_max.loc[df_group_2_min_mean_max['value_min'].isnull(), 'value_min'] = df_group_2_min_mean_max.loc[df_group_2_min_mean_max['value_min'].isnull(), 'value']
df_group_2_min_mean_max.loc[df_group_2_min_mean_max['value_max'].isnull(), 'value_max'] = df_group_2_min_mean_max.loc[df_group_2_min_mean_max['value_max'].isnull(), 'value']

df_group_3_min_mean_max.loc[df_group_3_min_mean_max['value_min'].isnull(), 'value_min'] = df_group_3_min_mean_max.loc[df_group_3_min_mean_max['value_min'].isnull(), 'value']
df_group_3_min_mean_max.loc[df_group_3_min_mean_max['value_max'].isnull(), 'value_max'] = df_group_3_min_mean_max.loc[df_group_3_min_mean_max['value_max'].isnull(), 'value']

df_group_4_min_mean_max.loc[df_group_4_min_mean_max['value_min'].isnull(), 'value_min'] = df_group_4_min_mean_max.loc[df_group_4_min_mean_max['value_min'].isnull(), 'value']
df_group_4_min_mean_max.loc[df_group_4_min_mean_max['value_max'].isnull(), 'value_max'] = df_group_4_min_mean_max.loc[df_group_4_min_mean_max['value_max'].isnull(), 'value']



df_all = df_group_2_min_mean_max.merge(df_group_3_min_mean_max, how='outer',
                            on=['countryiso3code', 'date'], suffixes=('_2', '_3')
                                  ).merge(df_group_4_min_mean_max, how='outer',
                                          on=['countryiso3code', 'date']
                                         ).merge(df_group_min_mean_max, how='outer',
                                                 on=['countryiso3code', 'date'], suffixes=('_4', '_0')
                                                )

from sklearn.neighbors import KNeighborsRegressor
from sklearn import linear_model

countries = df_all['countryiso3code'].unique()

df_all['value_KNeighbors'] = df_all['value_0'].copy()
df_all['value_KNeighbors_min'] = df_all['value_0'].copy()
df_all['value_KNeighbors_max'] = df_all['value_0'].copy()

total_countries_w_errros = []

for country in countries:
    try:
        if country not in total_countries_w_errros:
            if country != 'ERI' and country != 'KWT' and country != 'CPV' and country != 'GNB':
                X = \
                df_all[(df_all['date'] <= 2020) & (df_all['date'] >= 2015) & (df_all['countryiso3code'] == country)][
                    ['date', 'value_2', 'value_3', 'value_4']]
                Y = \
                df_all[(df_all['date'] <= 2020) & (df_all['date'] >= 2015) & (df_all['countryiso3code'] == country)][
                    'value_0']

                neigh = KNeighborsRegressor(n_neighbors=2)
                neigh = linear_model.Ridge(alpha=.5)
                neigh.fit(X.values, Y.values)

                X_to_predict = df_all[df_all['countryiso3code'] == country][['date', 'value_2', 'value_3', 'value_4']]
                Y_to_predict_fit = neigh.predict(X_to_predict.values)
                df_all.loc[df_all['countryiso3code'] == country, 'value_KNeighbors'] = Y_to_predict_fit

                X_to_predict = df_all[(df_all['countryiso3code'] == country)][
                    ['date', 'value_min_2', 'value_min_3', 'value_min_4']]
                Y_to_predict_fit = neigh.predict(X_to_predict.values)
                df_all.loc[df_all['countryiso3code'] == country, 'value_KNeighbors_min'] = Y_to_predict_fit

                X_to_predict = df_all[(df_all['countryiso3code'] == country)][
                    ['date', 'value_max_2', 'value_max_3', 'value_max_4']]
                Y_to_predict_fit = neigh.predict(X_to_predict.values)
                df_all.loc[df_all['countryiso3code'] == country, 'value_KNeighbors_max'] = Y_to_predict_fit

    except:
        print("An exception occurred with ", country)
        total_countries_w_errros.append(country)

print(total_countries_w_errros)
total_countries_w_errros.append(['CPV', 'GNB'])
print(len(total_countries_w_errros))

x_min = 2000
x_max = 2025
diff_porcentual = []

for country in countries:
    if country not in total_countries_w_errros:
        x = df_all[(df_all['date'] > x_min) & (df_all['countryiso3code'] == country)]['date']
        y_1 = df_all[(df_all['date'] > x_min) & (df_all['countryiso3code'] == country)]['value_0']
        y_1_min = df_all[(df_all['date'] > x_min) & (df_all['countryiso3code'] == country)]['value_min_0']
        y_1_max = df_all[(df_all['date'] > x_min) & (df_all['countryiso3code'] == country)]['value_max_0']

        y_2 = df_all[(df_all['date'] > x_min) & (df_all['countryiso3code'] == country)]['value_KNeighbors']
        y_2_min = df_all[(df_all['date'] > x_min) & (df_all['countryiso3code'] == country)]['value_KNeighbors_min']
        y_2_max = df_all[(df_all['date'] > x_min) & (df_all['countryiso3code'] == country)]['value_KNeighbors_max']
        # full_name_country = df[df['countryiso3code']==country]['country'].values[0]

        plt.plot(x, y_1, color='red')
        plt.plot(x, y_1_min, color='red', linestyle='dashed', alpha=0.6)
        plt.plot(x, y_1_max, color='red', linestyle='dashed', alpha=0.6)

        plt.plot(x, y_2, label='lineal model', color='blue')
        plt.plot(x, y_2_min, label='lineal model min', color='blue', linestyle='dashed', alpha=0.6)
        plt.plot(x, y_2_max, label='lineal model max', color='blue', linestyle='dashed', alpha=0.6)
        plt.legend()
        plt.show()

        if y_2.values[-1] != 0:
            if np.abs((y_2.values[-1] - y_1.values[-2]) * 100 / y_2.values[-1]) < 20:
                diff_porcentual.append((y_2.values[-1] - y_1.values[-2]) * 100 / y_2.values[-1])