# Pipe line to predict the population per country up to 2024 using Machine Learning lineal model Ridge
# 1) Download the data for each feature
# 2) Extrapolar the features up to 2024
# 3) Predict using ML model
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
import scipy
from scipy.optimize import curve_fit
import random
from scipy.stats.stats import pearsonr
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn import linear_model


def download_indicator(indicator):
    # Given a indicator, download ALL the data and save it into DataFrame df
    seed = 'http://api.worldbank.org/v2/country/all/indicator/' + indicator + '?format=json'
    response_seed = requests.get(seed).json()
    df = pd.DataFrame.from_dict(response_seed[1])

    for page in range(2, response_seed[0]['pages'] + 1):
        url = 'http://api.worldbank.org/v2/country/all/indicator/' + indicator + '?page=' + str(page) + '&format=json'
        response = requests.get(url).json()
        df = df.append(pd.DataFrame.from_dict(response[1]), ignore_index=True)
        print(page)
    return df


def fit_funtion(x, parameters):
    return np.exp(parameters[1]) * np.exp(x * parameters[0])


def fit_funtion_error(x, parameters):
    return parameters[0] * x * x + parameters[1] * x + parameters[2]


def fit_funtion_error_line(x, parameters):
    return parameters[0] * x + parameters[1]


def extrapolation_with_exponential_monteCarlo(df, total_countries_w_errros):
    df['date'] = df['date'].astype(float)
    df['value_min'] = df['value'].copy()
    df['value_max'] = df['value'].copy()

    df_group = df[df['date'] >= 1990].groupby(['countryiso3code', 'date'])[
        ['value', 'value_min', 'value_max']].sum().copy()
    df_group_min_mean_max = df_group.copy()
    df_group_min_mean_max['value_min'] = df_group_min_mean_max['value'].copy()
    df_group_min_mean_max['value_max'] = df_group_min_mean_max['value'].copy()

    countries = df_group.index.get_level_values(0).unique()
    for country in countries:
        if country not in total_countries_w_errros:
            try:
                x = df_group[df_group['value'] != 0]['value'].loc[country].index.tolist()
                y = df_group[df_group['value'] != 0]['value'].loc[country].values.tolist()

                x_min = x.copy()
                x_max = x.copy()
                y_min = y.copy()
                y_max = y.copy()

                if all(v == 0.0 for v in y):
                    continue

                x_adjustment = []
                y_adjustment = []
                delta_y_adjustment = []
                delta_y_adjustment_normal = []

                for j in range(len(x) - 4):  # Extrapolation using 4 years
                    x1 = [x[i] for i in range(j, j + 4)]
                    y1 = [y[i] for i in range(j, j + 4)]
                    parameters, res, _, _, _ = np.polyfit(x1, np.log(y1), 1, full=True)

                    if len(delta_y_adjustment) >= 2:
                        adjustment = np.mean(delta_y_adjustment[-2:])  # mean error using the last 2 years
                    elif len(delta_y_adjustment) == 1:
                        adjustment = delta_y_adjustment[0]
                    else:
                        adjustment = 0.1

                    y_predict = fit_funtion(x[j + 4], parameters) - adjustment

                    x_adjustment.append(x[j + 4])
                    y_adjustment.append(y_predict)
                    delta_y_adjustment.append(y_predict - y[j + 4])
                    delta_y_adjustment_normal.append(((y_predict - y[j + 4]) * 100 / y_predict))

                # predict x from the last years to 2026
                x_error_predict_extra = [float(i) for i in range(int(x_adjustment[-1] + 1), 2026)]
                MonteCarlo = pd.DataFrame([], columns=x_error_predict_extra)

                total_range = [i for i in range(-20, -1)]
                for monteCarlo in range(nomontecarlo):  # 10^2 errors, nomontecarlo is a input value

                    # Random select 8 years, include the last two years
                    random_index = np.sort(np.unique(random.sample(total_range, 6) + [-1, -2]))

                    x_error = [x_adjustment[i] for i in random_index]
                    y_error = [delta_y_adjustment_normal[i] for i in random_index]
                    parameters_error, res, _, _, _ = np.polyfit(x_error, y_error, 2, full=True)

                    y_error_predict = [fit_funtion_error(i, parameters_error) for i in x_error]
                    x_error_predict_extra = [float(i) for i in range(int(x_error[-1:][0] + 1), 2026)]
                    # Predict the error using cuadratic function
                    y_error_predict_extra = [fit_funtion_error(i, parameters_error) for i in x_error_predict_extra]

                    if np.abs(y_error_predict_extra[-1]) > 1.5:
                        # If the error es biger, predict the error using line
                        x_error = [x_adjustment[i] for i in random_index]
                        y_error = [delta_y_adjustment_normal[i] for i in random_index]
                        parameters_error, res, _, _, _ = np.polyfit(x_error, y_error, 1, full=True)
                        y_error_predict = [fit_funtion_error_line(i, parameters_error) for i in x_error]
                        x_error_predict_extra = [float(i) for i in range(int(x_error[-1:][0] + 1), 2026)]
                        y_error_predict_extra = [fit_funtion_error_line(i, parameters_error) for i in
                                                 x_error_predict_extra]

                    MonteCarlo = MonteCarlo.append(
                        pd.DataFrame(np.array(y_error_predict_extra).reshape(-1, len(y_error_predict_extra)),
                                     columns=x_error_predict_extra))

                delta_y_adjustment_max = delta_y_adjustment.copy()
                delta_y_adjustment_min = delta_y_adjustment.copy()
                parameters_min = parameters.copy()
                parameters_max = parameters.copy()

                for j, x_new in enumerate(x_error_predict_extra):

                    MonteCarloMin = MonteCarlo[x_new].min()

                    adjustment_min = np.mean(delta_y_adjustment_min[-2:])
                    y_predict_min = fit_funtion(x_new, parameters_min) - adjustment_min
                    y_real_min = (1 - MonteCarloMin / 100) * y_predict_min
                    delta_y_adjustment_min.append(y_predict_min - y_real_min)
                    x_min.append(x_new)
                    y_min.append(y_real_min)
                    df_group_min_mean_max.loc[(country, x_new), 'value_min'] = y_real_min
                    x1_min = x_min[-4:]
                    y1_min = y_min[-4:]
                    parameters_min, res, _, _, _ = np.polyfit(x1_min, np.log(y1_min), 1, full=True)

                    MonteCarloMean = MonteCarlo[x_new].mean()

                    adjustment = np.mean(delta_y_adjustment[-2:])
                    y_predict = fit_funtion(x_new, parameters) - adjustment
                    y_real = (1 - MonteCarloMean / 100) * y_predict
                    delta_y_adjustment.append(y_predict - y_real)

                    x.append(x_new)
                    y.append(y_real)
                    df_group.loc[(country, x_new), 'value'] = y_real
                    df_group_min_mean_max.loc[(country, x_new), 'value'] = y_real

                    x1 = x[-4:]
                    y1 = y[-4:]
                    parameters, res, _, _, _ = np.polyfit(x1, np.log(y1), 1, full=True)

                    MonteCarloMax = MonteCarlo[x_new].max()

                    adjustment_max = np.mean(delta_y_adjustment_max[-2:])
                    y_predict_max = fit_funtion(x_new, parameters_max) - adjustment_max
                    y_real_max = (1 - MonteCarloMax / 100) * y_predict_max
                    delta_y_adjustment_max.append(y_predict_max - y_real_max)
                    x_max.append(x_new)
                    y_max.append(y_real_max)
                    df_group_min_mean_max.loc[(country, x_new), 'value_max'] = y_real_max
                    x1_max = x_max[-4:]
                    y1_max = y_max[-4:]
                    parameters_max, res, _, _, _ = np.polyfit(x1_max, np.log(y1_max), 1, full=True)

            except:
                # print("An exception occurred with ", country)
                total_countries_w_errros.append(country)
    df_group_min_mean_max.loc[df_group_min_mean_max['value_min'].isnull(), 'value_min'] = \
        df_group_min_mean_max.loc[df_group_min_mean_max['value_min'].isnull(), 'value']
    return df_group_min_mean_max, countries_w_errros


def ml_model(df_all, countries_w_errros):
    df_all['value_KNeighbors'] = df_all['value_0'].copy()
    df_all['value_KNeighbors_min'] = df_all['value_0'].copy()
    df_all['value_KNeighbors_max'] = df_all['value_0'].copy()

    df_all.reset_index(inplace=True)
    countries = df_all['countryiso3code'].unique()

    for country in countries:
        try:
            if country not in countries_w_errros:
                X = df_all[
                    (df_all['date'] <= 2020) & (df_all['date'] >= 2015) & (df_all['countryiso3code'] == country)][
                    ['date', 'value_2', 'value_3', 'value_4']]
                Y = df_all[
                    (df_all['date'] <= 2020) & (df_all['date'] >= 2015) & (df_all['countryiso3code'] == country)][
                    'value_0']

                neigh = linear_model.Ridge(alpha=.5)
                neigh.fit(X.values, Y.values)

                X_to_predict = df_all[df_all['countryiso3code'] == country][
                    ['date', 'value_2', 'value_3', 'value_4']]
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
            countries_w_errros.append(country)
    return df_all


if __name__ == "__main__":

    for i in sys.argv[1:]:
        if i.startswith('load='):
            load = i[i.find('=') + 1:]
        if i.startswith('montecarlo='):
            nomontecarlo = int(float(i[i.find('=') + 1:]))

    if "load" not in globals():
        load = False
    if 'nomontecarlo' not in globals():
        nomontecarlo = 100

    if load:
        df = pd.read_csv('data/df.csv')
        df_2 = pd.read_csv('data/df_2.csv')
        df_3 = pd.read_csv('data/df_3.csv')
        df_4 = pd.read_csv('data/df_4.csv')
    else:
        df = download_indicator('SP.POP.TOTL')
        df_1 = download_indicator('SM.POP.NETM')
        df_2 = download_indicator('SP.DYN.AMRT.MA')
        df_3 = download_indicator('SP.DYN.AMRT.FE')
        df_4 = download_indicator('SP.DYN.TFRT.IN')

    countries_w_errros = []
    df_group_min_mean_max, countries_w_errros = extrapolation_with_exponential_monteCarlo(df, countries_w_errros)
    df_group_2_min_mean_max, countries_w_errros = extrapolation_with_exponential_monteCarlo(df_2, countries_w_errros)
    df_group_3_min_mean_max, countries_w_errros = extrapolation_with_exponential_monteCarlo(df_3, countries_w_errros)
    df_group_4_min_mean_max, countries_w_errros = extrapolation_with_exponential_monteCarlo(df_4, countries_w_errros)

    countries_w_errros.append(['CPV', 'GNB'])

    df_all = df_group_2_min_mean_max.merge(df_group_3_min_mean_max, how='outer',
                                           on=['countryiso3code', 'date'], suffixes=('_2', '_3')
                                           ).merge(df_group_4_min_mean_max, how='outer',
                                                   on=['countryiso3code', 'date']
                                                   ).merge(df_group_min_mean_max, how='outer',
                                                           on=['countryiso3code', 'date'], suffixes=('_4', '_0')
                                                           )

    df_all = ml_model(df_all, countries_w_errros)
    df_all.to_csv('results_predictions_pipeline.csv')
    print('Total countries with errors: ', len(countries_w_errros))
