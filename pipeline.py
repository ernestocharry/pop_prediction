# Pipe line to predict the population per country up to 2024 using Machine Learning lineal model Ridge
# 1) Download the data for each feature
# 2) Extrapolar the features up to 2024
# 3) Predict using ML model

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
import scipy
from scipy.optimize import curve_fit
import random
from scipy.stats.stats import pearsonr
from sklearn.svm import SVR


def download_indicator(indicator):
    # Given a indicator, download ALL the data and save it into DataFrame df
    seed = 'http://api.worldbank.org/v2/country/all/indicator/' + indicator + '?format=json'
    response_seed = requests.get(seed).json()
    df = pd.DataFrame.from_dict(response_seed[1])

    for page in range(2, response_seed[0]['pages'] + 1):
        url = 'http://api.worldbank.org/v2/country/all/indicator/' + indicator + '?page=' + str(page) + '&format=json'
        response = requests.get(url).json()
        df = df.append(pd.DataFrame.from_dict(response[1]), ignore_index=True)
    return df


def fit_funtion(x, parameters):
    return np.exp(parameters[1]) * np.exp(x * parameters[0])


def fit_funtion_error(x, parameters):
    return parameters[0] * x * x + parameters[1] * x + parameters[2]


def fit_funtion_error_line(x, parameters):
    return parameters[0] * x + parameters[1]


def extrapolation_with_exponential_monteCarlo(df_group):
    for country in countries[0:25]:
        if country not in total_countries_w_errros:
            full_name_country = df[df['countryiso3code'] == country]['country'].values[0]

            try:
                x = df_group[df_group['value'] != 0].loc[country].index.tolist()
                y = df_group[df_group['value'] != 0].loc[country].values.tolist()

                x_min = x.copy()
                x_max = x.copy()
                y_min = y.copy()
                y_max = y.copy()

                if all(v[0] == 0.0 for v in y):
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
                    delta_y_adjustment_normal.append(((y_predict - y[j + 4]) * 100 / y_predict)[0])

                # predict x from the last years to 2026
                x_error_predict_extra = [float(i) for i in range(int(x_adjustment[-1] + 1), 2026)]
                MonteCarlo = pd.DataFrame([], columns=x_error_predict_extra)

                total_range = [i for i in range(-20, -1)]

                for monteCarlo in range(100):  # 10^2 errors

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
                print("An exception occurred with ", country)
                total_countries_w_errros.append(country)


if __name__ == "__main__":
    print('main')
