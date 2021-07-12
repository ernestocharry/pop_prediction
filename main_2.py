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


def fit_funtion(x_1, parameters_1):
    return np.exp(parameters_1[1]) * np.exp(x_1 * parameters_1[0])


def fit_funtion_error(x_2, parameters_2):
    return parameters_2[0] * x_2 * x_2 + parameters_2[1] * x_2 + parameters_2[2]


def fit_funtion_error_line(x_3, parameters_3):
    return parameters_3[0] * x_3 + parameters_3[1]


# Model validation: prediction to populations in the years 2015 - 2020

df = pd.read_csv('data/df.csv')  # Data download from API
df_1 = pd.read_csv('data/df_1.csv')
df_2 = pd.read_csv('data/df_2.csv')
df_3 = pd.read_csv('data/df_3.csv')
df_4 = pd.read_csv('data/df_4.csv')

df_group = pd.read_csv('data/df_group.csv', index_col=['countryiso3code', 'date'])  # df in group_by country
df_group_1 = pd.read_csv('data/df_group_1.csv', index_col=['countryiso3code', 'date'])
df_group_2 = pd.read_csv('data/df_group_2.csv', index_col=['countryiso3code', 'date'])
df_group_3 = pd.read_csv('data/df_group_3.csv', index_col=['countryiso3code', 'date'])
df_group_4 = pd.read_csv('data/df_group_4.csv', index_col=['countryiso3code', 'date'])

# df_group_min_mean_max = pd.read_csv('data/df_group_min_mean_max.csv')  # df with min, mean, max values for 2020 - 2025
df_group_min_mean_max = df_group.copy()
df_group_2_min_mean_max = pd.read_csv('data/df_group_min_mean_max_2.csv')
df_group_3_min_mean_max = pd.read_csv('data/df_group_min_mean_max_3.csv')
df_group_4_min_mean_max = pd.read_csv('data/df_group_min_mean_max_4.csv')

total_countries_w_errros = []

countries = df_group.index.get_level_values(0).unique()

# Smooth Model
for country in countries[0:5]:
    if country not in total_countries_w_errros:
        full_name_country = df[df['countryiso3code'] == country]['country'].values[0]

        try:
            x = df_group[df_group['value'] != 0].loc[country].index.tolist().copy()
            y = df_group[df_group['value'] != 0].loc[country].values.tolist().copy()

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

            for j in range(len(x) - 10):  # Do using the data for dates > 2015
                x1 = [x[i] for i in range(j, j + 4)]
                y1 = [y[i] for i in range(j, j + 4)]
                parameters, res, _, _, _ = np.polyfit(x1, np.log(y1), 1, full=True)
                if len(delta_y_adjustment) >= 2:
                    adjustment = np.mean(delta_y_adjustment[-2:])
                elif len(delta_y_adjustment) == 1:
                    adjustment = delta_y_adjustment[0]
                else:
                    adjustment = 0.1

                y_predict = fit_funtion(x[j + 4], parameters) - adjustment
                x_adjustment.append(x[j + 4])
                y_adjustment.append(y_predict)
                delta_y_adjustment.append(y_predict - y[j + 4])
                delta_y_adjustment_normal.append(((y_predict - y[j + 4]) * 100 / y_predict)[0])

            x_error_predict_extra = [float(i) for i in range(int(x_adjustment[-1] + 1), 2026)]
            MonteCarlo = pd.DataFrame([], columns=x_error_predict_extra)

            total_range = [i for i in range(-20, -1)]

            for monteCarlo in range(100):
                random_index = np.sort(np.unique(random.sample(total_range, 6) + [-1, -2]))

                x_error = [x_adjustment[i] for i in random_index]
                y_error = [delta_y_adjustment_normal[i] for i in random_index]
                parameters_error, res, _, _, _ = np.polyfit(x_error, y_error, 2, full=True)

                y_error_predict = [fit_funtion_error(i, parameters_error) for i in x_error]
                x_error_predict_extra = [float(i) for i in range(int(x_error[-1:][0] + 1), 2026)]
                print(x_error_predict_extra)
                y_error_predict_extra = [fit_funtion_error(i, parameters_error) for i in x_error_predict_extra]

                if np.abs(y_error_predict_extra[-1]) > 1.5:
                    x_error = [x_adjustment[i] for i in random_index]
                    y_error = [delta_y_adjustment_normal[i] for i in random_index]
                    parameters_error, res, _, _, _ = np.polyfit(x_error, y_error, 1, full=True)
                    y_error_predict = [fit_funtion_error_line(i, parameters_error) for i in x_error]
                    x_error_predict_extra = [float(i) for i in range(int(x_error[-1:][0] + 1), 2026)]
                    y_error_predict_extra = [fit_funtion_error_line(i, parameters_error) for i in x_error_predict_extra]

                MonteCarlo = MonteCarlo.append(
                    pd.DataFrame(np.array(y_error_predict_extra).reshape(-1, len(y_error_predict_extra)),
                                 columns=x_error_predict_extra))
                plt.scatter(x_error_predict_extra, y_error_predict_extra, color='red', alpha=0.004)

            plt.plot(x_adjustment, delta_y_adjustment_normal, label=country + ' : ' + full_name_country)
            plt.plot(x_error, y_error_predict, color='red')
            plt.hlines(-2, x_adjustment[0], x_error_predict_extra[-1], color='purple')
            plt.hlines(2, x_adjustment[0], x_error_predict_extra[-1], color='purple')
            plt.hlines(-1.5, x_adjustment[0], x_error_predict_extra[-1], color='green')
            plt.hlines(1.5, x_adjustment[0], x_error_predict_extra[-1], color='green')
            # plt.show()

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
                df_group_min_mean_max.loc[(country, x_new), 'value_mean'] = y_real
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

            plt.plot(x, y, label=country + ' : ' + full_name_country, color='red')
            plt.plot(x_min, y_min, label=country + ' : ' + full_name_country, color='red', linestyle='dashed',
                     alpha=0.6)
            plt.plot(x_max, y_max, label=country + ' : ' + full_name_country, color='red', linestyle='dashed',
                     alpha=0.6)
            plt.plot(x_adjustment, y_adjustment, label=country + ' : ' + full_name_country)
            plt.legend()
            # plt.show()

        except:
            print("An exception occurred with ", country)
            total_countries_w_errros.append(country)

df_group_min_mean_max.loc[df_group_min_mean_max['value_min'].isnull(), 'value_min'] = df_group_min_mean_max.loc[
    df_group_min_mean_max['value_min'].isnull(), 'value']
df_group_min_mean_max.loc[df_group_min_mean_max['value_max'].isnull(), 'value_max'] = df_group_min_mean_max.loc[
    df_group_min_mean_max['value_max'].isnull(), 'value']

df_group_min_mean_max.loc[df_group_min_mean_max['value_mean'].isnull(), 'value_mean'] = df_group_min_mean_max.loc[
    df_group_min_mean_max['value_mean'].isnull(), 'value']
df_group_min_mean_max.to_csv('model_validation/smooth.csv')

# --- ML Model
df_group_min_mean_max = pd.read_csv('data/df_group_min_mean_max.csv')  # df with min, mean, max values for 2020 - 2025
df_group_2_min_mean_max = pd.read_csv('data/df_group_min_mean_max_2.csv')
df_group_3_min_mean_max = pd.read_csv('data/df_group_min_mean_max_3.csv')
df_group_4_min_mean_max = pd.read_csv('data/df_group_min_mean_max_4.csv')

df_group_min_mean_max.loc[df_group_min_mean_max['value_min'].isnull(), 'value_min'] = df_group_min_mean_max.loc[
    df_group_min_mean_max['value_min'].isnull(), 'value']
df_group_min_mean_max.loc[df_group_min_mean_max['value_max'].isnull(), 'value_max'] = df_group_min_mean_max.loc[
    df_group_min_mean_max['value_max'].isnull(), 'value']

df_group_2_min_mean_max.loc[df_group_2_min_mean_max['value_min'].isnull(), 'value_min'] = df_group_2_min_mean_max.loc[
    df_group_2_min_mean_max['value_min'].isnull(), 'value']
df_group_2_min_mean_max.loc[df_group_2_min_mean_max['value_max'].isnull(), 'value_max'] = df_group_2_min_mean_max.loc[
    df_group_2_min_mean_max['value_max'].isnull(), 'value']

df_group_3_min_mean_max.loc[df_group_3_min_mean_max['value_min'].isnull(), 'value_min'] = df_group_3_min_mean_max.loc[
    df_group_3_min_mean_max['value_min'].isnull(), 'value']
df_group_3_min_mean_max.loc[df_group_3_min_mean_max['value_max'].isnull(), 'value_max'] = df_group_3_min_mean_max.loc[
    df_group_3_min_mean_max['value_max'].isnull(), 'value']

df_group_4_min_mean_max.loc[df_group_4_min_mean_max['value_min'].isnull(), 'value_min'] = df_group_4_min_mean_max.loc[
    df_group_4_min_mean_max['value_min'].isnull(), 'value']
df_group_4_min_mean_max.loc[df_group_4_min_mean_max['value_max'].isnull(), 'value_max'] = df_group_4_min_mean_max.loc[
    df_group_4_min_mean_max['value_max'].isnull(), 'value']

df_all = df_group_2_min_mean_max.merge(df_group_3_min_mean_max, how='outer',
                                       on=['countryiso3code', 'date'], suffixes=('_2', '_3')
                                       ).merge(df_group_4_min_mean_max, how='outer',
                                               on=['countryiso3code', 'date']
                                               ).merge(df_group_min_mean_max, how='outer',
                                                       on=['countryiso3code', 'date'], suffixes=('_4', '_0')
                                                       )

countries = df_all['countryiso3code'].unique()

df_all['value_KNeighbors'] = df_all['value_0'].copy()
df_all['value_KNeighbors_min'] = df_all['value_0'].copy()
df_all['value_KNeighbors_max'] = df_all['value_0'].copy()

total_countries_w_errros = []
# Lineal Machine Lerning Model
for country in countries:
    try:
        if country not in total_countries_w_errros:
            if country != 'ERI' and country != 'KWT' and country != 'CPV' and country != 'GNB':
                X = \
                    df_all[
                        (df_all['date'] <= 2015) & (df_all['date'] >= 2010) & (df_all['countryiso3code'] == country)][
                        ['date', 'value_2', 'value_3', 'value_4']]
                Y = \
                    df_all[
                        (df_all['date'] <= 2015) & (df_all['date'] >= 2010) & (df_all['countryiso3code'] == country)][
                        'value_0']

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

df_all.to_csv('model_validation/ml_lineal.csv')
