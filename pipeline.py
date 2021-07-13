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
    seed = 'http://api.worldbank.org/v2/country/all/indicator/'+indicator+'?format=json'
    response_seed = requests.get(seed).json()
    df = pd.DataFrame.from_dict(response_seed[1])

    for page in range(2, response_seed[0]['pages']+1):
        url = 'http://api.worldbank.org/v2/country/all/indicator/'+indicator+'?page='+str(page)+'&format=json'
        response = requests.get(url).json()
        df = df.append(pd.DataFrame.from_dict(response[1]), ignore_index=True)
    return df


if __name__ == "__main__":
    print('main')

