import os

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from scipy.optimize import curve_fit



CSV_FILENAME = "data/time_series_19-covid-Confirmed.csv"
CSV_URL = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/' \
          'csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Confirmed.csv'
COUNTRY = "Germany"


def download_csv():
    df = pd.read_csv(CSV_URL)
    df.to_csv(CSV_FILENAME)


def exponential(x, a, k, b):
    return a*np.exp(x*k) + b


def main():

    if not os.path.isfile(CSV_FILENAME):

        download_csv()
    else:
        print("Use fetched data\n")

    df = pd.read_csv(CSV_FILENAME)

    # filter just one country
    df = df[df["Country/Region"] == COUNTRY]

    # remove unecessary information
    df = df.drop(columns=["Unnamed: 0", "Country/Region", "Province/State", "Lat", "Long"])

    # convert to series
    df = df.iloc[0]

    # start with first infection
    df = df[df.values!=0]

    # parse to datetime
    df.index = pd.to_datetime(df.index, format='%m/%d/%y')

    time_in_days = np.arange(len(df.values))

    poptimal_exponential, pcovariance_exponential = curve_fit(exponential, time_in_days, df.values, p0=[0.3, 0.205, 0])

    fig, ax = plt.subplots()

    ax.plot(df.index, df.values, '*' ,label="Infections in Germany")
    ax.plot(df.index, exponential(time_in_days, *poptimal_exponential), 'g-', label="Exponential Fit")

    ax.set_xlabel("Date")
    ax.set_ylabel("Number of Infections")

    ax.legend()
    fig.autofmt_xdate()

    fig.savefig("plots/exponential_fit.png")


    prediction_in_days = 5

    time_in_days = np.arange(start=len(df.values), stop=len(df.values)+prediction_in_days)
    prediction = exponential(time_in_days, *poptimal_exponential).astype(int)
    df_prediction = pd.Series(prediction)
    df_prediction.index = pd.date_range(df.index[-1], periods=prediction_in_days+1, closed="right")

    df_prediction = df.append(df_prediction)

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(df_prediction)


    fig, ax = plt.subplots()

    ax.plot(df.index, df.values, '*' ,label="Infections in Germany")
    ax.plot(df_prediction.index, df_prediction.values, 'r--' ,label="Predicted Number of Infections")

    ax.set_xlabel("Date")
    ax.set_ylabel("Number of Infections")

    ax.legend()
    fig.autofmt_xdate()

    fig.savefig("plots/exponential_extrapolation.png")


if __name__ == '__main__':
    main()