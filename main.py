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
    ax.plot(df.index, exponential(time_in_days, *poptimal_exponential), 'r-', label="Exponential Fit")

    ax.set_xlabel("Days")
    ax.set_ylabel("Number of Infections")

    ax.legend()
    fig.autofmt_xdate()

    fig.savefig("plots/exponential_fit.png")

    plt.show()



if __name__ == '__main__':
    main()