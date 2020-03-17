import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np


CSV_FILENAME = "data/time_series_19-covid-Confirmed.csv"
CSV_URL = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Confirmed.csv'
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

    df = df[df["Country/Region"] == COUNTRY]
    df = df.drop(columns=["Unnamed: 0", "Country/Region", "Province/State", "Lat", "Long"])

    # convert to series
    df = df.iloc[0]

    # parse to datetime
    df.index = pd.to_datetime(df.index, format='%m/%d/%y')

    print(df.index)

    #df.plot()

    #plt.show()

    time = np.arange(len(df.values))

    poptimal_exponential, pcovariance_exponential = curve_fit(exponential, time, df.values)

    print(poptimal_exponential)

    plt.plot(time, df.values)
    plt.plot(time, exponential(time, poptimal_exponential[0], poptimal_exponential[1], poptimal_exponential[2]))
    plt.show()

    print(df.values)






if __name__ == '__main__':
    main()