import os

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from datetime import date
from scipy.optimize import curve_fit

CSV_URL = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/' \
          'csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv'

COUNTRIES = [
    {"country": "Germany", "min_infections": 15, "duration_exponential_phase": 45},
    {"country": "US", "min_infections": 15, "duration_exponential_phase": 45},
]

def exponential(x, a, k, b):
    return a*np.exp(x*k) + b


def sigmoidal(x, a, k, b):
    return a*(1.0 / (1+np.exp(-(x-k)))) + b


def get_csv():
    print("Downloading New Data")
    df = pd.read_csv(CSV_URL)
    return df


def plot_exponential(df, country, min_infections, duration_exponential_phase):

    # filter just one country
    df = df[df["Country/Region"] == country]
    df = df.drop(columns=["Country/Region", "Province/State", "Lat", "Long"])

    df = df.iloc[0]  # convert to pd.Series

    # start with first infections
    df = df[df.values > min_infections]

    # parse to datetime
    df.index = pd.to_datetime(df.index, format='%m/%d/%y')

    # fit to exponential function
    time_in_days = np.arange(duration_exponential_phase)
    poptimal_exponential, pcovariance_exponential = curve_fit(
        exponential, time_in_days, df.values[:duration_exponential_phase], p0=[1, 0.35, 0]
    )

    # Compute prediction
    prediction_in_days = 10
    time_in_days_extra = np.arange(
        start=0, stop=duration_exponential_phase+prediction_in_days
    )
    prediction = exponential(time_in_days_extra, *poptimal_exponential).astype(int)
    df_prediction = pd.Series(prediction)

    # convert index to dates
    df_prediction.index = pd.date_range(
        start=df.index[0],
        periods=duration_exponential_phase + prediction_in_days,
        closed="left"
    )

    fig, ax = plt.subplots(figsize=(15, 10))
    # plot real data
    ax.plot(
        df.index,
        df.values,
        '*',
        color="blue",
        markersize=5,
        label=f"Infections in {country}")
    # plot exponential phase
    ax.plot(
         df.index[:duration_exponential_phase],
         exponential(time_in_days, *poptimal_exponential),
         'g-',
         linewidth=2,
         label="Exponential Phase"
    )
    # plot prediction
    ax.plot(
        df_prediction.index[duration_exponential_phase:],
        df_prediction.values[duration_exponential_phase:],
        'r--',
        label="Predicted Number of Infections"
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Number of Infections")
    ax.legend()
    ax.grid()
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    fig.suptitle(f"{date.today()} - Number of Infected persons in {country}")
    fig.autofmt_xdate()
    fig.savefig(f"plots/exponential_fit_{country}.png", bbox_inches='tight')


if __name__ == '__main__':

    if not os.path.isdir("plots"):
        os.mkdir("plots")

    df = get_csv()

    for country_info in COUNTRIES:
        plot_exponential(df,
                         country_info["country"],
                         country_info["min_infections"],
                         country_info["duration_exponential_phase"])
