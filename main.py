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
    {"country": "Germany", "min_infections": 15,
     "start_exponential_phase": 0, "end_exponential_phase": 44,
     "start_linear_phase": 44, "end_linear_phase": 60},

    {"country": "US", "min_infections": 15,
     "start_exponential_phase": 0, "end_exponential_phase": 42,
     "start_linear_phase": 42, "end_linear_phase": 75},
]


def exponential(x, a, k, b):
    return a*np.exp(x*k) + b


def sigmoidal(x, a, k, b):
    return a*(1.0 / (1+np.exp(-(x-k)))) + b


def linear_func(x, a, b):
    return a*x + b


def get_csv():
    print("Downloading New Data")
    df = pd.read_csv(CSV_URL)
    return df


def plot_model(df, country, min_infections,
               start_exponential_phase, end_exponential_phase,
               start_linear_phase, end_linear_phase):

    # filter just one country
    df = df[df["Country/Region"] == country]
    df = df.drop(columns=["Country/Region", "Province/State", "Lat", "Long"])

    df = df.iloc[0]  # convert to pd.Series

    # start with first infections
    df = df[df.values > min_infections]

    # parse to datetime
    df.index = pd.to_datetime(df.index, format='%m/%d/%y')

    # fit to exponential function
    duration_exponential_phase = end_exponential_phase - start_exponential_phase
    days_exponential_phase = np.arange(duration_exponential_phase) + start_exponential_phase
    poptimal_exponential, pcovariance_exponential = curve_fit(
        exponential, days_exponential_phase, df.values[start_exponential_phase:end_exponential_phase], p0=[1, 0.35, 0]
    )

    # compute exponential prediction
    prediction_in_days = 10
    time_in_days_extra = np.arange(
        start=start_exponential_phase, stop=duration_exponential_phase+prediction_in_days
    )
    prediction = exponential(time_in_days_extra, *poptimal_exponential).astype(int)
    df_prediction_exponential = pd.Series(prediction)

    # convert index to dates
    df_prediction_exponential.index = pd.date_range(
        start=df.index[start_exponential_phase],
        periods=duration_exponential_phase + prediction_in_days,
        closed="left"
    )

    # fit to linear function
    duration_linear_phase = end_linear_phase - start_linear_phase
    days_linear_phase = np.arange(duration_linear_phase) + start_linear_phase
    poptimal_linear, pcovariance_linear = curve_fit(
        linear_func, days_linear_phase, df.values[start_linear_phase:end_linear_phase], p0=[1, 1]
    )

    # compute linear prediction
    prediction_in_days = 20
    time_in_days_extra = np.arange(
        start=start_linear_phase, stop=end_linear_phase+prediction_in_days
    )
    prediction_linear = linear_func(time_in_days_extra, *poptimal_linear).astype(int)
    df_prediction_linear = pd.Series(prediction_linear)

    # convert index to dates
    df_prediction_linear.index = pd.date_range(
        start=df.index[start_linear_phase],
        periods=duration_linear_phase + prediction_in_days,
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
         df.index[start_exponential_phase:end_exponential_phase],
         exponential(days_exponential_phase, *poptimal_exponential),
         'g-',
         linewidth=2,
         label="Exponential Phase"
    )
    # plot exponential prediction
    ax.plot(
        df_prediction_exponential.index[duration_exponential_phase:],
        df_prediction_exponential.values[duration_exponential_phase:],
        'r--',
        label="Exponential Phase Prediction"
    )
    # plot linear phase
    ax.plot(
        df.index[start_linear_phase:end_linear_phase],
        linear_func(days_linear_phase, *poptimal_linear),
        'm-',
        linewidth=2,
        label="Linear Phase"
    )
    # plot linear prediction
    ax.plot(
        df_prediction_linear.index[duration_linear_phase:],
        df_prediction_linear.values[duration_linear_phase:],
        'm--',
        label="Linear Phase Prediction"
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Number of Infections")
    ax.legend()
    ax.grid()
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    fig.suptitle(f"{date.today()} - Number of Infected persons in {country}")
    fig.autofmt_xdate()
    fig.savefig(f"plots/model_{country}.png", bbox_inches='tight')


if __name__ == '__main__':

    if not os.path.isdir("plots"):
        os.mkdir("plots")

    df = get_csv()

    for country_info in COUNTRIES:
        plot_model(df,
                   country_info["country"],
                   country_info["min_infections"],
                   country_info["start_exponential_phase"],
                   country_info["end_exponential_phase"],
                   country_info["start_linear_phase"],
                   country_info["end_linear_phase"])
