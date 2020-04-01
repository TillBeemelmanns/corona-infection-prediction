import os

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from datetime import date
import time
from scipy.optimize import curve_fit
from git import Repo

CSV_FILENAME = "data/time_series_covid19_confirmed_global.csv"
CSV_URL = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/' \
          'csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv'
COUNTRY = "Germany"
UPDATE_EVERY_HOUR = 0.5


def download_csv():
    df = pd.read_csv(CSV_URL)
    df.to_csv(CSV_FILENAME)


def exponential(x, a, k, b):
    return a*np.exp(x*k) + b


def main():

    df = pd.read_csv(CSV_FILENAME)

    # filter just one country
    df = df[df["Country/Region"] == COUNTRY]
    df = df.drop(columns=["Unnamed: 0", "Country/Region", "Province/State", "Lat", "Long"])
    df = df.iloc[0]  # convert to pd.Series

    # start with first infections
    df = df[df.values != 0]

    # parse to datetime
    df.index = pd.to_datetime(df.index, format='%m/%d/%y')

    # fit to exponential function
    time_in_days = np.arange(len(df.values))
    poptimal_exponential, pcovariance_exponential = curve_fit(exponential, time_in_days, df.values, p0=[0.3, 0.205, 0])

    # Plot current DATA
    fig, ax = plt.subplots(figsize=(15, 10))
    ax.plot(df.index, df.values, '*', label="Infections in Germany")
    ax.plot(df.index, exponential(time_in_days, *poptimal_exponential), 'g-', label="Exponential Fit")
    ax.set_xlabel("Date")
    ax.set_ylabel("Number of Infections")
    ax.legend()
    ax.grid()
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    fig.suptitle(date.today())
    fig.autofmt_xdate()
    fig.savefig("plots/exponential_fit.png", bbox_inches='tight')

    # Compute prediction
    prediction_in_days = 10
    time_in_days = np.arange(start=len(df.values), stop=len(df.values)+prediction_in_days)
    prediction = exponential(time_in_days, *poptimal_exponential).astype(int)
    df_prediction = pd.Series(prediction)

    # convert index to dates
    df_prediction.index = pd.date_range(df.index[-1], periods=prediction_in_days+1, closed="right")

    df_prediction = df.append(df_prediction)

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(df_prediction)

    # Plot prediction
    fig, ax = plt.subplots(figsize=(15, 10))
    ax.plot(df.index, df.values, '*', label="Infections in Germany")
    ax.plot(df_prediction.index, df_prediction.values, 'r--', label="Predicted Number of Infections")
    ax.set_xlabel("Date")
    ax.set_ylabel("Number of Infections")
    ax.legend()
    ax.grid()
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    fig.suptitle(date.today())
    fig.autofmt_xdate()
    fig.savefig("plots/exponential_extrapolation.png", bbox_inches='tight')


def git_push():
    repo_dir = '.'
    repo = Repo(repo_dir)
    file_list = [
        'main.py',
        'plots/exponential_extrapolation.png',
        'plots/exponential_fit.png'
    ]
    commit_message = 'Auto Update'
    repo.index.add(file_list)
    repo.index.commit(commit_message)
    origin = repo.remote('origin')
    origin.push()
    print("Auto Update Complete")


if __name__ == '__main__':

    while True:
        if not os.path.isfile(CSV_FILENAME):
            download_csv()
        else:
            os.remove(CSV_FILENAME)
            download_csv()

        main()
        git_push()
        time.sleep(UPDATE_EVERY_HOUR*60*60)
