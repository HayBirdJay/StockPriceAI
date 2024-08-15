import pandas as pd
import matplotlib.pyplot as plt
import argparse
from datetime import datetime

"""
Given CSV data, generate a plot of the predicted vs actual stock prices.
"""
if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-t", "--ticker", default="AAPL")
    argparser.add_argument("-m", "--model", required=True, help="model used to generate data ex: LSTM")
    argparser.add_argument("-c", "--csv", help="csv file to source data from")
    args = argparser.parse_args()

    df = pd.read_csv(f"results/csvs/{args.csv}.csv")
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(by=['date'], ascending=True)


    plt.figure(figsize=(10, 6))
    plt.plot(df['date'], df['predicted'], label='Predicted Value')
    plt.plot(df['date'], df['actual'], label='Actual Value')
    plt.title('Predicted vs Actual Values Over Time')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'results/graphs/{args.ticker}_{args.model}_{datetime.now().strftime("%I%M%S_%m_%Y")}')
    plt.close()