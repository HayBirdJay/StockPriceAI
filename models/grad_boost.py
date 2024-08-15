import pandas as pd
import json
import numpy as np
import xgboost as xgb
from datetime import datetime
import subprocess
import argparse

def load_stock_data(stock_file):
    stock_data = pd.read_csv(stock_file)  
    stock_data['date'] = pd.to_datetime(stock_data['date'])
    return stock_data


def load_sentiment_data(sentiment_file):
    with open(sentiment_file, 'r') as file:
        sentiment_data = json.load(file)
    return sentiment_data

def split_data_by_year(stock_data, sentiment_data, seq_length):
    sequences = []
    targets = []
    dates = []

    for idx in range(len(stock_data) - seq_length):
        date_str = stock_data['date'].dt.strftime('%Y-%m-%d').iloc[idx + seq_length]
        articles = sentiment_data.get(date_str, [])
        
        if articles:
            daily_sentiment = []
            for article in articles:
                article_sentiment = [
                    float(article['article_sentiment']),
                    float(article['ticker_sentiment']),
                    float(article['average_publication_sentiment']),
                    article['amount_of_tickers_mentioned'],
                    float(article['ticker_relevance'])
                ]
                daily_sentiment.append(article_sentiment)

            if len(daily_sentiment) >= seq_length:
                sequences.append(daily_sentiment[:seq_length])
                targets.append(stock_data['close'].iloc[idx + seq_length])
                dates.append(date_str)

    sequences = np.array(sequences)
    targets = np.array(targets)

    dates = pd.to_datetime(dates)
    dates = pd.Series(dates)

    train_mask = (dates.dt.year == 2022) | (dates.dt.year == 2023)
    test_mask = dates.dt.year == 2024

    X_train, X_test = sequences[train_mask], sequences[test_mask]
    y_train, y_test = targets[train_mask], targets[test_mask]
    train_dates, test_dates = dates[train_mask], dates[test_mask]

    print("Training set:")
    print(train_dates.min(), "to", train_dates.max())
    print("Testing set:")
    print(test_dates.min(), "to", test_dates.max())

    return X_train, X_test, y_train, y_test, train_dates, test_dates

def prepare_data(combined_data):
    combined_data['next_close'] = combined_data['close'].shift(-1) 
    features = combined_data.copy()
    targets = combined_data['next_close'].dropna().values
    
    features = features.drop(columns=['date', 'next_close']).dropna().values
    return features, targets[:-1] 

def main(stock_file, sentiment_file, seq_length=20):
    stock_data = load_stock_data(stock_file)
    sentiment_data = load_sentiment_data(sentiment_file)
    
    X_train, X_test, y_train, y_test, train_dates, test_dates = split_data_by_year(stock_data, sentiment_data, seq_length)

    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)

    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=1)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    return predictions, y_test, test_dates

def save_results(predictions, actual_prices, test_dates, output_csv):
    results_df = pd.DataFrame({
        'date': test_dates,
        'predicted': predictions.flatten(),
        'actual': actual_prices.flatten()
    })
    results_df.to_csv(f"results/{output_csv}", index=False)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-t', '--ticker', required=True)
    argparser.add_argument('-o', '--outputcsv', help="csv file to save results to.", required=True)
    argparser.add_argument('-g','--graphfile',help="if set, graphs the results and saves it to the specified file name.")
    args = argparser.parse_args()

    stock_file = f'training_data/{args.ticker}_prices_csv.csv'
    sentiment_file = f'training_data/{args.ticker}_articles_formatted.json'
    predictions, actual_prices, test_dates = main(stock_file, sentiment_file)


    min_length = min(len(test_dates), len(predictions), len(actual_prices))
    test_dates = test_dates[:min_length]
    predictions = predictions[:min_length]
    actual_prices = actual_prices[:min_length]

    save_results(predictions, actual_prices, test_dates, args.outputcsv)
    print("Predictions:", predictions.flatten())
    print("Actual Prices:", actual_prices.flatten())

    if args.graphfile:
        subprocess.run(['python', 'generate_graph.py', '-t', args.ticker, '-c', args.outputcsv, '-m', 'gboost'])
