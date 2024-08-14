import pandas as pd
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import xgboost as xgb
import matplotlib.pyplot as plt
from datetime import datetime

ticker = 'AAPL'
model_type = 'GBOOST'

# Load stock price data
def load_stock_data(stock_file):
    stock_data = pd.read_csv(stock_file)  # Assuming CSV with 'date' and 'close' columns
    stock_data['date'] = pd.to_datetime(stock_data['date'])
    return stock_data

# Load sentiment data from JSON
def load_sentiment_data(sentiment_file):
    with open(sentiment_file, 'r') as file:
        sentiment_data = json.load(file)
    return sentiment_data

# Combine stock and sentiment data
def combine_data(stock_data, sentiment_data, seq_length):
    combined_data = stock_data.copy()
    daily_sentiment = []
    dates = []

    # Process each date
    for date_str in combined_data['date'].dt.strftime('%Y-%m-%d'):
        articles = sentiment_data.get(date_str, [])
        if articles:
            # Prepare input for the article sentiment model
            article_sentiment_list = []
            for article in articles:
                article_sentiment = [
                    float(article['article_sentiment']),
                    float(article['ticker_sentiment']),
                    float(article['average_publication_sentiment']),
                    article['amount_of_tickers_mentioned'],
                    float(article['ticker_relevance'])
                ]
                article_sentiment_list.append(article_sentiment)
            
            # Average daily sentiment
            daily_sentiment.append(np.mean(article_sentiment_list, axis=0))
        else:
            daily_sentiment.append([0, 0, 0, 0, 0])  # No articles means zero sentiment
        if len(daily_sentiment) >= seq_length:
                dates.append(date_str)
    # Add daily sentiment to combined data
    daily_sentiment = np.array(daily_sentiment)
    for i in range(daily_sentiment.shape[1]):
        combined_data[f'daily_sentiment_{i}'] = daily_sentiment[:, i]
        dates.append(date_str)

    return combined_data, dates

# Prepare the features and target
def prepare_data(combined_data):
    combined_data['next_close'] = combined_data['close'].shift(-1)  # Predict next day's close
    features = combined_data.copy()
    targets = combined_data['next_close'].dropna().values
    
    features = features.drop(columns=['date', 'next_close']).dropna().values
    return features, targets[:-1]  # Remove the last row where y is NaN

# Main function to run the process
def main(stock_file, sentiment_file, seq_length=20):
    stock_data = load_stock_data(stock_file)
    sentiment_data = load_sentiment_data(sentiment_file)
    
    # Combine data and get daily sentiment
    combined_data, dates = combine_data(stock_data, sentiment_data, seq_length)

    # Prepare data for training
    X, y = prepare_data(combined_data)
    
    # Split into training and testing sets
    train_size = int(len(X) * 0.8)  # 80% for training
    test_dates = dates[train_size:]
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Create and fit the XGBoost model
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1)
    model.fit(X_train, y_train)

    # Predictions
    predictions = model.predict(X_test)
    return predictions, y_test, test_dates


# Plotting and saving results
def save_results(predictions, actual_prices, test_dates, output_csv=f'results/csvs/testresults.csv'):
    # Save predictions and actual prices to a CSV file
    results_df = pd.DataFrame({
        'date': test_dates,
        'predicted': predictions.flatten(),
        'actual': actual_prices.flatten()
    })
    key = f'{ticker}_{model_type}_{datetime.now().strftime("%I%M%S_%m_%Y")}'

    print(key)

    results_df.to_csv(f'results/csvs/{key}.csv', index=False)


stock_file = 'training_data/AAPL_prices_csv.csv'
sentiment_file = 'training_data/AAPL_articles_formatted.json'
predictions, actual_prices, test_dates = main(stock_file, sentiment_file)

min_length = min(len(test_dates), len(predictions), len(actual_prices))

# Trim all arrays to the same length
test_dates = test_dates[:min_length]
predictions = predictions[:min_length]
actual_prices = actual_prices[:min_length]

# Save results and plot
save_results(predictions, actual_prices, test_dates)
print("Predictions:", predictions.flatten())
print("Actual Prices:", actual_prices.flatten())


