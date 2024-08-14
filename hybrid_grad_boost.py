import pandas as pd
import json
import numpy as np
import xgboost as xgb
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

# Updated split logic to ensure more balanced train/test sets
def split_data_by_year(stock_data, sentiment_data, seq_length):
    sequences = []
    targets = []
    dates = []

    # Prepare sequences and targets (unchanged)
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

    # Convert to NumPy arrays
    sequences = np.array(sequences)
    targets = np.array(targets)

    # Convert dates back to datetime for easier filtering and convert to Series
    dates = pd.to_datetime(dates)
    dates = pd.Series(dates)  # Convert to Pandas Series for easier datetime access

    # Split by years, ensuring all 2022 and 2023 data is for training, and 2024 data is for testing
    train_mask = (dates.dt.year == 2022) | (dates.dt.year == 2023)
    test_mask = dates.dt.year == 2024

    # Split the data
    X_train, X_test = sequences[train_mask], sequences[test_mask]
    y_train, y_test = targets[train_mask], targets[test_mask]
    train_dates, test_dates = dates[train_mask], dates[test_mask]

    print("Training set:")
    print(train_dates.min(), "to", train_dates.max())
    print("Testing set:")
    print(test_dates.min(), "to", test_dates.max())

    return X_train, X_test, y_train, y_test, train_dates, test_dates

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
    
    # Split data by year for training and testing
    X_train, X_test, y_train, y_test, train_dates, test_dates = split_data_by_year(stock_data, sentiment_data, seq_length)

    X_train = X_train.reshape(X_train.shape[0], -1)  # Flatten seq_length * num_features into a single dimension
    X_test = X_test.reshape(X_test.shape[0], -1)

    # Create and fit the XGBoost model
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=1)
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

