import pandas as pd
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import xgboost as xgb
import matplotlib.pyplot as plt

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

# Define the article-level neural network model
class ArticleSentimentNN(nn.Module):
    def __init__(self):
        super(ArticleSentimentNN, self).__init__()
        self.fc1 = nn.Linear(5, 10)  # 5 sentiment stats
        self.fc2 = nn.Linear(10, 1)   # Output: daily sentiment score

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Combine stock and sentiment data
def combine_data(stock_data, sentiment_data):
    combined_data = stock_data.copy()
    daily_sentiment = []

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

            # Convert to tensor and process with neural network
            article_tensor = torch.tensor(article_sentiment_list, dtype=torch.float32)
            model = ArticleSentimentNN()
            daily_sentiment_scores = model(article_tensor).detach().numpy()
            daily_sentiment.append(np.mean(daily_sentiment_scores))  # Average daily sentiment
        else:
            daily_sentiment.append(0)  # No articles means zero sentiment

    # Add daily sentiment to combined data
    combined_data['daily_sentiment'] = daily_sentiment

    return combined_data

# Prepare the features and target
def prepare_data(combined_data):
    combined_data['next_close'] = combined_data['close'].shift(-1)  # Predict next day's close
    features = combined_data[['daily_sentiment']].copy()
    features['current_close'] = combined_data['close']
    targets = combined_data['next_close'].dropna().values
    return features.dropna().values, targets[:-1]  # Remove the last row where y is NaN

# Main function to run the process
def main(stock_file, sentiment_file):
    stock_data = load_stock_data(stock_file)
    sentiment_data = load_sentiment_data(sentiment_file)
    
    # Combine data and get daily sentiment
    combined_data = combine_data(stock_data, sentiment_data)

    # Prepare data for training
    X, y = prepare_data(combined_data)
    
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create and fit the XGBoost model
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1)
    model.fit(X_train, y_train)

    # Predictions
    predictions = model.predict(X_test)
    return predictions, y_test


# Plotting and saving results
def save_results(predictions, actual_prices, output_csv='results.csv', output_png='predictions_vs_actual.png'):
    # Save predictions and actual prices to a CSV file
    results_df = pd.DataFrame({
        'Predicted': predictions.flatten(),
        'Actual': actual_prices.flatten()
    })
    results_df.to_csv(output_csv, index=False)

    # Plotting the predictions vs actual prices
    plt.figure(figsize=(10, 6))
    plt.plot(actual_prices.flatten(), label='Actual Prices', color='blue')
    plt.plot(predictions.flatten(), label='Predicted Prices', color='orange')
    plt.title('Predicted vs Actual Stock Prices')
    plt.xlabel('Sample Index')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.grid()
    plt.savefig(output_png)
    plt.close()

stock_file = 'training_data/AAPL_prices_csv.csv'
sentiment_file = 'training_data/AAPL_articles_formatted.json'
predictions, actual_prices = main(stock_file, sentiment_file)

# Save results and plot
save_results(predictions, actual_prices)
print("Predictions:", predictions.flatten())
print("Actual Prices:", actual_prices.flatten())

