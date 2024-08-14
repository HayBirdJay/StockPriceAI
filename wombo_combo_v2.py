import pandas as pd
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from datetime import datetime

ticker = 'AAPL'
model_type = 'combo'

stock_file = 'training_data/AAPL_prices_csv.csv'
sentiment_file = 'training_data/AAPL_articles_formatted.json'

# Training the LSTM model
num_epochs = 10000  # Reduced epochs for demonstration
learning_rate=0.0001
seq_length = 20


# Load stock price data
def load_stock_data():
    stock_data = pd.read_csv(f'training_data/{ticker}_prices_csv.csv')  # Assuming CSV with 'date' and 'close' columns
    stock_data['date'] = pd.to_datetime(stock_data['date'])
    stock_data = stock_data.sort_values(by='date')
    return stock_data

# Load sentiment data from JSON
def load_sentiment_data():
    with open(f'training_data/{ticker}_articles_formatted.json', 'r') as file:
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


class StockPriceLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(StockPriceLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(0.3)  # Add dropout
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])  # Apply dropout to the LSTM output
        out = self.fc(out)
        return out

    
all_loss = []

# Modified main function to include Gradient Boosting
def main_with_gradient_boosting(seq_length=seq_length):
    stock_data = load_stock_data()
    sentiment_data = load_sentiment_data()

    # Split data by year for training and testing
    X_train, X_test, y_train, y_test, train_dates, test_dates = split_data_by_year(stock_data, sentiment_data, seq_length)

    # Convert to tensors for LSTM
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

    # Create the LSTM model
    model = StockPriceLSTM(input_size=5, hidden_size=50, output_size=1)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f'LSTM Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
            all_loss.append(loss.item())

    # Obtain LSTM predictions on training and testing data
    model.eval()
    with torch.no_grad():
        lstm_train_predictions = model(X_train_tensor).numpy().flatten()
        lstm_test_predictions = model(X_test_tensor).numpy().flatten()

    # Calculate residuals on training data
    train_residuals = y_train - lstm_train_predictions

    # Train Gradient Boosting Regressor on residuals
    gbr = GradientBoostingRegressor(n_estimators=100, learning_rate=learning_rate, max_depth=3)
    # Flatten X_train for GBR or use aggregated features
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    gbr.fit(X_train_flat, train_residuals)

    # Predict residuals on testing data
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    residual_predictions = gbr.predict(X_test_flat)

    # Final predictions by adding LSTM predictions and GBR residual predictions
    final_test_predictions = lstm_test_predictions + residual_predictions

    return final_test_predictions, y_test, test_dates



# Plotting and saving results
def save_results(predictions, actual_prices, test_dates):
    # Save predictions and actual prices to a CSV file
    results_df = pd.DataFrame({
        'Date': test_dates,
        'Predicted': predictions.flatten(),
        'Actual': actual_prices.flatten()
    })

    key = f'{ticker}_{model_type}_{datetime.now().strftime("%I%M%S_%m_%Y")}'

    print(key)

    results_df.to_csv(f'results/csvs/{key}.csv', index=False)

    with open(f'results/loss/{key}.txt', 'w') as f:
        f.write(str(all_loss))
        f.close()

predictions, actual_prices, test_dates = main_with_gradient_boosting()

# Save results and plot
save_results(predictions, actual_prices, test_dates)