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

# Prepare input sequences and targets for LSTM
def prepare_sequences(stock_data, sentiment_data, seq_length):
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

    return np.array(sequences), np.array(targets), dates

# Define the LSTM model
class StockPriceLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(StockPriceLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # Get the output of the last time step
        out = self.fc(out)
        return out

# Combine stock and sentiment data
def combine_data(stock_data, sentiment_data, seq_length, model):
    combined_data = stock_data.copy()
    daily_sentiment = []

    # Prepare input for LSTM
    X, _, dates = prepare_sequences(stock_data, sentiment_data, seq_length)
    
    # Convert to tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)

    # Get LSTM features
    model.eval()
    with torch.no_grad():
        lstm_features = model(X_tensor).numpy()

    combined_data = combined_data.iloc[seq_length:]  # Remove initial rows without sentiment data
    combined_data['lstm_features'] = list(lstm_features)

    return combined_data, dates[seq_length:]

# Prepare the features and target
def prepare_data(combined_data):
    combined_data['next_close'] = combined_data['close'].shift(-1)  # Predict next day's close
    features = combined_data[['lstm_features']].copy()
    features['current_close'] = combined_data['close']
    targets = combined_data['next_close'].dropna().values
    return features.dropna().values.tolist(), targets[:-1]  # Remove the last row where y is NaN

# Main function to run the process
def main(stock_file, sentiment_file, seq_length=20):
    stock_data = load_stock_data(stock_file)
    sentiment_data = load_sentiment_data(sentiment_file)

    # Prepare data for LSTM
    X, y, _ = prepare_sequences(stock_data, sentiment_data, seq_length)

    # Split into training and testing sets based on temporal order
    train_size = int(len(X) * 0.8)  # 80% for training
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Convert to tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

    # Create the LSTM model
    lstm_model = StockPriceLSTM(input_size=5, hidden_size=50, output_size=1)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(lstm_model.parameters(), lr=0.001)

    # Training the LSTM model
    num_epochs = 100
    for epoch in range(num_epochs):
        lstm_model.train()
        optimizer.zero_grad()
        outputs = lstm_model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    # Combine data and get LSTM features
    combined_data, dates = combine_data(stock_data, sentiment_data, seq_length, lstm_model)

    # Prepare data for XGBoost
    X, y = prepare_data(combined_data)

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create and fit the XGBoost model
    xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1)
    xgb_model.fit(X_train, y_train)

    # Predictions
    predictions = xgb_model.predict(X_test)
    
    # Plot predictions vs actual values
    plt.figure(figsize=(10, 5))
    plt.plot(predictions, label='Predictions')
    plt.plot(y_test, label='Actual')
    plt.legend()
    plt.show()
    
    return predictions, y_test, dates[len(dates) - len(y_test):]

# Run the main function
if __name__ == "__main__":
    stock_file = 'path_to_stock_data.csv'
    sentiment_file = 'path_to_sentiment_data.json'
    predictions, actuals, test_dates = main(stock_file, sentiment_file, seq_length=20)
