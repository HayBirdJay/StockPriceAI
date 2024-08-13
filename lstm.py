import pandas as pd
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

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

    for date_str in stock_data['date'].dt.strftime('%Y-%m-%d'):
        articles = sentiment_data.get(date_str, [])
        if articles:
            daily_sentiment = []
            for article in articles:
                article_sentiment = [
                    article['article_sentiment'],
                    article['ticker_sentiment_score'],
                    article['average_sentiment_for_publication'],
                    article['amount_of_tickers_mentioned'],
                    article['ticker_relevance']
                ]
                daily_sentiment.append(article_sentiment)
                
            # Create a sequence if we have enough articles
            if len(daily_sentiment) >= seq_length:
                for i in range(len(daily_sentiment) - seq_length):
                    sequences.append(daily_sentiment[i:i + seq_length])
                    targets.append(stock_data.loc[stock_data['date'] == date_str, 'close'].values[0])
    
    return np.array(sequences), np.array(targets)

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

# Main function to run the process
def main(stock_file, sentiment_file, seq_length=3):
    stock_data = load_stock_data(stock_file)
    sentiment_data = load_sentiment_data(sentiment_file)

    # Prepare data for LSTM
    X, y = prepare_sequences(stock_data, sentiment_data, seq_length)

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Convert to tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

    # Create the LSTM model
    model = StockPriceLSTM(input_size=5, hidden_size=50, output_size=1)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training the model
    num_epochs = 100
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    # Testing the model
    model.eval()
    with torch.no_grad():
        predictions = model(X_test_tensor)

    return predictions.numpy(), y_test


stock_file = 'stock_prices.csv'
sentiment_file = 'sentiment_data.json'
predictions, actual_prices = main(stock_file, sentiment_file)
print("Predictions:", predictions.flatten())
print("Actual Prices:", actual_prices.flatten())
