import pandas as pd
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Load stock price data
def load_stock_data(stock_file):
    stock_data = pd.read_csv(stock_file)  # Assuming CSV with 'date' and 'close' columns
    stock_data['date'] = pd.to_datetime(stock_data['date'])
    stock_data = stock_data.sort_values(by='date')  
    return stock_data

# Load sentiment data from JSON
def load_sentiment_data(sentiment_file):
    with open(sentiment_file, 'r') as file:
        sentiment_data = json.load(file)
    
    # Convert sentiment data to a DataFrame for sorting
    sentiment_list = []
    for date_str, articles in sentiment_data.items():
        for article in articles:
            article_data = {
                'date': pd.to_datetime(date_str),  # Convert to datetime
                'article_sentiment': article['article_sentiment'],
                'ticker_sentiment': article['ticker_sentiment'],
                'average_publication_sentiment': article['average_publication_sentiment'],
                'amount_of_tickers_mentioned': article['amount_of_tickers_mentioned'],
                'ticker_relevance': article['ticker_relevance']
            }
            sentiment_list.append(article_data)
    
    # Create DataFrame and sort by date
    sentiment_df = pd.DataFrame(sentiment_list)
    sentiment_df = sentiment_df.sort_values(by='date')  # Sort from oldest to newest

    # Convert back to dict
    sorted_sentiment_data = {}
    for _, row in sentiment_df.iterrows():
        date_str = row['date'].strftime('%Y-%m-%d')
        if date_str not in sorted_sentiment_data:
            sorted_sentiment_data[date_str] = []
        sorted_sentiment_data[date_str].append(row.to_dict())
    
    return sorted_sentiment_data


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

    # Print counts for debugging
    print("Total sequences:", len(sequences))
    print("Total targets:", len(targets))
    print("Total dates:", len(dates))

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

# Main function to run the process
def main(stock_file, sentiment_file, seq_length=20):
    stock_data = load_stock_data(stock_file)
    sentiment_data = load_sentiment_data(sentiment_file)

    # Prepare data for LSTM
    X, y, dates = prepare_sequences(stock_data, sentiment_data, seq_length)

    # Split into training and testing sets based on temporal order
    train_size = int(len(X) * 0.8)  # 80% for training
    test_dates = dates[train_size:]
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Convert to tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

    # Create the LSTM model
    model = StockPriceLSTM(input_size=5, hidden_size=50, output_size=1)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # Training the model
    num_epochs = 100000
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
        train_predictions = model(X_train_tensor)

    train_results_df = pd.DataFrame({
        'date': dates[:train_size],
        'predicted': train_predictions.flatten(),
        'actual': y_train.flatten()
    })
    train_results_df.to_csv("train.csv", index=False)

    return predictions.numpy(), y_test, test_dates

# Plotting and saving results
def save_results(predictions, actual_prices, test_dates, output_csv='results_lstm_100000_again.csv', output_png='predictions_vs_actual_15_slower.png'):
    # Save predictions and actual prices to a CSV file
    print(test_dates)
    print(predictions.flatten())
    print(actual_prices.flatten())
    results_df = pd.DataFrame({
        'date': test_dates,
        'predicted': predictions.flatten(),
        'actual': actual_prices.flatten()
    })
    results_df.to_csv(output_csv, index=False)


stock_file = 'training_data/AAPL_prices_csv.csv'
sentiment_file = 'training_data/AAPL_articles_formatted.json'
predictions, actual_prices, test_dates = main(stock_file, sentiment_file)

# Save results and plot
save_results(predictions, actual_prices, test_dates)
print("Predictions:", predictions.flatten())
print("Actual Prices:", actual_prices.flatten())
