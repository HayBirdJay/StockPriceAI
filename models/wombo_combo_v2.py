import pandas as pd
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from datetime import datetime
import argparse, subprocess


num_epochs = 10000  
learning_rate=0.0001
seq_length = 20

def load_stock_data(ticker):
    stock_data = pd.read_csv(f'training_data/{ticker}_prices_csv.csv')  
    stock_data['date'] = pd.to_datetime(stock_data['date'])
    stock_data = stock_data.sort_values(by='date')
    return stock_data

def load_sentiment_data(ticker):
    with open(f'training_data/{ticker}_articles_formatted.json', 'r') as file:
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


class StockPriceLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(StockPriceLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(0.3)  # Add dropout
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])  
        out = self.fc(out)
        return out

    
all_loss = []

def main_with_gradient_boosting(ticker, seq_length=seq_length):
    stock_data = load_stock_data(ticker)
    sentiment_data = load_sentiment_data(ticker)
    X_train, X_test, y_train, y_test, train_dates, test_dates = split_data_by_year(stock_data, sentiment_data, seq_length)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

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

    model.eval()
    with torch.no_grad():
        lstm_train_predictions = model(X_train_tensor).numpy().flatten()
        lstm_test_predictions = model(X_test_tensor).numpy().flatten()

    train_residuals = y_train - lstm_train_predictions

    gbr = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3)

    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    gbr.fit(X_train_flat, train_residuals)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    residual_predictions = gbr.predict(X_test_flat)

    final_test_predictions = lstm_test_predictions + residual_predictions

    return final_test_predictions, y_test, test_dates


def save_results(predictions, actual_prices, test_dates, outputcsv):
    results_df = pd.DataFrame({
        'date': test_dates,
        'predicted': predictions.flatten(),
        'actual': actual_prices.flatten()
    })


    results_df.to_csv(f'results/{outputcsv}.csv', index=False)

    with open(f'results/loss/{outputcsv}.txt', 'w') as f:
        f.write(str(all_loss))
        f.close()


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-t', '--ticker', required=True)
    argparser.add_argument('-o', '--outputcsv', help="csv file to save results to.", required=True)
    argparser.add_argument('-g','--graphfile',help="if set, graphs the results and saves it to the specified file name.")
    args = argparser.parse_args()

    stock_file = f'training_data/{args.ticker}_prices_csv.csv'
    sentiment_file = f'training_data/{args.ticker}_articles_formatted.json'

    predictions, actual_prices, test_dates = main_with_gradient_boosting(args.ticker)
    save_results(predictions, actual_prices, test_dates)
    print("Predictions:", predictions.flatten())
    print("Actual Prices:", actual_prices.flatten())

    if args.graphfile:
        subprocess.run(['python', 'generate_graph.py', '-t', args.ticker, '-c', args.outputcsv, '-m', 'lstm'])