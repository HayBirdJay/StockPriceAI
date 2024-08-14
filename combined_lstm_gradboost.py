import pandas as pd
import numpy as np
import json
import torch
import torch.nn as nn
import torch.optim as optim
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import plot_importance
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

# Prepare data for XGBoost
def prepare_data_for_xgb(data):
    data['next_close'] = data['close'].shift(-1)  # Predict next day's close
    features = data[['daily_sentiment']].copy()
    features['current_close'] = data['close']
    targets = data['next_close'].dropna().values
    return features.dropna().values, targets[:-1]  # Remove the last row where y is NaN

# Train initial XGBoost model
def train_xgb_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1)
    model.fit(X_train, y_train)
    return model, X_test, y_test

# Identify most important features
def get_important_features(model):
    plot_importance(model)
    plt.show()
    importance_df = pd.DataFrame(model.feature_importances_, index=['EMA', 'EMV', 'close'], columns=['importance'])
    important_features = importance_df.nlargest(2, 'importance').index.tolist()
    return important_features

# Prepare sequences for LSTM
def prepare_sequences(data, feature, seq_length=20):
    sequences = []
    targets = []
    for idx in range(len(data) - seq_length):
        sequences.append(data[feature].iloc[idx:idx + seq_length].values)
        targets.append(data['close'].iloc[idx + seq_length])
    return np.array(sequences), np.array(targets)

# Define univariate LSTM model
class UnivariateLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(UnivariateLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, (hn, cn) = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out, hn  # Returning hidden state

# Train LSTM model
def train_lstm_model(X_train, y_train, input_size, hidden_size, output_size):
    model = UnivariateLSTM(input_size, hidden_size, output_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

    num_epochs = 100
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs, _ = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    return model

# Create features using the second-to-last LSTM layer
def create_lstm_features(model, X):
    model.eval()
    X_tensor = torch.tensor(X, dtype=torch.float32)
    with torch.no_grad():
        _, hidden_states = model(X_tensor)
    return hidden_states.squeeze().numpy()

# Retrain XGBoost model with additional features
def retrain_xgb_with_lstm_features(X, y, lstm_features):
    X_combined = np.hstack((X, lstm_features))
    X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42)
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1)
    model.fit(X_train, y_train)
    return model, X_test, y_test

# Main function to run the process
def main(stock_file):

    stock_data = load_stock_data(stock_file)
    sentiment_data = load_sentiment_data(sentiment_file)
    
    # Combine data and get daily sentiment
    combined_data = combine_data(stock_data, sentiment_data)

    X, y = prepare_data_for_xgb(combined_data)

    # Step 1: Train initial XGBoost model
    xgb_model, X_test, y_test = train_xgb_model(X, y)

    # Step 2: Identify the most important features
    important_features = get_important_features(xgb_model)
    print(f"Important features: {important_features}")

    # Step 3: Train univariate LSTM on the most important features
    feature_data = stock_data[important_features]
    X_lstm, y_lstm = prepare_sequences(stock_data, important_features[0])  # Using first important feature for simplicity

    # Split LSTM data
    train_size = int(len(X_lstm) * 0.8)
    X_train_lstm, y_train_lstm = X_lstm[:train_size], y_lstm[:train_size]
    X_test_lstm, y_test_lstm = X_lstm[train_size:], y_lstm[train_size:]

    lstm_model = train_lstm_model(X_train_lstm, y_train_lstm, input_size=1, hidden_size=50, output_size=1)

    # Step 4: Use the second-to-last LSTM layer to create features for XGBoost
    lstm_features_train = create_lstm_features(lstm_model, X_train_lstm)
    lstm_features_test = create_lstm_features(lstm_model, X_test_lstm)

    # Step 5: Retrain XGBoost model with additional LSTM features
    xgb_final_model, X_test_combined, y_test_combined = retrain_xgb_with_lstm_features(X[:train_size], y[:train_size], lstm_features_train)

    # Make predictions
    predictions = xgb_final_model.predict(X_test_combined)

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
predictions, actual_prices, test_dates = main(stock_file)

# Save results and plot
save_results(predictions, actual_prices, test_dates)
print("Predictions:", predictions.flatten())
print("Actual Prices:", actual_prices.flatten())
