import pandas as pd
import numpy as np
import yfinance as yf
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim


# Load your article data into a DataFrame
articles_df = pd.read_csv('articles.csv')
articles_df['date'] = pd.to_datetime(articles_df['date'])


# Perform Sentiment Analysis using BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')

def get_sentiment_score(text):
    inputs = tokenizer(text, return_tensors='pt', max_length=512, truncation=True, padding=True)
    outputs = model(**inputs)
    scores = outputs.logits.detach().numpy()
    sentiment = np.argmax(scores)
    return sentiment + 1  # Assuming sentiment score from 1 to 5

articles_df['sentiment_score'] = articles_df['article_text'].apply(get_sentiment_score)

# Load and Process Stock Price Data
chipotle_stock = yf.download('CMG', start='YYYY-MM-DD', end='YYYY-MM-DD')
chipotle_stock.reset_index(inplace=True)
stock_prices = chipotle_stock[['Date', 'Close']]
stock_prices.columns = ['date', 'stock_price']

# Merge Article Sentiment Data with Stock Price Data
daily_sentiment = articles_df.groupby('date')['sentiment_score'].mean().reset_index()
merged_df = pd.merge(stock_prices, daily_sentiment, on='date', how='inner')


# Prepare Data for Training
for lag in range(1, 8):
    merged_df[f'sentiment_lag_{lag}'] = merged_df['sentiment_score'].shift(lag)

merged_df.dropna(inplace=True)

X = merged_df[[f'sentiment_lag_{lag}' for lag in range(1, 8)]]
y = merged_df['stock_price']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# Train-Test Split and DataLoader Preparation
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train.values, dtype=torch.float32))
test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test.values, dtype=torch.float32))

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define the Model
class StockPredictor(nn.Module):
    def __init__(self, input_dim):
        super(StockPredictor, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = StockPredictor(input_dim=X_train.shape[1])
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# Train the Model
num_epochs = 50

for epoch in range(num_epochs):
    model.train()
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs.flatten(), y_batch)
        loss.backward()
        optimizer.step()
    
    # Validation
    model.eval()
    with torch.no_grad():
        val_losses = []
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch)
            val_loss = criterion(outputs.flatten(), y_batch)
            val_losses.append(val_loss.item())
    print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {loss.item()}, Validation Loss: {np.mean(val_losses)}')

# Evaluate Model
model.eval()
with torch.no_grad():
    predictions = []
    true_values = []
    for X_batch, y_batch in test_loader:
        outputs = model(X_batch)
        predictions.extend(outputs.flatten().numpy())
        true_values.extend(y_batch.numpy())

mse = mean_squared_error(true_values, predictions)
print(f'Mean Squared Error: {mse}')


# Predict Future Stock Prices
# Prepare recent sentiment data
recent_sentiments = ...  # Assume this DataFrame has recent sentiment scores

# Scale recent sentiment data
recent_sentiments_scaled = scaler.transform(recent_sentiments[[f'sentiment_lag_{lag}' for lag in range(1, 8)]])

# Convert to tensor and predict
model.eval()
with torch.no_grad():
    recent_sentiments_tensor = torch.tensor(recent_sentiments_scaled, dtype=torch.float32)
    predicted_stock_price = model(recent_sentiments_tensor)
    print(f'Predicted Stock Price: {predicted_stock_price.numpy()}')
