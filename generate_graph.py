import csv
import json
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

ticker = "AAPL"

model = 'LSTM'

csv_name = 'AAPL_GBOOST_064019_08_2024'

# Read the CSV file
df = pd.read_csv(f"results/csvs/{csv_name}.csv")


# Convert the 'date' column to datetime format
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(by=['date'], ascending=True)


# Plot the data
plt.figure(figsize=(10, 6))
plt.plot(df['date'], df['predicted'], label='Predicted Value')
plt.plot(df['date'], df['actual'], label='Actual Value')

# Adding titles and labels
plt.title('Predicted vs Actual Values Over Time')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.grid(True)

# Show the plot
plt.savefig(f'results/graphs/{ticker}_{model}_{datetime.now().strftime("%I%M%S_%m_%Y")}')
