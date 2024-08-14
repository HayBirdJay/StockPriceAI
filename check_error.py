import pandas as pd
import numpy as np

df = pd.read_csv("results/csvs/AAPL_combo_054110_08_2024.csv")

# Calculate Mean Squared Error (MSE) loss
mse_loss = np.mean((df['Predicted'] - df['Actual']) ** 2)

print(f'Mean Squared Error Loss: {mse_loss:.4f}')