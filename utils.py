import json
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast

def json_to_csv(ticker="AAPL"):
    """
    Takes in a JSON of the formatted price data and resturns it as a CSV file.
    """
    with open(f'training_data/{ticker}_prices_formatted.json', 'r') as f:
        prices = json.loads(f.read())
        f.close()

    with open(f'training_data/{ticker}_prices_csv.csv', 'w') as f:
        csv_writer = csv.writer(f)

        csv_writer.writerow(["date", "close"])
        for day in prices:
            csv_writer.writerow([day, prices[day]])
        
        f.close()


def check_loss(csv_file):
    """
    Given a CSV file that has at least two time series columns corresponding to 
    predicted and actual values, calculate the loss as a mean square error. 
    """
    df = pd.read_csv(csv_file)
    mse_loss = np.mean((df['Predicted'] - df['Actual']) ** 2)
    print(f'Mean Squared Error Loss: {mse_loss:.4f}')

def generate_loss_graph(csv_name):
    """
    Given a csv file graph the loss as a function of epoch count.
    """
    with open(f'results/loss/{csv_name}.txt', 'r') as file:
        content = file.read().strip()
        numbers = ast.literal_eval(content)

    x = np.arange(len(numbers))
    plt.figure(figsize=(10, 6))
    plt.plot(x, numbers, marker='o')
    plt.title('Training Loss over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.grid(True)
    plt.savefig(f'results/loss_graphs/{csv_name}.png')