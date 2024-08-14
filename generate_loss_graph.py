import matplotlib.pyplot as plt
import numpy as np
import ast

name = 'AAPL_GBOOST_065312_08_2024'
# Read numbers from the file
with open(f'results/loss/{name}.txt', 'r') as file:
    content = file.read().strip()
    numbers = ast.literal_eval(content)

# Create x-axis values (indices of the numbers)
x = np.arange(len(numbers))

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(x, numbers, marker='o')
plt.title('Training Loss over Time')
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.grid(True)

# Save the plot as an image file
plt.savefig(f'results/loss_graphs/{name}.png')