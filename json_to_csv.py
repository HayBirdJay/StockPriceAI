import json
import csv

ticker = "AAPL"

with open(f'training_data/{ticker}_prices_formatted.json', 'r') as f:
    prices = json.loads(f.read())
    f.close()

with open(f'training_data/{ticker}_prices_csv.csv', 'w') as f:
    csv_writer = csv.writer(f)

    for day in prices:
        csv_writer.writerow([day, prices[day]])
    
    f.close()