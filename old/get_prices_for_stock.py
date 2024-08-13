# api key : DO88KI5CUS76U1AN

import requests
import csv


# replace the "demo" apikey below with your own key from https://www.alphavantage.co/support/#api-key
url = 'https://www.alphavantage.co/query?function=TIME_SERIES_WEEKLY&symbol=IBM&apikey=DO88KI5CUS76U1AN&datatype=csv'
r = requests.get(url)

decoded_content = r.content.decode('utf-8')

cr = csv.reader(decoded_content.splitlines(), delimiter=',')
my_list = list(cr)
f = open("test.csv", "a")
for row in my_list:
    f.write(f"{",".join(row)}\n")