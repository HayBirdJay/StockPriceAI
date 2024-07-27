# api key : DO88KI5CUS76U1AN
import json
import requests
import csv


# replace the "demo" apikey below with your own key from https://www.alphavantage.co/support/#api-key
url = 'https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers=TSLA&apikey=DO88KI5CUS76U1AN&limit=1000'
r = requests.get(url)
if(r.status_code==200):
    with open("articles.json", "a") as f:
        json.dump(r.json(), f, ensure_ascii=False, indent=4)