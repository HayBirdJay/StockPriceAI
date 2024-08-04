# api key : DO88KI5CUS76U1AN
import json
import requests
import csv
from datetime import datetime, timedelta

def getFirstWeek():
    """
    Returns the earliest week we have financial data for. 
    """
    with open("test.csv", "r") as file:
        #add code here to retrieve first week

def main():
    current_date = datetime.now().strftime("%Y%m%d") + "T0000"
    print(current_date)
    week_ago = datetime.now() - timedelta(weeks =1)
    print(week_ago)


# # replace the "demo" apikey below with your own key from https://www.alphavantage.co/support/#api-key
# url = 'https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers=TSLA&apikey=DO88KI5CUS76U1AN&limit=1000'
# r = requests.get(url)
# if(r.status_code==200):
#     with open("articles.json", "w") as f:
#         json.dump(r.json(), f, ensure_ascii=False, indent=4)
    
#     with open("articles.json", "r") as file: 
#         articles = file.read()

#     articles = articles.replace("}{", "},{")
#     articles = f"[{articles}]"

#     with open("articles.json", 'w') as file:
#         file.write(articles)
main()
