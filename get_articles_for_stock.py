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
        reader = csv.reader(file)
        for row in reader:
            last_val = row[0]
    last_val = last_val[0:4] + last_val[5:7] + last_val[8:]
    return last_val

def getDatafromURL(url):
    r = requests.get(url)
    if(r.status_code==200):
        with open("articles.json", "a") as f:
            json.dump(r.json(), f, ensure_ascii=False, indent=4)

def main():
    #current_date = datetime.now().strftime("%Y%m%d") + "T0000"
    current_week = datetime.now()
    print(current_week)
    week_ago = datetime.now() - timedelta(weeks =1)
    print(week_ago)

    first_week = datetime.strptime(getFirstWeek(), "%Y%m%d")
    while current_week > first_week:
        week_ago_formatted = week_ago.strftime("%Y%m%d") + "T0000"
        current_week_formatted = current_week.strftime("%Y%m%d") + "T0000"
        url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers=TSLA&apikey=DO88KI5CUS76U1AN&time_from={week_ago_formatted}&time_to={current_week_formatted}&limit=1000"
        getDatafromURL(url)
        week_ago = week_ago - timedelta(weeks=1)
        current_week = current_week - timedelta(weeks =1)

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
getFirstWeek()
