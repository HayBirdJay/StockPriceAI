# old api key : DO88KI5CUS76U1AN
#new api key (to bypass daily limit): WD67GOQENJJEPXYE
#news api key: 73d0513b2a044fc7812d45b914a6f6f7
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
    current_week = datetime.now()
    current_week = current_week.strftime("%Y%m%d") + "T0000"
    print(current_week)
    first_week = getFirstWeek() + "T0000"
    print(first_week)
    # while current_week > first_week:
    #     week_ago_formatted = week_ago.strftime("%Y%m%d") + "T0000"
    #     current_week_formatted = current_week.strftime("%Y%m%d") + "T0000"
    #     url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers=TSLA&apikey=DO88KI5CUS76U1AN&time_from={week_ago_formatted}&time_to={current_week_formatted}&limit=1000"
    #     getDatafromURL(url)
    #     week_ago = week_ago - timedelta(weeks=1)
    #     current_week = current_week - timedelta(weeks =1)
    url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers=CMG&apikey=70OWRJFFVU26WXS3&time_from={first_week}&time_to={current_week}"
    getDatafromURL(url)

    with open("articles.json", "r") as file: 
        articles = file.read()

    articles = articles.replace("}{", "},{")
    articles = f"[{articles}]"

    with open("articles.json", 'w') as file:
        file.write(articles)

    

main()
