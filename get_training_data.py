from pathlib import Path
import requests
import json
from datetime import datetime

# api keys : DO88KI5CUS76U1AN
#            WD67GOQENJJEPXYE
# premium key: BGXU14MCWEJ90CQ6

api_key_1 = "DO88KI5CUS76U1AN"
api_key_2 = "WD67GOQENJJEPXYE"
premium_key = "BGXU14MCWEJ90CQ6"

# choose ticker to get training data for
ticker = "AAPL"

# setting api urls

price_url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={ticker}&apikey={premium_key}&outputsize=full"
articles_url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={ticker}&apikey={premium_key}&limit=1000&sort=EARLIEST"


# either write prices to file or let user know we already have that data

# get PRICES
print(f"getting historical {ticker} prices")
with open(f"training_data/{ticker}_prices_unformatted.json", "w") as f:
    response = requests.get(price_url).json()
    f.write(json.dumps(response["Time Series (Daily)"], indent=2))
    f.close()
        
# get ARTICLES
print(f"getting historical {ticker} articles")
with open(f"training_data/{ticker}_articles_unformatted.json", "w") as f:
    response = requests.get(articles_url).json()

    all_articles = []

    publications = {}
    
    counter = 0

    while True:
        feed = response["feed"]
        
        for article in feed:

            ticker_stats = next((t for t in article["ticker_sentiment"] if t["ticker"] == ticker), None)

            format_article = {
                "date": datetime.strptime(article["time_published"][:8], "%Y%m%d").strftime("%Y-%m-%d"),
                "article_sentiment": article["overall_sentiment_score"],
                "ticker_sentiment": ticker_stats["ticker_sentiment_score"],
                "ticker_relevance": ticker_stats["relevance_score"],
                "average_publication_sentiment": article["source_domain"],
                "amount_of_tickers_mentioned": len(article["ticker_sentiment"])
            }

            if article["source_domain"] not in publications:
                publications[article["source_domain"]] = [1, article["overall_sentiment_score"]]

            else:
                publications[article["source_domain"]][0] = publications[article["source_domain"]][0] + 1
                publications[article["source_domain"]][1] = publications[article["source_domain"]][1] + article["overall_sentiment_score"]

            all_articles.append(format_article)

        last_article_datetime = feed[-1]["time_published"][:13]
        response = requests.get(f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={ticker}&apikey={premium_key}&limit=1000&sort=EARLIEST&time_from={last_article_datetime}").json()
        print(len(all_articles))
        print(last_article_datetime)
        if(int(response["items"]) == 1):
            break


    publications = {k: v[1]/v[0] for k, v in publications.items()}

    for article in all_articles:
        article["average_publication_sentiment"] = publications[article["average_publication_sentiment"]]

    f.write(json.dumps(all_articles, indent=2))
    f.close()



