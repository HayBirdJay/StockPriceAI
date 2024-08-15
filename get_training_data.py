from pathlib import Path
import requests
import json
from datetime import datetime, timedelta
from dotenv import load_dotenv
import os


def getPriceData(ticker, premium_key):
    """
    gets all stock close data available for given ticker on alphavantage.
    :param ticker: company's stock ticker
    "param premium_key: premium API key for alphavantage.
    """
    price_url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={ticker}&apikey={premium_key}&outputsize=full"
    print(f"getting historical {ticker} prices")
    with open(f"training_data/{ticker}_prices_unformatted.json", "w") as f:
        response = requests.get(price_url).json()
        f.write(json.dumps(response["Time Series (Daily)"], indent=2))
        f.close()

def getArticleData(ticker, premium_key):
    """
    gets all article data available for given ticker on alphavantage.
    :param ticker: company's stock ticker
    "param premium_key: premium API key for alphavantage.
    """
    articles_url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={ticker}&apikey={premium_key}&limit=1000&sort=EARLIEST"          
    print(f"getting historical {ticker} articles")
    with open(f"training_data/{ticker}_articles_unformatted.json", "w") as f:
        response = requests.get(articles_url).json()

        all_articles = []

        publications = {}

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

def formatPriceData(ticker):
    """
    Formats the price data according to our model's expectations by filling in days with no price value
    i.e. weekends and bank holidays with holdover data from the previous day. 
    """
    with open(f"training_data/{ticker}_prices_unformatted.json", "r") as f:
        prices = json.loads(f.read())

        start_date = list(prices.keys())[len(prices.keys()) - 1]
        end_date = list(prices.keys())[0]
        
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        
        current_date = end

        while current_date >= start:
            str_curr_date = current_date.strftime("%Y-%m-%d")   

            if str_curr_date not in prices:
                tomorrow_ish = current_date + timedelta(days=1)
                prices[str_curr_date] = prices[tomorrow_ish.strftime("%Y-%m-%d")]
            else:
                prices[str_curr_date] = prices[str_curr_date]["4. close"]

            current_date -= timedelta(days=1)
        f.close()

        with open(f"training_data/{ticker}_prices_formatted.json", "w") as f:
            f.write(json.dumps(prices))
            f.close()

def formatArticleData(ticker):
    """
    Formats article data to our model's expecation by filling in empty sentiment days with holds with a temperature loss
    from the last available data point.
    """
    with open(f"training_data/{ticker}_articles_unformatted.json", "r") as f:
        articles = json.loads(f.read())
        new_articles = {}

        for article in articles:
            date = article["date"]
            article.pop('date', None)
            if date not in new_articles:
                new_articles[date] = [article]
            else:
                new_articles[date].append(article)

        start_date = list(new_articles.keys())[0]
        end_date = list(new_articles.keys())[len(new_articles.keys()) - 1]
        
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")

        current_date = start

        while current_date <= end:
            str_curr_date = current_date.strftime("%Y-%m-%d")   
            yesterday = current_date - timedelta(days=1)
            
            if str_curr_date not in new_articles:
                new_articles[str_curr_date] = new_articles[yesterday.strftime("%Y-%m-%d")]
            current_date += timedelta(days=1)
        f.close()

        with open(f"training_data/{ticker}_articles_formatted.json", "w") as f:
            f.write(json.dumps(new_articles))
            f.close()


def getTrainingData(ticker):
    load_dotenv()
    premium_key = os.getenv("API_KEY")

    getPriceData(ticker, premium_key)
    formatPriceData(ticker)
    getArticleData(ticker, premium_key)
    formatArticleData(ticker)


    



