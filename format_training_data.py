from datetime import datetime, timedelta
import json

# make sure same ticker as get_training_data.py
ticker = "AAPL"

# prices
with open(f"training_data/{ticker}_prices_unformatted.json", "r") as f:
    prices = json.loads(f.read())

    start_date = list(prices.keys())[len(prices.keys()) - 1]
    end_date = list(prices.keys())[0]
    
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    
    current_date = end

    while current_date >= start:
        str_curr_date = current_date.strftime("%Y-%m-%d")   

        # if it's a weekend, just wrap over from the next monday
        if str_curr_date not in prices:
            tomorrow_ish = current_date + timedelta(days=1)
            prices[str_curr_date] = prices[tomorrow_ish.strftime("%Y-%m-%d")]
        else:
            prices[str_curr_date] = prices[str_curr_date]["4. close"]

        # move to next day
        current_date -= timedelta(days=1)
    f.close()

    with open(f"training_data/{ticker}_prices_formatted.json", "w") as f:
        f.write(json.dumps(prices))
        f.close()

# articles
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
        
        if str_curr_date not in new_articles:
            new_articles[str_curr_date] = []
        
        # move to next day
        current_date += timedelta(days=1)
    f.close()

    with open(f"training_data/{ticker}_articles_formatted.json", "w") as f:
        f.write(json.dumps(new_articles))
        f.close()




    