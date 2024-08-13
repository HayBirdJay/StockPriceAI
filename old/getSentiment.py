import requests
import json, csv
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from transformers import pipeline
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.safari.service import Service as SafariService

def takeRollingAverage(data, curr_avg, num_articles):
    return ((curr_avg*num_articles) + data)/ (num_articles + 1)

def getTickerStats(ticker, article_data):
    for ticker_data in article_data["ticker_sentiment"]:
        if ticker_data["ticker"] == ticker:
            return float(ticker_data["ticker_sentiment_score"]), float(ticker_data["relevance_score"])
    raise ValueError(f"no ticker data found for {ticker}")

def getMostRelevantTopic(topic_list):
    most_relevant = topic_list[0]["topic"]
    max_score = 0
    for topic in topic_list: 
        if float(topic["relevance_score"]) > max_score:
            most_relevant = topic["topic"]
            max_score = float(topic["relevance_score"])
    return most_relevant


def getSentimentFromArticleJSON(ticker: str):
    with open("articles.json", "r") as file:
        articles = json.load(file)

    old_date = "20240805"
    avg_sentiment_score = 0
    avg_relevance_score = 0
    num_articles = 0
    date_sentiment = {}
    relevant_topics = {}
    for article_dict in articles: 
        for article in article_dict["feed"]:
            date = article["time_published"][:8]
            if datetime.strptime(date, "%Y%m%d") < datetime.strptime(old_date, "%Y%m%d") - timedelta(weeks = 1):
                most_relevant = max(relevant_topics, key=relevant_topics.get)
                date_sentiment[old_date] = [avg_sentiment_score, avg_relevance_score, most_relevant]
                avg_sentiment_score = 0
                avg_relevance_score = 0
                relevant_topics = {}
                num_articles = 0
                old_date = date

            article_score, ticker_relevancy_score = getTickerStats(ticker, article)
            relevant_topic = getMostRelevantTopic(article["topics"])

            avg_sentiment_score = takeRollingAverage(article_score, avg_sentiment_score, num_articles)
            avg_relevance_score = takeRollingAverage(ticker_relevancy_score, avg_relevance_score, num_articles)

            if relevant_topics.get(relevant_topic, "") == "":
                relevant_topics[relevant_topic] = 1
            else:
                relevant_topics[relevant_topic] += 1
            num_articles += 1
    
    with open("sentiments.csv", "w") as file:
        writer = csv.writer(file)
        for date, sentiment in date_sentiment.items():
            writer.writerow([date, sentiment[0], sentiment[1], sentiment[2]])


# Example usage
article_url = "https://www.fool.com/investing/2024/07/27/the-best-growth-stock-that-nobody-is-talking-about/"
# main(article_url)
getSentimentFromArticleJSON("CMG")
