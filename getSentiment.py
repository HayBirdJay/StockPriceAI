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

# Function to set up the WebDriver
def setup_driver():
    # For Chrome
    # service = Service('path/to/chromedriver')
    # driver = webdriver.Chrome(service=service)
    
    # For Safari
    service = SafariService()
    driver = webdriver.Safari(service=service)
    
    return driver

# Function to scrape text from a financial article
def scrape_article_text(url):
    driver = setup_driver()
    driver.get(url)
    
    # Wait until the main content is loaded (you may need to adjust this for different sites)
    try:
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.TAG_NAME, "p"))
        )
    except Exception as e:
        print(f"Error loading page content: {e}")
        driver.quit()
        return ""
    
    # Extract text from <p> tags
    paragraphs = driver.find_elements(By.TAG_NAME, "p")
    article_text = ' '.join([para.text for para in paragraphs])
    
    driver.quit()
    return article_text

# Function to perform sentiment analysis using a pre-trained BERT model
def analyze_sentiment_bert(article_text):
    sentiment_pipeline = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")
    result = sentiment_pipeline(article_text[:512])  # BERT models have a token limit; adjust as needed
    score = result[0]['score']
    
    # # Convert the sentiment label to a numeric score
    # label_to_score = {
    #     '1 star': -10,
    #     '2 stars': -5,
    #     '3 stars': 0,
    #     '4 stars': 5,
    #     '5 stars': 10
    # }
    # sentiment_label = result[0]['label']
    # normalized_score = label_to_score.get(sentiment_label, 0)
    
    return score

# Main function
def main(article_url):
    article_text = scrape_article_text(article_url)
    if article_text:
        sentiment_score = analyze_sentiment_bert(article_text)
        print(f"Sentiment Score: {sentiment_score}")
    else:
        print("Could not extract article text.")

def getSentimentFromArticleJSON():
    with open("articles.json", "r") as file:
        articles = json.load(file)
        
    old_date = "20240805"
    avg_sentiment_score = 0
    num_articles = 0
    date_sentiment = {}
    for article in articles["feed"]:
        date = article["time_published"][:8]
        if datetime.strptime(date, "%Y%m%d") < datetime.strptime(old_date, "%Y%m%d") - timedelta(weeks = 1):
             date_sentiment[old_date] = avg_sentiment_score
             avg_sentiment_score = 0
             num_articles = 0
             old_date = date
        article_score = article["overall_sentiment_score"]
        avg_sentiment_score = ((avg_sentiment_score*num_articles) + article_score)/ (num_articles + 1)
        num_articles += 1
    
    with open("sentiments.csv", "w") as file:
        writer = csv.writer(file)
        for date, sentiment in date_sentiment.items():
            writer.writerow([date, sentiment])


# Example usage
article_url = "https://www.fool.com/investing/2024/07/27/the-best-growth-stock-that-nobody-is-talking-about/"
# main(article_url)
getSentimentFromArticleJSON()
