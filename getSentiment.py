# import time
# from selenium import webdriver
# from selenium.webdriver.common.by import By
# from selenium.webdriver.chrome.service import Service
# from selenium.webdriver.support.ui import WebDriverWait
# from selenium.webdriver.support import expected_conditions as EC
# from selenium.webdriver.safari.service import Service as SafariService
# from selenium.webdriver.common.keys import Keys
# from nltk.sentiment.vader import SentimentIntensityAnalyzer
# from textblob import TextBlob
# import nltk

# # Ensure VADER lexicon is downloaded
# nltk.download('vader_lexicon')

# # Function to set up the WebDriver
# def setup_driver():
#     # For Chrome
#     # service = Service('path/to/chromedriver')
#     # driver = webdriver.Chrome(service=service)
    
#     # For Safari
#     service = SafariService()
#     driver = webdriver.Safari(service=service)
    
#     return driver

# # Function to scrape text from a financial article
# def scrape_article_text(url):
#     driver = setup_driver()
#     driver.get(url)
    
#     # Wait until the main content is loaded (you may need to adjust this for different sites)
#     try:
#         WebDriverWait(driver, 10).until(
#             EC.presence_of_element_located((By.TAG_NAME, "p"))
#         )
#     except Exception as e:
#         print(f"Error loading page content: {e}")
#         driver.quit()
#         return ""
    
#     # Extract text from <p> tags
#     paragraphs = driver.find_elements(By.TAG_NAME, "p")
#     article_text = ' '.join([para.text for para in paragraphs])
    
#     driver.quit()
#     print(article_text)
#     return article_text

# # Function to perform sentiment analysis using VADER
# def analyze_sentiment_vader(article_text):
#     sia = SentimentIntensityAnalyzer()
#     sentiment = sia.polarity_scores(article_text)
    
#     # VADER returns a compound score between -1 and 1
#     compound_score = sentiment['compound']
    
#     # # Normalize score to -10 to 10
#     # normalized_score = compound_score * 10
    
#     return compound_score

# # Function to perform sentiment analysis using TextBlob
# def analyze_sentiment_textblob(article_text):
#     blob = TextBlob(article_text)
#     polarity = blob.sentiment.polarity  # Polarity is between -1 and 1
    
#     # # Normalize score to -10 to 10
#     # normalized_score = polarity * 10
    
#     return polarity

# # Function to combine sentiment scores from multiple methods
# def combined_sentiment_score(article_text):
#     vader_score = analyze_sentiment_vader(article_text)
#     textblob_score = analyze_sentiment_textblob(article_text)
    
#     # Combine the scores (e.g., average)
#     combined_score = (vader_score + textblob_score) / 2
#     return combined_score

# # Main function
# def main(article_url):
#     article_text = scrape_article_text(article_url)
#     if article_text:
#         sentiment_score = combined_sentiment_score(article_text)
#         print(f"Sentiment Score: {sentiment_score} (Scale: -10 to 10)")
#     else:
#         print("Could not extract article text.")

# # Example usage
# article_url = "https://www.cnn.com/2024/07/27/middleeast/lebanon-israel-golan-strikes-intl-latam/index.html"
# main(article_url)

import requests
from bs4 import BeautifulSoup
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
        print(f"Sentiment Score: {sentiment_score} (Scale: -10 to 10)")
    else:
        print("Could not extract article text.")

# Example usage
article_url = "https://www.cnn.com/2024/07/27/middleeast/lebanon-israel-golan-strikes-intl-latam/index.html"
main(article_url)

