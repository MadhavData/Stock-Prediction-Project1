#!/usr/bin/env python
# coding: utf-8

# In[2]:


import tweepy
API_KEY = '0pyCK0JcXf2OES6ZVK4kEzbTX'  # Consumer Key
API_SECRET_KEY = 'Z6Q3A9eKxsUe3yzzSWcKcmPm39vQVuYylkIx2kieRv0wGFBCgW'  
ACCESS_TOKEN = '1859283643769798656-DnasgKhnKJPlqTuaPmZ9RtCiLKfwXB'  
ACCESS_TOKEN_SECRET = 'll6ZkdnAUjRs1by2JcIGbz0vRLlr7pX5cpUXsB6xQsz6A'
auth = tweepy.OAuth1UserHandler(
    consumer_key=API_KEY,
    consumer_secret=API_SECRET_KEY,
    access_token=ACCESS_TOKEN,
    access_token_secret=ACCESS_TOKEN_SECRET
)
api = tweepy.API(auth)
try:
    user = api.verify_credentials()
    print(f"Authenticated as {user.screen_name}")
except tweepy.TweepError as e:
    print(f"Error: Unable to authenticate. {e}")


# In[3]:


import tweepy
import pandas as pd
def scrape_tweets(keyword, num_tweets=100):
    tweets = tweepy.Cursor(api.search_tweets, q=keyword, lang="en", tweet_mode='extended').items(num_tweets)
    tweet_data = []
    for tweet in tweets:
        tweet_data.append([tweet.created_at, tweet.user.screen_name, tweet.full_text, tweet.favorite_count, tweet.retweet_count])
    df = pd.DataFrame(tweet_data, columns=["Date", "User", "Text", "Likes", "Retweets"])
    return df
tweets_df = scrape_tweets("#stocks", 100)
print(tweets_df.head())


# In[4]:


import tweepy
BEARER_TOKEN = 'AAAAAAAAAAAAAAAAAAAAANZWxAEAAAAAIVhCKzMxHPW0M6baJJ4MWbHKkfA%3DsDVx34j6T3uLoToi5jvb2v7KDSnKuxRWPDRBM7mfItHmeqgqvo'
client = tweepy.Client(bearer_token=BEARER_TOKEN)
response = client.search_recent_tweets(query="stocks", max_results=100)
for tweet in response.data:
    print(tweet.text)


# In[5]:


import tweepy
import pandas as pd
import re
from nltk.corpus import stopwords
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Your Twitter API credentials
API_KEY = '0pyCK0JcXf2OES6ZVK4kEzbTX'
API_SECRET_KEY = 'Z6Q3A9eKxsUe3yzzSWcKcmPm39vQVuYylkIx2kieRv0wGFBCgW'
ACCESS_TOKEN = '1859283643769798656-DnasgKhnKJPlqTuaPmZ9RtCiLKfwXB'
ACCESS_TOKEN_SECRET = 'll6ZkdnAUjRs1by2JcIGbz0vRLlr7pX5cpUXsB6xQsz6A'

# Authenticate to Twitter
auth = tweepy.OAuth1UserHandler(consumer_key=API_KEY, 
                                consumer_secret=API_SECRET_KEY,
                                access_token=ACCESS_TOKEN,
                                access_token_secret=ACCESS_TOKEN_SECRET)
api = tweepy.API(auth)

# Function to scrape tweets based on a keyword/hashtag
def scrape_tweets(keyword, num_tweets=100):
    tweets = tweepy.Cursor(api.search_tweets, q=keyword, lang="en", tweet_mode='extended').items(num_tweets)
    tweet_data = []
    for tweet in tweets:
        tweet_data.append([tweet.created_at, tweet.user.screen_name, tweet.full_text, tweet.favorite_count, tweet.retweet_count])
    
    # Convert the tweet data into a pandas DataFrame
    df = pd.DataFrame(tweet_data, columns=["Date", "User", "Text", "Likes", "Retweets"])
    return df

# Example: Scrape 100 tweets with the hashtag #stocks
tweets_df = scrape_tweets("#stocks", 100)

# Preprocessing function to clean tweet text
def preprocess_text(text):
    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    
    # Remove special characters and digits
    text = re.sub(r"[^A-Za-z\s]", "", text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove stopwords
    stop_words = set(stopwords.words("english"))
    text = " ".join([word for word in text.split() if word not in stop_words])
    
    return text

# Apply the preprocessing function to the 'Text' column in the DataFrame
tweets_df['Cleaned_Text'] = tweets_df['Text'].apply(preprocess_text)

# Initialize VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Function to get sentiment score
def get_sentiment(text):
    sentiment = analyzer.polarity_scores(text)
    return sentiment['compound']  # The compound score is the overall sentiment

# Apply sentiment analysis to the cleaned tweet text
tweets_df['Sentiment'] = tweets_df['Cleaned_Text'].apply(get_sentiment)

# Show the structured and cleaned DataFrame
print(tweets_df[['Date', 'User', 'Text', 'Likes', 'Retweets', 'Cleaned_Text', 'Sentiment']].head())

# Optionally: Save the DataFrame to a CSV file
tweets_df.to_csv('scraped_tweets_with_sentiment.csv', index=False)


# In[5]:


import tweepy
import pandas as pd
import re
from nltk.corpus import stopwords
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Your Twitter API credentials
API_KEY = '0pyCK0JcXf2OES6ZVK4kEzbTX'
API_SECRET_KEY = 'Z6Q3A9eKxsUe3yzzSWcKcmPm39vQVuYylkIx2kieRv0wGFBCgW'
ACCESS_TOKEN = '1859283643769798656-DnasgKhnKJPlqTuaPmZ9RtCiLKfwXB'
ACCESS_TOKEN_SECRET = 'll6ZkdnAUjRs1by2JcIGbz0vRLlr7pX5cpUXsB6xQsz6A'

# Authenticate to Twitter
auth = tweepy.OAuth1UserHandler(consumer_key=API_KEY, 
                                consumer_secret=API_SECRET_KEY,
                                access_token=ACCESS_TOKEN,
                                access_token_secret=ACCESS_TOKEN_SECRET)
api = tweepy.API(auth)

# Function to scrape tweets based on a keyword/hashtag
def scrape_tweets(keyword, num_tweets=100):
    tweets = tweepy.Cursor(api.search_tweets, q=keyword, lang="en", tweet_mode='extended').items(num_tweets)
    tweet_data = []
    for tweet in tweets:
        tweet_data.append([tweet.created_at, tweet.user.screen_name, tweet.full_text, tweet.favorite_count, tweet.retweet_count])
    
    # Convert the tweet data into a pandas DataFrame
    df = pd.DataFrame(tweet_data, columns=["Date", "User", "Text", "Likes", "Retweets"])
    return df

# Example: Scrape 100 tweets with the hashtag #stocks
tweets_df = scrape_tweets("#stocks", 100)

# Preprocessing function to clean tweet text
def preprocess_text(text):
    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    
    # Remove special characters and digits
    text = re.sub(r"[^A-Za-z\s]", "", text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove stopwords
    stop_words = set(stopwords.words("english"))
    text = " ".join([word for word in text.split() if word not in stop_words])
    
    return text

# Apply the preprocessing function to the 'Text' column in the DataFrame
tweets_df['Cleaned_Text'] = tweets_df['Text'].apply(preprocess_text)

# Initialize VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Function to get sentiment score
def get_sentiment(text):
    sentiment = analyzer.polarity_scores(text)
    return sentiment['compound']  # The compound score is the overall sentiment

# Apply sentiment analysis to the cleaned tweet text
tweets_df['Sentiment'] = tweets_df['Cleaned_Text'].apply(get_sentiment)

# Show the structured and cleaned DataFrame
print(tweets_df[['Date', 'User', 'Text', 'Likes', 'Retweets', 'Cleaned_Text', 'Sentiment']].head())

# Optionally: Save the DataFrame to a CSV file
tweets_df.to_csv('scraped_tweets_with_sentiment.csv', index=False)


# In[ ]:





# In[6]:


import tweepy
import pandas as pd
import re
from nltk.corpus import stopwords
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Your Twitter API credentials
API_KEY = '0pyCK0JcXf2OES6ZVK4kEzbTX'
API_SECRET_KEY = 'Z6Q3A9eKxsUe3yzzSWcKcmPm39vQVuYylkIx2kieRv0wGFBCgW'
ACCESS_TOKEN = '1859283643769798656-DnasgKhnKJPlqTuaPmZ9RtCiLKfwXB'
ACCESS_TOKEN_SECRET = 'll6ZkdnAUjRs1by2JcIGbz0vRLlr7pX5cpUXsB6xQsz6A'

# Authenticate to Twitter
auth = tweepy.OAuth1UserHandler(consumer_key=API_KEY, 
                                consumer_secret=API_SECRET_KEY,
                                access_token=ACCESS_TOKEN,
                                access_token_secret=ACCESS_TOKEN_SECRET)
api = tweepy.API(auth)

# Function to scrape tweets based on a keyword/hashtag
def scrape_tweets(keyword, num_tweets=100):
    tweets = tweepy.Cursor(api.search_tweets, q=keyword, lang="en", tweet_mode='extended').items(num_tweets)
    tweet_data = []
    for tweet in tweets:
        tweet_data.append([tweet.created_at, tweet.user.screen_name, tweet.full_text, tweet.favorite_count, tweet.retweet_count])
    
    # Convert the tweet data into a pandas DataFrame
    df = pd.DataFrame(tweet_data, columns=["Date", "User", "Text", "Likes", "Retweets"])
    return df

# Example: Scrape 100 tweets with the hashtag #stocks
tweets_df = scrape_tweets("#stocks", 100)

# Preprocessing function to clean tweet text
def preprocess_text(text):
    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    
    # Remove special characters and digits
    text = re.sub(r"[^A-Za-z\s]", "", text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove stopwords
    stop_words = set(stopwords.words("english"))
    text = " ".join([word for word in text.split() if word not in stop_words])
    
    return text

# Apply the preprocessing function to the 'Text' column in the DataFrame
tweets_df['Cleaned_Text'] = tweets_df['Text'].apply(preprocess_text)

# Initialize VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Function to get sentiment score
def get_sentiment(text):
    sentiment = analyzer.polarity_scores(text)
    return sentiment['compound']  # The compound score is the overall sentiment

# Apply sentiment analysis to the cleaned tweet text
tweets_df['Sentiment'] = tweets_df['Cleaned_Text'].apply(get_sentiment)

# Show the structured and cleaned DataFrame
print(tweets_df[['Date', 'User', 'Text', 'Likes', 'Retweets', 'Cleaned_Text', 'Sentiment']].head())

# Optionally: Save the DataFrame to a CSV file
tweets_df.to_csv('scraped_tweets_with_sentiment.csv', index=False)


# In[7]:


import tweepy
import pandas as pd
import re
from nltk.corpus import stopwords
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Your Twitter API credentials
API_KEY = '0pyCK0JcXf2OES6ZVK4kEzbTX'
API_SECRET_KEY = 'Z6Q3A9eKxsUe3yzzSWcKcmPm39vQVuYylkIx2kieRv0wGFBCgW'
ACCESS_TOKEN = '1859283643769798656-DnasgKhnKJPlqTuaPmZ9RtCiLKfwXB'
ACCESS_TOKEN_SECRET = 'll6ZkdnAUjRs1by2JcIGbz0vRLlr7pX5cpUXsB6xQsz6A'

# Authenticate to Twitter
auth = tweepy.OAuth1UserHandler(consumer_key=API_KEY, 
                                consumer_secret=API_SECRET_KEY,
                                access_token=ACCESS_TOKEN,
                                access_token_secret=ACCESS_TOKEN_SECRET)
api = tweepy.API(auth)

# Function to scrape tweets based on a keyword/hashtag
def scrape_tweets(keyword, num_tweets=100):
    tweets = tweepy.Cursor(api.search_tweets, q=keyword, lang="en", tweet_mode='extended').items(num_tweets)
    tweet_data = []
    for tweet in tweets:
        tweet_data.append([tweet.created_at, tweet.user.screen_name, tweet.full_text, tweet.favorite_count, tweet.retweet_count])
    
    # Convert the tweet data into a pandas DataFrame
    df = pd.DataFrame(tweet_data, columns=["Date", "User", "Text", "Likes", "Retweets"])
    return df

# Example: Scrape 100 tweets with the hashtag #stocks
tweets_df = scrape_tweets("#stocks", 100)

# Preprocessing function to clean tweet text
def preprocess_text(text):
    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    
    # Remove special characters and digits
    text = re.sub(r"[^A-Za-z\s]", "", text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove stopwords
    stop_words = set(stopwords.words("english"))
    text = " ".join([word for word in text.split() if word not in stop_words])
    
    return text

# Apply the preprocessing function to the 'Text' column in the DataFrame
tweets_df['Cleaned_Text'] = tweets_df['Text'].apply(preprocess_text)

# Initialize VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Function to get sentiment score
def get_sentiment(text):
    sentiment = analyzer.polarity_scores(text)
    return sentiment['compound']  # The compound score is the overall sentiment

# Apply sentiment analysis to the cleaned tweet text
tweets_df['Sentiment'] = tweets_df['Cleaned_Text'].apply(get_sentiment)

# Show the structured and cleaned DataFrame
print(tweets_df[['Date', 'User', 'Text', 'Likes', 'Retweets', 'Cleaned_Text', 'Sentiment']].head())

# Optionally: Save the DataFrame to a CSV file
tweets_df.to_csv('scraped_tweets_with_sentiment.csv', index=False)


# In[8]:


import tweepy
import pandas as pd

# Function to scrape tweets based on a keyword/hashtag
def scrape_tweets(keyword, num_tweets=100):
    tweets = tweepy.Cursor(api.search_tweets, q=keyword, lang="en", tweet_mode='extended').items(num_tweets)
    tweet_data = []
    for tweet in tweets:
        tweet_data.append([tweet.created_at, tweet.user.screen_name, tweet.full_text, tweet.favorite_count, tweet.retweet_count])
    
    # Convert the tweet data into a pandas DataFrame
    df = pd.DataFrame(tweet_data, columns=["Date", "User", "Text", "Likes", "Retweets"])
    return df

# Example: Scrape 100 tweets with the hashtag #stocks
tweets_df = scrape_tweets("#stocks", 100)

# Show the first few rows of the scraped and structured data
print(tweets_df.head())


# In[9]:


import requests
from bs4 import BeautifulSoup

# Function to scrape tweets from a specific hashtag or keyword
def scrape_twitter_hashtag(hashtag, num_tweets=10):
    # Construct the URL for the hashtag search
    url = f"https://twitter.com/search?q={hashtag}&src=typed_query&f=live"

    # Send an HTTP GET request to the URL
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    response = requests.get(url, headers=headers)

    # Parse the content with BeautifulSoup
    soup = BeautifulSoup(response.content, 'html.parser')

    # Find the tweet containers
    tweets = soup.find_all('div', {'data-testid': 'tweet'})

    tweet_data = []
    
    # Loop through each tweet and extract relevant details
    for tweet in tweets[:num_tweets]:
        try:
            # Extract tweet text
            text = tweet.find('div', {'lang': True}).get_text()
            
            # Extract tweet user name
            user = tweet.find('div', {'dir': 'ltr'}).get_text()

            # Extract tweet date (relative time)
            date = tweet.find('time')['datetime']

            tweet_data.append([date, user, text])

        except AttributeError:
            continue
    
    return tweet_data

# Example usage: Scraping the most recent tweets with the hashtag #stocks
tweets = scrape_twitter_hashtag("#stocks", 10)

# Print the scraped tweets
for tweet in tweets:
    print(f"Date: {tweet[0]}\nUser: {tweet[1]}\nText: {tweet[2]}\n")
    print("-" * 50)


# In[10]:


import requests
from bs4 import BeautifulSoup

# Function to scrape tweets from a specific hashtag or keyword
def scrape_twitter_hashtag(hashtag, num_tweets=10):
    # Construct the URL for the hashtag search
    url = f"https://twitter.com/search?q={hashtag}&src=typed_query&f=live"

    # Send an HTTP GET request to the URL
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    response = requests.get(url, headers=headers)

    # Parse the content with BeautifulSoup
    soup = BeautifulSoup(response.content, 'html.parser')

    # Find the tweet containers
    tweets = soup.find_all('div', {'data-testid': 'tweet'})

    tweet_data = []
    
    # Loop through each tweet and extract relevant details
    for tweet in tweets[:num_tweets]:
        try:
            # Extract tweet text
            text = tweet.find('div', {'lang': True}).get_text()
            
            # Extract tweet user name
            user = tweet.find('div', {'dir': 'ltr'}).get_text()

            # Extract tweet date (relative time)
            date = tweet.find('time')['datetime']

            tweet_data.append([date, user, text])

        except AttributeError:
            continue
    
    return tweet_data

# Example usage: Scraping the most recent tweets with the hashtag #stocks
tweets = scrape_twitter_hashtag("#stocks", 10)

# Print the scraped tweets
for tweet in tweets:
    print(f"Date: {tweet[0]}\nUser: {tweet[1]}\nText: {tweet[2]}\n")
    print("-" * 50)


# In[11]:


import asyncio
import aiohttp
from bs4 import BeautifulSoup

# Function to scrape tweets using aiohttp (async requests)
async def fetch_tweet_page(session, url):
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    async with session.get(url, headers=headers) as response:
        return await response.text()

# Function to scrape tweets from a hashtag using aiohttp
async def scrape_twitter_hashtag_async(hashtag, num_tweets=10):
    url = f"https://twitter.com/search?q={hashtag}&src=typed_query&f=live"
    async with aiohttp.ClientSession() as session:
        html = await fetch_tweet_page(session, url)
        soup = BeautifulSoup(html, 'html.parser')
        
        # Find tweet containers
        tweets = soup.find_all('div', {'data-testid': 'tweet'})
        tweet_data = []
        
        # Loop through tweets and extract relevant data
        for tweet in tweets[:num_tweets]:
            try:
                # Extract tweet text, username, and date
                text = tweet.find('div', {'lang': True}).get_text()
                user = tweet.find('div', {'dir': 'ltr'}).get_text()
                date = tweet.find('time')['datetime']

                tweet_data.append([date, user, text])
            except AttributeError:
                continue

        return tweet_data

# Run the async scraping function
async def main():
    hashtag = "#stocks"
    tweets = await scrape_twitter_hashtag_async(hashtag, num_tweets=30)
    for tweet in tweets:
        print(f"Date: {tweet[0]}\nUser: {tweet[1]}\nText: {tweet[2]}\n")
        print("-" * 50)

# Run the main async function
asyncio.run(main())


# In[12]:


import asyncio
import aiohttp
from bs4 import BeautifulSoup

# Function to scrape tweets using aiohttp (async requests)
async def fetch_tweet_page(session, url):
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    async with session.get(url, headers=headers) as response:
        return await response.text()

# Function to scrape tweets from a hashtag using aiohttp
async def scrape_twitter_hashtag_async(hashtag, num_tweets=10):
    url = f"https://twitter.com/search?q={hashtag}&src=typed_query&f=live"
    async with aiohttp.ClientSession() as session:
        html = await fetch_tweet_page(session, url)
        soup = BeautifulSoup(html, 'html.parser')
        
        # Find tweet containers
        tweets = soup.find_all('div', {'data-testid': 'tweet'})
        tweet_data = []
        
        # Loop through tweets and extract relevant data
        for tweet in tweets[:num_tweets]:
            try:
                # Extract tweet text, username, and date
                text = tweet.find('div', {'lang': True}).get_text()
                user = tweet.find('div', {'dir': 'ltr'}).get_text()
                date = tweet.find('time')['datetime']

                tweet_data.append([date, user, text])
            except AttributeError:
                continue

        return tweet_data

# Run the async scraping function
async def main():
    hashtag = "#stocks"
    tweets = await scrape_twitter_hashtag_async(hashtag, num_tweets=30)
    for tweet in tweets:
        print(f"Date: {tweet[0]}\nUser: {tweet[1]}\nText: {tweet[2]}\n")
        print("-" * 50)

# Run the main async function
asyncio.run(main())


# In[13]:


import asyncio
import aiohttp
from bs4 import BeautifulSoup

# Function to scrape tweets using aiohttp (async requests)
async def fetch_tweet_page(session, url):
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    async with session.get(url, headers=headers) as response:
        return await response.text()

# Function to scrape tweets from a hashtag using aiohttp
async def scrape_twitter_hashtag_async(hashtag, num_tweets=10):
    url = f"https://twitter.com/search?q={hashtag}&src=typed_query&f=live"
    async with aiohttp.ClientSession() as session:
        html = await fetch_tweet_page(session, url)
        soup = BeautifulSoup(html, 'html.parser')
        
        # Find tweet containers
        tweets = soup.find_all('div', {'data-testid': 'tweet'})
        tweet_data = []
        
        # Loop through tweets and extract relevant data
        for tweet in tweets[:num_tweets]:
            try:
                # Extract tweet text, username, and date
                text = tweet.find('div', {'lang': True}).get_text()
                user = tweet.find('div', {'dir': 'ltr'}).get_text()
                date = tweet.find('time')['datetime']

                tweet_data.append([date, user, text])
            except AttributeError:
                continue

        return tweet_data

# Run the async scraping function
async def main():
    hashtag = "#stocks"
    tweets = await scrape_twitter_hashtag_async(hashtag, num_tweets=30)
    for tweet in tweets:
        print(f"Date: {tweet[0]}\nUser: {tweet[1]}\nText: {tweet[2]}\n")
        print("-" * 50)

# Run the main async function
asyncio.run(main())


# In[14]:


import asyncio
import aiohttp
from bs4 import BeautifulSoup

# Function to scrape tweets using aiohttp (async requests)
async def fetch_tweet_page(session, url):
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    async with session.get(url, headers=headers) as response:
        return await response.text()

# Function to scrape tweets from a hashtag using aiohttp
async def scrape_twitter_hashtag_async(hashtag, num_tweets=10):
    url = f"https://twitter.com/search?q={hashtag}&src=typed_query&f=live"
    async with aiohttp.ClientSession() as session:
        html = await fetch_tweet_page(session, url)
        soup = BeautifulSoup(html, 'html.parser')
        
        # Find tweet containers
        tweets = soup.find_all('div', {'data-testid': 'tweet'})
        tweet_data = []
        
        # Loop through tweets and extract relevant data
        for tweet in tweets[:num_tweets]:
            try:
                # Extract tweet text, username, and date
                text = tweet.find('div', {'lang': True}).get_text()
                user = tweet.find('div', {'dir': 'ltr'}).get_text()
                date = tweet.find('time')['datetime']

                tweet_data.append([date, user, text])
            except AttributeError:
                continue

        return tweet_data

# Run the async scraping function
async def main():
    hashtag = "#stocks"
    tweets = await scrape_twitter_hashtag_async(hashtag, num_tweets=30)
    for tweet in tweets:
        print(f"Date: {tweet[0]}\nUser: {tweet[1]}\nText: {tweet[2]}\n")
        print("-" * 50)

# Run the main async function
asyncio.run(main())


# In[15]:


import asyncio
import aiohttp
from bs4 import BeautifulSoup

# Function to scrape tweets using aiohttp (async requests)
async def fetch_tweet_page(session, url):
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    async with session.get(url, headers=headers) as response:
        return await response.text()

# Function to scrape tweets from a hashtag using aiohttp
async def scrape_twitter_hashtag_async(hashtag, num_tweets=10):
    url = f"https://twitter.com/search?q={hashtag}&src=typed_query&f=live"
    async with aiohttp.ClientSession() as session:
        html = await fetch_tweet_page(session, url)
        soup = BeautifulSoup(html, 'html.parser')
        
        # Find tweet containers
        tweets = soup.find_all('div', {'data-testid': 'tweet'})
        tweet_data = []
        
        # Loop through tweets and extract relevant data
        for tweet in tweets[:num_tweets]:
            try:
                # Extract tweet text, username, and date
                text = tweet.find('div', {'lang': True}).get_text()
                user = tweet.find('div', {'dir': 'ltr'}).get_text()
                date = tweet.find('time')['datetime']

                tweet_data.append([date, user, text])
            except AttributeError:
                continue

        return tweet_data

# Now directly use await instead of asyncio.run
# Example usage
hashtag = "#stocks"
tweets = await scrape_twitter_hashtag_async(hashtag, num_tweets=30)

# Print the results
for tweet in tweets:
    print(f"Date: {tweet[0]}\nUser: {tweet[1]}\nText: {tweet[2]}\n")
    print("-" * 50)


# In[16]:


import tweepy
BEARER_TOKEN = 'AAAAAAAAAAAAAAAAAAAAANZWxAEAAAAAIVhCKzMxHPW0M6baJJ4MWbHKkfA%3DsDVx34j6T3uLoToi5jvb2v7KDSnKuxRWPDRBM7mfItHmeqgqvo'
client = tweepy.Client(bearer_token=BEARER_TOKEN)
response = client.search_recent_tweets(query="stocks", max_results=100)
for tweet in response.data:
    print(tweet.text)


# In[17]:


import tweepy
import pandas as pd

# Define Bearer Token for Twitter API v2
BEARER_TOKEN = 'AAAAAAAAAAAAAAAAAAAAANZWxAEAAAAAIVhCKzMxHPW0M6baJJ4MWbHKkfA%3DsDVx34j6T3uLoToi5jvb2v7KDSnKuxRWPDRBM7mfItHmeqgqvo'

# Authenticate to Twitter API v2 using Bearer Token
client = tweepy.Client(bearer_token=BEARER_TOKEN)

# Perform a search for recent tweets containing the keyword "stocks"
response = client.search_recent_tweets(query="stocks", max_results=100)

# Create an empty list to store tweet data
tweet_data = []

# Extract and structure the tweet data
if response.data:
    for tweet in response.data:
        tweet_info = {
            "Date": tweet.created_at,  # Tweet date and time
            "User": tweet.author_id,   # User ID (to identify the author)
            "Text": tweet.text,        # The text content of the tweet
            "Tweet ID": tweet.id,      # Tweet ID
            "Like Count": tweet.public_metrics['like_count'],  # Number of likes
            "Retweet Count": tweet.public_metrics['retweet_count']  # Number of retweets
        }
        tweet_data.append(tweet_info)

# Convert the tweet data list into a pandas DataFrame
tweets_df = pd.DataFrame(tweet_data)

# Show the structured data
print(tweets_df)

# Optionally: Save the DataFrame to a CSV file for later use
tweets_df.to_csv('stocks_tweets.csv', index=False)


# In[18]:


import tweepy
import time
import pandas as pd

# Define Bearer Token for Twitter API v2
BEARER_TOKEN = 'AAAAAAAAAAAAAAAAAAAAANZWxAEAAAAAIVhCKzMxHPW0M6baJJ4MWbHKkfA%3DsDVx34j6T3uLoToi5jvb2v7KDSnKuxRWPDRBM7mfItHmeqgqvo'

# Authenticate to Twitter API v2 using Bearer Token
client = tweepy.Client(bearer_token=BEARER_TOKEN)

# Create a function to scrape tweets with rate limiting handled
def scrape_twitter_with_rate_limit(query, max_results=100, delay=15):
    tweet_data = []
    response = client.search_recent_tweets(query=query, max_results=max_results)
    
    # Check if there's data in the response
    if response.data:
        for tweet in response.data:
            tweet_info = {
                "Date": tweet.created_at,
                "User": tweet.author_id,
                "Text": tweet.text,
                "Tweet ID": tweet.id,
                "Like Count": tweet.public_metrics['like_count'],
                "Retweet Count": tweet.public_metrics['retweet_count']
            }
            tweet_data.append(tweet_info)
    
    # Check remaining requests in the current window
    remaining_requests = response.headers.get("x-rate-limit-remaining", 0)
    reset_time = response.headers.get("x-rate-limit-reset", time.time())
    
    # If there are no remaining requests, wait until the reset time
    if remaining_requests == 0:
        wait_time = int(reset_time) - int(time.time()) + 1  # Adding 1 second for safety
        print(f"Rate limit hit. Waiting for {wait_time} seconds.")
        time.sleep(wait_time)

    # Return the collected tweet data
    return tweet_data

# Example: Scrape 100 tweets with the hashtag #stocks
tweets = scrape_twitter_with_rate_limit("#stocks", 100)

# Convert to DataFrame
tweets_df = pd.DataFrame(tweets)

# Print the structured data
print(tweets_df.head())


# In[19]:


import tweepy
import time
import pandas as pd

# Define Bearer Token for Twitter API v2
BEARER_TOKEN = 'AAAAAAAAAAAAAAAAAAAAANZWxAEAAAAAIVhCKzMxHPW0M6baJJ4MWbHKkfA%3DsDVx34j6T3uLoToi5jvb2v7KDSnKuxRWPDRBM7mfItHmeqgqvo'

# Authenticate to Twitter API v2 using Bearer Token
client = tweepy.Client(bearer_token=BEARER_TOKEN)

# Function to scrape tweets while handling rate limits
def scrape_twitter_with_rate_limit(query, max_results=100):
    tweet_data = []
    try:
        response = client.search_recent_tweets(query=query, max_results=max_results)
        
        # Collect tweet data
        if response.data:
            for tweet in response.data:
                tweet_info = {
                    "Date": tweet.created_at,
                    "User": tweet.author_id,
                    "Text": tweet.text,
                    "Tweet ID": tweet.id,
                    "Like Count": tweet.public_metrics['like_count'],
                    "Retweet Count": tweet.public_metrics['retweet_count']
                }
                tweet_data.append(tweet_info)
        
        # Check remaining requests in the current window
        remaining_requests = int(response.headers.get("x-rate-limit-remaining", 0))
        reset_time = int(response.headers.get("x-rate-limit-reset", time.time()))  # Reset time
        
        # If no requests are remaining, wait until reset time
        if remaining_requests == 0:
            wait_time = reset_time - int(time.time()) + 1  # Adding 1 second to be safe
            print(f"Rate limit hit. Waiting for {wait_time} seconds until reset.")
            time.sleep(wait_time)
        
    except Exception as e:
        print(f"Error occurred: {e}")
    
    return tweet_data

# Example: Scrape 100 tweets with the hashtag #stocks
tweets = scrape_twitter_with_rate_limit("#stocks", max_results=100)

# Convert to DataFrame
tweets_df = pd.DataFrame(tweets)

# Print the structured data
print(tweets_df.head())


# In[20]:


# Example: Scraping multiple hashtags to spread the requests across different queries
hashtags = ["#stocks", "#investing", "#stockmarket", "#finance", "#trading"]

all_tweets = []

for hashtag in hashtags:
    print(f"Scraping tweets for {hashtag}...")
    tweets = scrape_twitter_with_rate_limit(hashtag, max_results=100)
    all_tweets.extend(tweets)

# Convert to DataFrame
tweets_df = pd.DataFrame(all_tweets)

# Show the results
print(tweets_df.head())


# In[22]:


import tweepy
import time
import pandas as pd

# Define Bearer Token for Twitter API v2
BEARER_TOKEN = 'AAAAAAAAAAAAAAAAAAAAANZWxAEAAAAAIVhCKzMxHPW0M6baJJ4MWbHKkfA%3DsDVx34j6T3uLoToi5jvb2v7KDSnKuxRWPDRBM7mfItHmeqgqvo'

# Authenticate to Twitter API v2 using Bearer Token
client = tweepy.Client(bearer_token=BEARER_TOKEN)

# Function to scrape tweets while handling rate limits
def scrape_twitter_with_rate_limit(query, max_results=100):
    tweet_data = []
    try:
        # Make the API request
        response = client.search_recent_tweets(query=query, max_results=max_results)
        
        # Collect tweet data
        if response.data:
            for tweet in response.data:
                tweet_info = {
                    "Date": tweet.created_at,
                    "User": tweet.author_id,
                    "Text": tweet.text,
                    "Tweet ID": tweet.id,
                    "Like Count": tweet.public_metrics['like_count'],
                    "Retweet Count": tweet.public_metrics['retweet_count']
                }
                tweet_data.append(tweet_info)
        
        # Check the rate limit headers for remaining requests and reset time
        remaining_requests = int(response.headers.get("x-rate-limit-remaining", 0))
        reset_time = int(response.headers.get("x-rate-limit-reset", time.time()))  # Reset time
        
        # If no requests are remaining, wait until the reset time
        if remaining_requests == 0:
            wait_time = reset_time - int(time.time()) + 1  # Adding 1 second for safety
            print(f"Rate limit hit. Waiting for {wait_time} seconds until reset.")
            time.sleep(wait_time)
        
    except Exception as e:
        print(f"Error occurred: {e}")
    
    return tweet_data

# Example: Scrape 100 tweets with the hashtag #stocks
tweets = scrape_twitter_with_rate_limit("#stocks", max_results=100)

# Convert to DataFrame
tweets_df = pd.DataFrame(tweets)

# Print the structured data
print(tweets_df.head())


# In[23]:


from pyrogram import Client

# Create a Pyrogram client
app = Client("my_bot", api_id="your_api_id", api_hash="your_api_hash")

# Start the client and connect
with app:
    messages = app.get_chat_history("username_or_channel_name", limit=10)
    for message in messages:
        print(message.text)


# In[25]:


from pyrogram import Client

# Create a Pyrogram client
app = Client("my_bot", api_id="29189989", api_hash="41fa49875f13cb3146e72fd2754d0130")

# Start the client and connect
with app:
    messages = app.get_chat_history("username_or_channel_name", limit=10)
    for message in messages:
        print(message.text)


# In[26]:


from telethon.sync import TelegramClient

# Replace with your own API ID and hash (from the Telegram Developer Portal)
api_id = '29189989'
api_hash = '41fa49875f13cb3146e72fd2754d0130'

# Create the client
client = TelegramClient('session_name', api_id, api_hash)

# Start the client and connect
client.start()

# Async function to scrape messages
async def scrape_channel(channel_name):
    async for message in client.iter_messages(channel_name, limit=10):
        print(message.sender_id, message.text)

# Example: Scraping a public channel or group
channel = 'username_or_channel_name'
client.loop.run_until_complete(scrape_channel(channel))


# In[27]:


from telethon.sync import TelegramClient

# Replace with your own API ID and hash (from the Telegram Developer Portal)
api_id = '29189989'
api_hash = '41fa49875f13cb3146e72fd2754d0130'

# Create the client
client = TelegramClient('session_name', api_id, api_hash)

# Start the client and connect
client.start()

# Async function to scrape messages
async def scrape_channel(channel_name):
    async for message in client.iter_messages(channel_name, limit=10):
        print(message.sender_id, message.text)

# If you're in a notebook or an environment with an active event loop, use await instead of asyncio.run()
channel = 'username_or_channel_name'
# In a notebook environment, directly use await
await scrape_channel(channel)


# In[1]:


from telethon.sync import TelegramClient

api_id = '29189989'
api_hash = '41fa49875f13cb3146e72fd2754d0130'

client = TelegramClient('session_name', api_id, api_hash)

# Start the client and connect
client.start()

# Async function to scrape messages
async def scrape_channel(channel_name):
    async for message in client.iter_messages(channel_name, limit=10):
        print(message.sender_id, message.text)

# Use await in Jupyter/Colab or loop.run_until_complete() in other environments
await scrape_channel('username_or_channel_name')

# Close the client properly
client.disconnect()


# In[2]:


from telethon.sync import TelegramClient

# Replace with your own API ID and hash (from the Telegram Developer Portal)
api_id = '29189989'
api_hash = '41fa49875f13cb3146e72fd2754d0130'

# Create the client (use a unique session name to avoid conflicts)
client = TelegramClient('unique_session_name', api_id, api_hash)

# Start the client and connect
client.start()

# Async function to scrape messages
async def scrape_channel(channel_name):
    # Scrape the most recent messages from the specified channel
    async for message in client.iter_messages(channel_name, limit=10):
        print(message.sender_id, message.text)

# Use await in Jupyter/Colab or loop.run_until_complete() in other environments
# Start scraping messages from a channel (replace with an actual channel)
channel = 'username_or_channel_name'
await scrape_channel(channel)

# After finishing the requests, disconnect the client
client.disconnect()  # Disconnecting after you're done


# In[6]:


import requests

# Your bot token (replace with your token)
bot_token = '7629043626:AAH-4XAQKkf4XMRGCkbEmGmTF3XZK-wg7OI'

# The chat ID (can be your chat ID or a group/channel ID)
chat_id = '6193264931'

# The message you want to send
message = 'Hello, world!'

# Send the message using the Telegram Bot API
url = f'https://api.telegram.org/bot{bot_token}/sendMessage'
params = {'chat_id': chat_id, 'text': message}

# Sending the request
response = requests.get(url, params=params)

# Print response
print(response.json())


# In[5]:


import requests

# Your bot token (replace with your bot's token)
bot_token = '7629043626:AAH-4XAQKkf4XMRGCkbEmGmTF3XZK-wg7OI'

# The URL to get updates (messages) from your bot
url = f'https://api.telegram.org/bot{bot_token}/getUpdates'

# Send the request to get updates
response = requests.get(url)

# Parse the response as JSON
updates = response.json()

# Print the updates (you will see the chat_id here)
print(updates)


# In[7]:


import requests

# Your bot token (replace with your actual bot token)
bot_token = '7629043626:AAH-4XAQKkf4XMRGCkbEmGmTF3XZK-wg7OI'

# URL to get updates (messages) from the bot
url = f'https://api.telegram.org/bot{bot_token}/getUpdates'

# Send request to get updates
response = requests.get(url)
data = response.json()

# Print all updates
for update in data['result']:
    print(f"Message: {update['message']['text']}")
    print(f"Chat ID: {update['message']['chat']['id']}")


# In[12]:


import praw

# Set up the Reddit client using your API credentials
reddit = praw.Reddit(
    client_id="zUJLqNK8XQqLsvONP-3g3w",  # Your client_id from Reddit
    client_secret=	"7XoR0bgARdBVgy_L5q95MutJzFLmzw",  # Your client_secret from Reddit
    user_agent="SentimentAnalyzer/1.0 by MadhavPaluru"  # A unique user agent string
)

# Test the connection by printing the authenticated user
print(f"Logged in as: {reddit.user.me()}")


# In[13]:


# Scrape a specific subreddit (e.g., 'datascience')
subreddit = reddit.subreddit('datascience')
for post in subreddit.hot(limit=5):  # Limit to 5 top posts
    print(f"Title: {post.title}")
    print(f"Score: {post.score}")
    print(f"URL: {post.url}")
    print(f"Post Text: {post.selftext}\n")


# In[14]:


import praw
import csv

# Initialize PRAW (Reddit API client)
reddit = praw.Reddit(
    client_id='zUJLqNK8XQqLsvONP-3g3w',
    client_secret='7XoR0bgARdBVgy_L5q95MutJzFLmzw',
    user_agent='SentimentAnalyzer/1.0 by MadhavPaluru'
)

# Define the subreddit and the keywords to search for
subreddit = reddit.subreddit('datascience')  # Change to your target subreddit
keywords = ['AI', 'Machine Learning', 'Data Science']

# Open CSV file to write the data
with open('reddit_data.csv', mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    # Write header row
    writer.writerow(['Title', 'Score', 'URL', 'Post Text', 'Date', 'Keyword', 'Comments'])

    # Scrape posts that match the keywords
    for keyword in keywords:
        for post in subreddit.search(keyword, limit=10):  # Adjust the limit as needed
            post_data = [post.title, post.score, post.url, post.selftext, post.created_utc, keyword]
            
            # Scrape comments for sentiment analysis
            comments = []
            post.comments.replace_more(limit=0)  # Replace 'more comments' objects
            for comment in post.comments.list():
                comments.append(comment.body)
            
            # Join all comments with a separator (e.g., " || ")
            comments_text = " || ".join(comments)
            
            # Write post data and comments to CSV
            writer.writerow(post_data + [comments_text])

print("Data has been written to 'reddit_data.csv' with comments.")


# In[15]:


import os

# Print the current working directory
print(f"File saved at: {os.path.abspath('reddit_data.csv')}")



# In[17]:


import praw
import csv

# Initialize PRAW (Reddit API client)
reddit = praw.Reddit(
    client_id='zUJLqNK8XQqLsvONP-3g3w',
    client_secret='7XoR0bgARdBVgy_L5q95MutJzFLmzw',
    user_agent='SentimentAnalyzer/1.0 by MadhavPaluru'
)

# Define the subreddit and the keywords to search for
subreddit = reddit.subreddit('datascience')  # Change to your target subreddit
keywords = ['Stocks', 'Machine Learning','predictions','stock price trends']

# Open CSV file in append mode
with open('reddit_data.csv', mode='a', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    
    # Skip writing the header row again if the file already has data
    # (Check if the file is empty by reading the first line)
    file.seek(0, 0)
    first_line = file.readline()
    if not first_line:
        # If file is empty, write the header row
        writer.writerow(['Title', 'Score', 'URL', 'Post Text', 'Date', 'Keyword', 'Comments'])
    
    # Scrape posts that match the keywords
    for keyword in keywords:
        for post in subreddit.search(keyword, limit=100):  # Adjust the limit as needed
            post_data = [post.title, post.score, post.url, post.selftext, post.created_utc, keyword]
            
            # Scrape comments for sentiment analysis
            comments = []
            post.comments.replace_more(limit=0)  # Replace 'more comments' objects
            for comment in post.comments.list():
                comments.append(comment.body)
            
            # Join all comments with a separator (e.g., " || ")
            comments_text = " || ".join(comments)
            
            # Write post data and comments to CSV
            writer.writerow(post_data + [comments_text])

print("New data has been appended to 'reddit_data.csv' with comments.")


# In[2]:


import praw
import csv
import os

# Initialize PRAW (Reddit API client)
reddit = praw.Reddit(
    client_id='zUJLqNK8XQqLsvONP-3g3w',
    client_secret='7XoR0bgARdBVgy_L5q95MutJzFLmzw',
    user_agent='SentimentAnalyzer/1.0 by MadhavPaluru'
)

# Define the subreddit and the keywords to search for
subreddit = reddit.subreddit('wallstreetbets+investing+stocks+pennystocks+options')  # Change to your target subreddit
keywords = ['AAPL', 'TSLA', 'bullish', 'bearish', 'earnings', 'market', 'growth stock', 'dividends', 'options trading']

# Check if the file exists and is not empty
file_exists = os.path.exists('reddit_data.csv')

# Open the CSV file in append mode
with open('reddit_data.csv', mode='a', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    
    # If the file does not exist or is empty, write the header row
    if not file_exists or file.tell() == 0:
        writer.writerow(['Title', 'Score', 'URL', 'Post Text', 'Date', 'Keyword', 'Comments'])
    
    # Scrape posts that match the keywords
    for keyword in keywords:
        for post in subreddit.search(keyword, limit=100):  # Adjust the limit as needed
            post_data = [post.title, post.score, post.url, post.selftext, post.created_utc, keyword]
            
            # Scrape comments for sentiment analysis
            comments = []
            post.comments.replace_more(limit=0)  # Replace 'more comments' objects
            for comment in post.comments.list():
                comments.append(comment.body)
            
            # Join all comments with a separator (e.g., " || ")
            comments_text = " || ".join(comments)
            
            # Write post data and comments to CSV
            writer.writerow(post_data + [comments_text])

print("New data has been appended to 'reddit_data.csv' with comments.")


# In[19]:


import os

# Print the current working directory
print(f"File saved at: {os.path.abspath('reddit_data.csv')}")


# In[21]:


import pandas as pd
df=pd.read_csv(r'C:\Users\hp\reddit_data.csv')
df


# In[ ]:


# Install necessary libraries
get_ipython().system('pip install praw vaderSentiment pandas')

# Import required libraries
import praw
import csv
import os
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd

# Initialize Reddit API client
reddit = praw.Reddit(
    client_id='zUJLqNK8XQqLsvONP-3g3w',         # Replace with your Reddit client ID
    client_secret='7XoR0bgARdBVgy_L5q95MutJzFLmzw', # Replace with your Reddit client secret
    user_agent='SentimentAnalyzer/1.0 by MadhavPaluru'  # Replace with your app's user agent
)

# Initialize VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Define the target subreddits and keywords
subreddit = reddit.subreddit('wallstreetbets+investing+stocks+pennystocks+options')  # Multiple subreddits
keywords = ['AAPL', 'TSLA', 'bullish', 'bearish', 'earnings', 'market', 'growth stock', 'dividends', 'options trading']

# Check if the CSV file already exists
file_path = 'reddit_data.csv'
file_exists = os.path.exists(file_path)

# Open CSV file for appending
with open(file_path, mode='a', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    
    # Write the header row if the file doesn't exist
    if not file_exists:
        writer.writerow(['Title', 'Score', 'URL', 'Post Text', 'Date', 'Keyword', 'Comments', 'Post Sentiment', 'Comments Sentiment'])
    
    # Scrape data for each keyword
    for keyword in keywords:
        print(f"Scraping posts for keyword: {keyword}")
        for post in subreddit.search(keyword, limit=100):  # Adjust the limit as needed
            post_data = [post.title, post.score, post.url, post.selftext, post.created_utc, keyword]
            
            # Analyze sentiment of the post text
            post_sentiment_score = analyzer.polarity_scores(post.selftext)['compound']
            
            # Scrape and analyze comments
            comments = []
            post.comments.replace_more(limit=0)  # Expand comments
            for comment in post.comments.list():
                comments.append(comment.body)
            
            # Combine comments into a single string
            comments_text = " || ".join(comments)
            
            # Analyze sentiment of the comments
            comments_sentiment_score = analyzer.polarity_scores(comments_text)['compound']
            
            # Write data to CSV
            writer.writerow(post_data + [comments_text, post_sentiment_score, comments_sentiment_score])

print(f"Data scraping completed. Data has been saved to {file_path}")

# Load the scraped data into a Pandas DataFrame for preview
df = pd.read_csv(file_path)
print("Preview of scraped data:")
print(df.head())


# In[ ]:





# In[ ]:


# Install necessary libraries
get_ipython().system('pip install praw pandas vaderSentiment')

# Import required libraries
import praw
import csv
import os
import time
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Initialize Reddit API client
reddit = praw.Reddit(
    client_id='zUJLqNK8XQqLsvONP-3g3w',          # Replace with your Reddit client ID
    client_secret='7XoR0bgARdBVgy_L5q95MutJzFLmzw',  # Replace with your Reddit client secret
    user_agent='SentimentAnalyzer/1.0 by MadhavPaluru'   # Replace with your app's user agent
)

# Initialize VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Define parameters
subreddits = ['wallstreetbets', 'stocks', 'investing', 'pennystocks', 'cryptocurrency']
keywords = ['AAPL', 'TSLA', 'bullish', 'bearish', 'growth stock', 'options trading']
posts_per_keyword = 50  # Limit the number of posts per keyword to reduce time
output_file = 'reddit_data_optimized.csv'

# Check if the file exists
file_exists = os.path.exists(output_file)

# Open CSV file for appending
with open(output_file, mode='a', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)

    # Write header if the file doesn't already exist
    if not file_exists:
        writer.writerow([
            'Title', 'Score', 'URL', 'Post Text', 'Date', 'Keyword',
            'Top Comments', 'Post Sentiment', 'Comments Sentiment'
        ])
    
    # Iterate through each subreddit and keyword
    for subreddit_name in subreddits:
        subreddit = reddit.subreddit(subreddit_name)
        for keyword in keywords:
            print(f"Scraping '{keyword}' from r/{subreddit_name}...")
            start_time = time.time()

            # Fetch posts matching the keyword
            for post in subreddit.search(keyword, limit=posts_per_keyword):
                try:
                    # Basic post details
                    post_data = [
                        post.title,           # Title of the post
                        post.score,           # Post score
                        post.url,             # Post URL
                        post.selftext[:500],  # First 500 characters of the post text (to save space)
                        post.created_utc,     # Post creation date (in UTC timestamp)
                        keyword               # Keyword that matched
                    ]

                    # Fetch top comments (limit to 5 to reduce time)
                    post.comments.replace_more(limit=0)
                    top_comments = [comment.body for comment in post.comments[:5]]
                    top_comments_text = " || ".join(top_comments)

                    # Perform sentiment analysis
                    post_sentiment = analyzer.polarity_scores(post.selftext)['compound']
                    comments_sentiment = analyzer.polarity_scores(top_comments_text)['compound']

                    # Add sentiment and comments to post data
                    post_data += [top_comments_text[:1000], post_sentiment, comments_sentiment]

                    # Write to CSV
                    writer.writerow(post_data)

                except Exception as e:
                    print(f"Error processing post: {e}")
            
            print(f"Finished scraping '{keyword}' in r/{subreddit_name} ({time.time() - start_time:.2f}s)")

print(f"Scraping completed. Data saved to {output_file}.")


# In[3]:


# Import required libraries
import praw
import csv
import os
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Initialize Reddit API client
reddit = praw.Reddit(
    client_id='zUJLqNK8XQqLsvONP-3g3w',          # Replace with your Reddit client ID
    client_secret='7XoR0bgARdBVgy_L5q95MutJzFLmzw',  # Replace with your Reddit client secret
    user_agent='SentimentAnalyzer/1.0 by MadhavPaluru'   # Replace with your app's user agent
)

# Initialize VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Define parameters
subreddits = ['wallstreetbets', 'stocks']  # Example subreddits
keywords = ['AAPL', 'TSLA']  # Example keywords
posts_per_keyword = 10  # Fetch only 10 posts per keyword
output_file = 'reddit_data_minimal.csv'

# Check if the file exists
file_exists = os.path.exists(output_file)

# Function to scrape and append data
def scrape_and_append():
    with open(output_file, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)

        # Write header if the file doesn't already exist
        if not file_exists:
            writer.writerow([
                'Title', 'Score', 'URL', 'Post Text', 'Date', 'Keyword',
                'Top Comments', 'Post Sentiment', 'Comments Sentiment'
            ])
        
        # Iterate through each subreddit and keyword
        for subreddit_name in subreddits:
            subreddit = reddit.subreddit(subreddit_name)
            for keyword in keywords:
                print(f"Scraping '{keyword}' from r/{subreddit_name}...")

                # Fetch posts matching the keyword
                for post in subreddit.search(keyword, limit=posts_per_keyword):
                    try:
                        # Extract basic post details
                        post_data = [
                            post.title,           # Title of the post
                            post.score,           # Post score
                            post.url,             # Post URL
                            post.selftext[:500],  # First 500 characters of the post text
                            post.created_utc,     # Post creation date
                            keyword               # Keyword that matched
                        ]

                        # Fetch top comments (limit to 2 for faster processing)
                        post.comments.replace_more(limit=0)
                        top_comments = [comment.body for comment in post.comments[:2]]
                        top_comments_text = " || ".join(top_comments)

                        # Perform sentiment analysis
                        post_sentiment = analyzer.polarity_scores(post.selftext)['compound']
                        comments_sentiment = analyzer.polarity_scores(top_comments_text)['compound']

                        # Add sentiment and comments to post data
                        post_data += [top_comments_text[:500], post_sentiment, comments_sentiment]

                        # Write to CSV
                        writer.writerow(post_data)

                        print(f"Post '{post.title[:30]}...' scraped and appended.")

                    except Exception as e:
                        print(f"Error processing post: {e}")

# Run the function
scrape_and_append()

print(f"Scraping completed. Data appended to {output_file}.")


# In[4]:


import os

# Print the current working directory
print(f"File saved at: {os.path.abspath('reddit_data_minimal.csv')}")


# In[5]:


import pandas as pd
df=pd.read_csv(r'C:\Users\hp\reddit_data_minimal.csv')
df


# In[6]:


# Import required libraries
import praw
import csv
import os
import time
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Initialize Reddit API client
reddit = praw.Reddit(
    client_id='zUJLqNK8XQqLsvONP-3g3w',          # Replace with your Reddit client ID
    client_secret='7XoR0bgARdBVgy_L5q95MutJzFLmzw',  # Replace with your Reddit client secret
    user_agent='SentimentAnalyzer/1.0 by MadhavPaluru'   # Replace with your app's user agent
)

# Initialize VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Define parameters
subreddits = [
    'wallstreetbets', 'stocks', 'investing', 'StockMarket', 'options'
]
keywords = [
    'AAPL', 'TSLA', 'MSFT', 'AMZN', 'GOOGL', 'bull market', 
    'bear market', 'stock predictions', 'market crash', 'earnings report'
]
posts_per_batch = 100  # Posts per batch
total_batches = 5  # Number of batches to fetch
output_file = 'reddit_stock_data.csv'

# Check if the file exists
file_exists = os.path.exists(output_file)

# Function to scrape and append data
def scrape_and_append(batch_number):
    with open(output_file, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)

        # Write header if the file doesn't already exist
        if not file_exists and batch_number == 1:
            writer.writerow([
                'Title', 'Score', 'URL', 'Post Text', 'Date', 'Keyword',
                'Top Comments', 'Post Sentiment', 'Comments Sentiment'
            ])
        
        post_count = 0

        # Iterate through each subreddit and keyword
        for subreddit_name in subreddits:
            subreddit = reddit.subreddit(subreddit_name)
            for keyword in keywords:
                print(f"Batch {batch_number}: Scraping '{keyword}' from r/{subreddit_name}...")

                # Fetch posts matching the keyword
                for post in subreddit.search(keyword, limit=posts_per_batch):
                    try:
                        if post_count >= posts_per_batch:
                            break

                        # Extract basic post details
                        post_data = [
                            post.title,           # Title of the post
                            post.score,           # Post score
                            post.url,             # Post URL
                            post.selftext[:500],  # First 500 characters of the post text
                            post.created_utc,     # Post creation date
                            keyword               # Keyword that matched
                        ]

                        # Fetch top comments (limit to 3 for speed)
                        post.comments.replace_more(limit=0)
                        top_comments = [comment.body for comment in post.comments[:3]]
                        top_comments_text = " || ".join(top_comments)

                        # Perform sentiment analysis
                        post_sentiment = analyzer.polarity_scores(post.selftext)['compound']
                        comments_sentiment = analyzer.polarity_scores(top_comments_text)['compound']

                        # Add sentiment and comments to post data
                        post_data += [top_comments_text[:500], post_sentiment, comments_sentiment]

                        # Write to CSV
                        writer.writerow(post_data)
                        post_count += 1

                        print(f"Post '{post.title[:30]}...' appended ({post_count}/{posts_per_batch}).")

                    except Exception as e:
                        print(f"Error processing post: {e}")

                if post_count >= posts_per_batch:
                    break

        print(f"Batch {batch_number} completed. {post_count} posts appended.")

# Run the scraping process for multiple batches
for batch_number in range(1, total_batches + 1):
    scrape_and_append(batch_number)
    print(f"Waiting before next batch...")
    time.sleep(5)  # Delay between batches to avoid rate-limiting

print(f"Scraping process completed. Data saved to {output_file}.")


# In[7]:


import os

# Print the current working directory
print(f"File saved at: {os.path.abspath('reddit_stock_data.csv')}")



# In[9]:


import pandas as pd
df= pd.read_csv(r'C:\Users\hp\reddit_stock_data.csv')
df


# In[10]:


import pandas as pd

# Load the dataset
file_path = 'reddit_stock_data.csv'  # Update this if your file is named differently
df = pd.read_csv(file_path)

# Display the first few rows
df.head()


# In[11]:


# Check basic info
df.info()

# Check for duplicates
print("Number of duplicates:", df.duplicated().sum())

# Check for missing values
print("Missing values per column:\n", df.isnull().sum())

# Summary statistics for numerical columns
df.describe()


# In[12]:


# Drop rows where critical columns are missing
df = df.dropna(subset=['Title', 'Post Text', 'Top Comments'])

# Fill missing sentiment scores with 0
df['Post Sentiment'] = df['Post Sentiment'].fillna(0)
df['Comments Sentiment'] = df['Comments Sentiment'].fillna(0)


# In[13]:


df = df.drop_duplicates()


# In[14]:


import re

def clean_text(text):
    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    # Remove special characters
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Clean Post Text and Comments
df['Post Text'] = df['Post Text'].apply(clean_text)
df['Top Comments'] = df['Top Comments'].apply(clean_text)


# In[15]:


from datetime import datetime

# Convert Date from UNIX timestamp to readable format
df['Date'] = pd.to_datetime(df['Date'], unit='s')

# Sort by date
df = df.sort_values(by='Date')


# In[16]:


# Save the cleaned DataFrame to a new CSV file
cleaned_file_path = 'cleaned_reddit_stock_data.csv'
df.to_csv(cleaned_file_path, index=False)
print(f"Cleaned data saved to {cleaned_file_path}")


# In[17]:


import tweepy
import pandas as pd

# Set up your Twitter API keys here
consumer_key = 'TFJpR0RRNnVLdVJMWkVHQkI1Uks6MTpjaQ'
consumer_secret = '0rTkBg6lRsIX4i8bDQkRaGfEn4uOZqys3tNdr7UNjIreIFesCU'
access_token = '1859283643769798656-qG27kS4iriz5LYAJL2JVKDfWSWB5GA'
access_token_secret = 'Bstj2QE2X118sE4cJcdCCogt9gFLAzshfEBD7dgIcChMr'

# Authenticate to Twitter
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)

# Define the keywords/hashtags to search for
keywords = ['#stocks', '#stockmarket', '#investing', '#trading', '#stockprediction']

# Define how many tweets you want to scrape
tweet_count = 100  # Adjust as needed

# Create an empty list to store tweet data
tweets_data = []

# Scraping tweets for each keyword
for keyword in keywords:
    for tweet in tweepy.Cursor(api.search, q=keyword, lang="en", tweet_mode="extended").items(tweet_count):
        tweet_data = {
            'Date': tweet.created_at,
            'User': tweet.user.screen_name,
            'Text': tweet.full_text,
            'Sentiment': 0  # You can add sentiment analysis later
        }
        tweets_data.append(tweet_data)

# Convert the data into a DataFrame
tweets_df = pd.DataFrame(tweets_data)

# Save the data to a CSV file
tweets_df.to_csv('twitter_stock_data.csv', index=False)
print(f"Scraped {len(tweets_df)} tweets related to stock market.")


# In[18]:


import tweepy
import pandas as pd

# Set up your Twitter API keys here
consumer_key = 'TFJpR0RRNnVLdVJMWkVHQkI1Uks6MTpjaQ'
consumer_secret = '0rTkBg6lRsIX4i8bDQkRaGfEn4uOZqys3tNdr7UNjIreIFesCU'
access_token = '1859283643769798656-qG27kS4iriz5LYAJL2JVKDfWSWB5GA'
access_token_secret = 'Bstj2QE2X118sE4cJcdCCogt9gFLAzshfEBD7dgIcChMr'

# Authenticate to Twitter
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)

# Define the keywords/hashtags to search for
keywords = ['#stocks', '#stockmarket', '#investing', '#trading', '#stockprediction']

# Define how many tweets you want to scrape
tweet_count = 100  # Adjust as needed

# Create an empty list to store tweet data
tweets_data = []

# Scraping tweets for each keyword
for keyword in keywords:
    for tweet in tweepy.Cursor(api.search, q=keyword, lang="en", tweet_mode="extended").items(tweet_count):
        tweet_data = {
            'Date': tweet.created_at,
            'User': tweet.user.screen_name,
            'Text': tweet.full_text,
            'Sentiment': 0  # You can add sentiment analysis later
        }
        tweets_data.append(tweet_data)

# Convert the data into a DataFrame
tweets_df = pd.DataFrame(tweets_data)

# Save the data to a CSV file
tweets_df.to_csv('twitter_stock_data.csv', index=False)
print(f"Scraped {len(tweets_df)} tweets related to stock market.")


# In[19]:


import tweepy
import pandas as pd

# Set up your Twitter API keys here
consumer_key = 'TFJpR0RRNnVLdVJMWkVHQkI1Uks6MTpjaQ'
consumer_secret = '0rTkBg6lRsIX4i8bDQkRaGfEn4uOZqys3tNdr7UNjIreIFesCU'
access_token = '1859283643769798656-qG27kS4iriz5LYAJL2JVKDfWSWB5GA'
access_token_secret = 'Bstj2QE2X118sE4cJcdCCogt9gFLAzshfEBD7dgIcChMr'
# Authenticate to Twitter using Tweepy v4.x Client
client = tweepy.Client(consumer_key=consumer_key, consumer_secret=consumer_secret,
                       access_token=access_token, access_token_secret=access_token_secret)

# Define the keywords/hashtags to search for
keywords = ['#stocks', '#stockmarket', '#investing', '#trading', '#stockprediction']

# Define how many tweets you want to scrape
tweet_count = 100  # Adjust as needed

# Create an empty list to store tweet data
tweets_data = []

# Scraping tweets for each keyword
for keyword in keywords:
    # Search for recent tweets containing the keyword
    tweets = client.search_recent_tweets(query=keyword, max_results=tweet_count, tweet_fields=['created_at', 'author_id', 'text'])

    for tweet in tweets.data:
        tweet_data = {
            'Date': tweet.created_at,
            'User ID': tweet.author_id,
            'Text': tweet.text,
            'Sentiment': 0  # You can add sentiment analysis later
        }
        tweets_data.append(tweet_data)

# Convert the data into a DataFrame
tweets_df = pd.DataFrame(tweets_data)

# Save the data to a CSV file
tweets_df.to_csv('twitter_stock_data.csv', index=False)
print(f"Scraped {len(tweets_df)} tweets related to stock market.")


# In[20]:


import tweepy
import pandas as pd

# Your Twitter API credentials (replace with your actual credentials)
consumer_key = 'TFJpR0RRNnVLdVJMWkVHQkI1Uks6MTpjaQ'
consumer_secret = '0rTkBg6lRsIX4i8bDQkRaGfEn4uOZqys3tNdr7UNjIreIFesCU'
access_token = '1859283643769798656-qG27kS4iriz5LYAJL2JVKDfWSWB5GA'
access_token_secret = 'Bstj2QE2X118sE4cJcdCCogt9gFLAzshfEBD7dgIcChMr'

# Authenticate using Tweepy with Essential Access
client = tweepy.Client(consumer_key=consumer_key, consumer_secret=consumer_secret,
                       access_token=access_token, access_token_secret=access_token_secret)

# Define the keywords/hashtags to search for (relevant to stock predictions)
keywords = ['#stocks', '#stockmarket', '#investing', '#trading', '#stockprediction']

# Define how many tweets you want to scrape
tweet_count = 100  # Adjust as needed

# Create an empty list to store tweet data
tweets_data = []

# Scraping tweets for each keyword
for keyword in keywords:
    # Search for recent tweets containing the keyword
    tweets = client.search_recent_tweets(query=keyword, max_results=tweet_count, tweet_fields=['created_at', 'author_id', 'text'])

    # Loop through the tweets and store the relevant data
    for tweet in tweets.data:
        tweet_data = {
            'Date': tweet.created_at,
            'User ID': tweet.author_id,
            'Text': tweet.text,
            'Sentiment': 0  # Placeholder for sentiment analysis
        }
        tweets_data.append(tweet_data)

# Convert the data into a DataFrame
tweets_df = pd.DataFrame(tweets_data)

# Save the data to a CSV file
tweets_df.to_csv('twitter_stock_data_essential_access.csv', index=False)
print(f"Scraped {len(tweets_df)} tweets related to stock market using Essential Access.")


# In[21]:


import tweepy
import pandas as pd

# Your Twitter API credentials (replace with your actual credentials)
consumer_key = 'TFJpR0RRNnVLdVJMWkVHQkI1Uks6MTpjaQ'
consumer_secret = '0rTkBg6lRsIX4i8bDQkRaGfEn4uOZqys3tNdr7UNjIreIFesCU'
access_token = '1859283643769798656-qG27kS4iriz5LYAJL2JVKDfWSWB5GA'
access_token_secret = 'Bstj2QE2X118sE4cJcdCCogt9gFLAzshfEBD7dgIcChMr'

# Authenticate using OAuth1 for V1.1 access
auth = tweepy.OAuth1UserHandler(consumer_key, consumer_secret, access_token, access_token_secret)
api = tweepy.API(auth)

# Test if authentication is successful
try:
    api.verify_credentials()
    print("Authentication successful")
except tweepy.TweepError as e:
    print("Authentication failed:", e)

# Define the keywords/hashtags to search for (relevant to stock predictions)
keywords = ['#stocks', '#stockmarket', '#investing', '#trading', '#stockprediction']

# Define how many tweets you want to scrape
tweet_count = 100  # Adjust as needed

# Create an empty list to store tweet data
tweets_data = []

# Scraping tweets for each keyword
for keyword in keywords:
    # Search for recent tweets containing the keyword
    tweets = api.search(q=keyword, count=tweet_count, tweet_mode='extended')

    for tweet in tweets:
        tweet_data = {
            'Date': tweet.created_at,
            'User': tweet.user.screen_name,
            'Text': tweet.full_text,
            'Sentiment': 0  # Placeholder for sentiment analysis
        }
        tweets_data.append(tweet_data)

# Convert the data into a DataFrame
tweets_df = pd.DataFrame(tweets_data)

# Save the data to a CSV file
tweets_df.to_csv('twitter_stock_data_v1.1.csv', index=False)
print(f"Scraped {len(tweets_df)} tweets related to stock market.")


# In[23]:


import tweepy

# Your Twitter API credentials (replace with your actual credentials)
consumer_key = 'TFJpR0RRNnVLdVJMWkVHQkI1Uks6MTpjaQ'
consumer_secret = '0rTkBg6lRsIX4i8bDQkRaGfEn4uOZqys3tNdr7UNjIreIFesCU'
access_token = '1859283643769798656-qG27kS4iriz5LYAJL2JVKDfWSWB5GA'
access_token_secret = 'Bstj2QE2X118sE4cJcdCCogt9gFLAzshfEBD7dgIcChMr'

# Authenticate using OAuth1 for V1.1 access
auth = tweepy.OAuth1UserHandler(consumer_key, consumer_secret, access_token, access_token_secret)

# Create an API object to interact with Twitter
api = tweepy.API(auth)

# Verify authentication
try:
    api.verify_credentials()  # This checks if the credentials are valid
    print("Authentication successful!")
except tweepy.TweepError as e:
    print(f"Authentication failed: {e}")


# In[25]:


import tweepy

consumer_key = 'FB9QmT7SYQT3guSK5ISDTTrvO'
consumer_secret = 'vR1HZKzsOQRTBnsaQW8G0jIreGYCUOr3tAaurtgxRSqjwqBiYH'
access_token = '1859283643769798656-9GRFrMdvPfmmuoNvcet8SKhYxWhUaZ'
access_token_secret = 'b0zsQ8zAZ9BDMWvA6Sc3QCDARFKz3k2DnSoV23hVXIW6i'

# Authenticate using OAuth1 for V1.1 access
auth = tweepy.OAuth1UserHandler(consumer_key, consumer_secret, access_token, access_token_secret)

# Create an API object to interact with Twitter
api = tweepy.API(auth)

# Verify authentication
try:
    api.verify_credentials()  # This checks if the credentials are valid
    print("Authentication successful!")
except tweepy.TweepyException as e:  # Catch the general TweepyException for v4.x
    print(f"Authentication failed: {e}")


# In[26]:


import tweepy
import pandas as pd

# Your Twitter API credentials (replace with your actual credentials)
consumer_key = 'FB9QmT7SYQT3guSK5ISDTTrvO'
consumer_secret = 'vR1HZKzsOQRTBnsaQW8G0jIreGYCUOr3tAaurtgxRSqjwqBiYH'
access_token = '1859283643769798656-9GRFrMdvPfmmuoNvcet8SKhYxWhUaZ'
access_token_secret = 'b0zsQ8zAZ9BDMWvA6Sc3QCDARFKz3k2DnSoV23hVXIW6i'

# Authenticate using OAuth1 for V1.1 access
auth = tweepy.OAuth1UserHandler(consumer_key, consumer_secret, access_token, access_token_secret)
api = tweepy.API(auth)

# Test if authentication is successful
try:
    api.verify_credentials()
    print("Authentication successful")
except tweepy.TweepError as e:
    print("Authentication failed:", e)

# Define the keywords/hashtags to search for (relevant to stock predictions)
keywords = ['#stocks', '#stockmarket', '#investing', '#trading', '#stockprediction']

# Define how many tweets you want to scrape
tweet_count = 100  # Adjust as needed

# Create an empty list to store tweet data
tweets_data = []

# Scraping tweets for each keyword
for keyword in keywords:
    # Search for recent tweets containing the keyword
    tweets = api.search(q=keyword, count=tweet_count, tweet_mode='extended')

    for tweet in tweets:
        tweet_data = {
            'Date': tweet.created_at,
            'User': tweet.user.screen_name,
            'Text': tweet.full_text,
            'Sentiment': 0  # Placeholder for sentiment analysis
        }
        tweets_data.append(tweet_data)

# Convert the data into a DataFrame
tweets_df = pd.DataFrame(tweets_data)

# Save the data to a CSV file
tweets_df.to_csv('twitter_stock_data_v1.1.csv', index=False)
print(f"Scraped {len(tweets_df)} tweets related to stock market.")


# In[27]:


import tweepy
import pandas as pd

# Your Twitter API credentials (replace with your actual credentials)
bearer_token = 'AAAAAAAAAAAAAAAAAAAAANZWxAEAAAAAZwcGvC8ZkwcVXtPSyjgUwdoIuNg%3DwcqjfEFjRfYkzhNwOWABCx3U1ujIo1GH7F4njv1G2LCb0xSS0j'  # You can use Bearer Token for Tweepy v4.x

# Authenticate using the Bearer Token (for v4.x access)
client = tweepy.Client(bearer_token=bearer_token)

# Define the keywords/hashtags to search for (relevant to stock predictions)
keywords = ['#stocks', '#stockmarket', '#investing', '#trading', '#stockprediction']

# Define how many tweets you want to scrape
tweet_count = 100  # Adjust as needed

# Create an empty list to store tweet data
tweets_data = []

# Scraping tweets for each keyword
for keyword in keywords:
    # Search for recent tweets containing the keyword
    tweets = client.search_recent_tweets(query=keyword, max_results=tweet_count)

    for tweet in tweets.data:
        tweet_data = {
            'Date': tweet.created_at,
            'User': tweet.author_id,  # You can also retrieve more info like screen_name if needed
            'Text': tweet.text,
            'Sentiment': 0  # Placeholder for sentiment analysis
        }
        tweets_data.append(tweet_data)

# Convert the data into a DataFrame
tweets_df = pd.DataFrame(tweets_data)

# Save the data to a CSV file
tweets_df.to_csv('twitter_stock_data_v4.csv', index=False)
print(f"Scraped {len(tweets_df)} tweets related to stock market.")


# In[28]:


import requests

symbol = 'AAPL'  # Stock symbol (e.g., Apple)
url = f'https://api.stocktwits.com/api/2/streams/symbol/{symbol}.json'

response = requests.get(url)
data = response.json()
print(data)


# In[29]:


import requests

# Example URL for StockTwits (replace with your actual endpoint)
url = 'https://api.stocktwits.com/api/2/streams/symbol/AAPL.json'
response = requests.get(url)

# Print the raw response content
print(response.text)

# Try decoding the JSON only if the response is valid
if response.status_code == 200:
    try:
        data = response.json()
        print(data)
    except ValueError as e:
        print(f"Error decoding JSON: {e}")
else:
    print(f"Failed to fetch data. Status code: {response.status_code}")


# In[30]:


if response.status_code == 200:
    try:
        data = response.json()
        if data:  # Check if data is not empty
            print(data)
        else:
            print("Empty response")
    except ValueError as e:
        print(f"Error decoding JSON: {e}")
else:
    print(f"Failed to fetch data. Status code: {response.status_code}")


# In[31]:


rate_limit_remaining = response.headers.get('X-RateLimit-Remaining')
print(f"Remaining API calls: {rate_limit_remaining}")


# In[32]:


import time

url = 'https://api.stocktwits.com/api/2/streams/symbol/AAPL.json'
response = requests.get(url)

if response.status_code == 429:
    print("Rate limit exceeded. Sleeping for 60 seconds.")
    time.sleep(60)  # Wait for 60 seconds before retrying
    response = requests.get(url)  # Retry the request

if response.status_code == 200:
    data = response.json()
    print(data)
else:
    print(f"Failed to fetch data. Status code: {response.status_code}")


# In[33]:


import time
import random

def get_data_with_backoff(url):
    retries = 0
    while retries < 5:
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 429:
            print(f"Rate limit exceeded. Retrying in {2 ** retries} seconds...")
            time.sleep(2 ** retries + random.random())  # Exponential backoff with jitter
            retries += 1
        else:
            print(f"Failed to fetch data. Status code: {response.status_code}")
            break
    return None

url = 'https://api.stocktwits.com/api/2/streams/symbol/AAPL.json'
data = get_data_with_backoff(url)
if data:
    print(data)


# In[34]:


import requests

url = 'https://api.stocktwits.com/api/2/streams/symbol/AAPL.json'

headers = {
    'Authorization': 'Bearer YOUR_BEARER_TOKEN'
}

response = requests.get(url, headers=headers)

if response.status_code == 200:
    data = response.json()
    print(data)
elif response.status_code == 403:
    print("Access Forbidden (403): You don't have permission to access this resource.")
    print("Response body:", response.text)  # Check response body for more details
else:
    print(f"Failed to fetch data. Status code: {response.status_code}")


# In[35]:


import requests
import pandas as pd

# Your Alpha Vantage API key
api_key = 'BEAJN61YVAP411BR'

# Function to fetch daily stock data
def get_stock_data(symbol, api_key):
    url = f'https://www.alphavantage.co/query'
    params = {
        'function': 'TIME_SERIES_DAILY',
        'symbol': symbol,
        'apikey': api_key,
        'outputsize': 'compact',  # 'compact' for last 100 data points, 'full' for full data
    }
    
    # Send GET request
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        data = response.json()
        if "Time Series (Daily)" in data:
            return data["Time Series (Daily)"]
        else:
            print("Error in fetching data:", data)
            return None
    else:
        print(f"Failed to fetch data. Status code: {response.status_code}")
        return None

# Fetch stock data for Apple (AAPL)
symbol = 'AAPL'
stock_data = get_stock_data(symbol, api_key)

# Convert the data into a Pandas DataFrame
if stock_data:
    df = pd.DataFrame.from_dict(stock_data, orient='index')
    df = df[['1. open', '2. high', '3. low', '4. close', '5. volume']]
    df.index = pd.to_datetime(df.index)
    df = df.sort_index(ascending=True)  # Sort data by date
    print(df.head())  # Print the first few rows

    # Save data to CSV
    df.to_csv('stock_data.csv', index=True)
else:
    print("No data fetched")


# In[37]:


url = 'https://www.alphavantage.co/query'
params = {
    'function': 'TIME_SERIES_DAILY',
    'symbol': 'AAPL',
    'apikey': 'BEAJN61YVAP411BR',
    'outputsize': 'full'  # This will fetch the entire history of the stock.
}


# In[38]:


import requests
import pandas as pd
import time

def fetch_stock_data(symbol, api_key, start_date, end_date, batch_size=100):
    all_data = pd.DataFrame()  # DataFrame to store all the batches
    
    # Loop through the dates in batches
    while True:
        url = 'https://www.alphavantage.co/query'
        params = {
            'function': 'TIME_SERIES_DAILY',
            'symbol': symbol,
            'apikey': api_key,
            'outputsize': 'compact',  # 'compact' gives the last 100 data points
        }
        
        response = requests.get(url, params=params)
        data = response.json()

        # Check if the API responded with valid data
        if 'Time Series (Daily)' in data:
            time_series = data['Time Series (Daily)']
            df = pd.DataFrame(time_series).T  # Convert to DataFrame and transpose
            df = df.sort_index()  # Sort by date
            
            # Append new data to the existing DataFrame
            all_data = pd.concat([all_data, df], ignore_index=True)

            # Check if we fetched enough data (adjust batch size)
            if len(all_data) >= batch_size:
                break
        else:
            print("Error fetching data:", data)
            break

        # Delay to avoid hitting rate limits
        time.sleep(12)  # Adjust time based on rate limit (12 seconds for Alpha Vantage)

    return all_data

# Example 


# In[40]:


import requests
import pandas as pd
import time

def fetch_stock_data(symbol, api_key, start_date, end_date, batch_size=100):
    all_data = pd.DataFrame()  # DataFrame to store all the batches
    
    # Loop through the dates in batches
    while True:
        url = 'https://www.alphavantage.co/query'
        params = {
            'function': 'TIME_SERIES_DAILY',
            'symbol': symbol,
            'apikey': api_key,
            'outputsize': 'compact',  # 'compact' gives the last 100 data points
        }
        
        response = requests.get(url, params=params)
        data = response.json()

        # Check if the API responded with valid data
        if 'Time Series (Daily)' in data:
            time_series = data['Time Series (Daily)']
            df = pd.DataFrame(time_series).T  # Convert to DataFrame and transpose
            df = df.sort_index()  # Sort by date
            
            # Append new data to the existing DataFrame
            all_data = pd.concat([all_data, df], ignore_index=True)

            # Check if we fetched enough data (adjust batch size)
            if len(all_data) >= batch_size:
                break
        else:
            print("Error fetching data:", data)
            break

        # Delay to avoid hitting rate limits
        time.sleep(12)  # Adjust time based on rate limit (12 seconds for Alpha Vantage)

    return all_data

# Example usage
symbol = 'AAPL'
api_key = 'BEAJN61YVAP411BR'
start_date = '2024-01-01'
end_date = '2024-07-01'

# Fetch 500 rows in batches of 100
stock_data = fetch_stock_data(symbol, api_key, start_date, end_date, batch_size=500)
print(stock_data)


# In[43]:


import pandas as pd
from datetime import timedelta

# Load the cleaned Reddit data (161 rows)
reddit_data = pd.read_csv('reddit_stock_data.csv')

# Load the stock data (500 rows with stock prices)
stock_data = pd.read_csv('stock_data.csv')

# Convert the 'date' columns to datetime format for both datasets
reddit_data['Date'] = pd.to_datetime(reddit_data['Date'], errors='coerce')
stock_data['date'] = pd.to_datetime(stock_data['date'], errors='coerce')

# Function to find the closest stock date for each Reddit post
def closest_stock_date(reddit_post_date, stock_dates):
    # Find the date in stock_data closest to the Reddit post date
    closest_date = min(stock_dates, key=lambda x: abs(x - reddit_post_date))
    return closest_date

# Apply the function to each Reddit post
reddit_data['closest_stock_date'] = reddit_data['Date'].apply(lambda x: closest_stock_date(x, stock_data['Date']))

# Merge Reddit data with stock data based on the closest date
combined_data = pd.merge(reddit_data, stock_data, left_on='closest_stock_date', right_on='Date', how='left')

# Drop the extra 'closest_stock_date' and 'date' columns
combined_data = combined_data.drop(columns=['closest_stock_date', 'date_y'])

# Save the combined data to a new CSV file
combined_data.to_csv('combined_reddit_stock_data_with_closest_dates.csv', index=False)

# Print the first few rows to verify the result
print(combined_data.head())


# In[44]:


import pandas as pd

# Load stock data (500 rows with stock prices)
stock_data = pd.read_csv('stock_data.csv')

# Convert 'date' to datetime
stock_data['date'] = pd.to_datetime(stock_data['date'], errors='coerce')

# Set the date as the index
stock_data.set_index('date', inplace=True)

# Create sliding window features (e.g., past 7 days)
window_size = 7

# Creating lag features for the past 7 days (you can add more features like moving average, % change, etc.)
for i in range(1, window_size + 1):
    stock_data[f'lag_{i}'] = stock_data['close'].shift(i)

# Drop rows with missing values due to shifting
stock_data.dropna(inplace=True)

# Reset index for ease of merging
stock_data.reset_index(inplace=True)

# Display the first few rows of the stock data with the sliding window
print(stock_data.head())


# In[45]:


# Ensure correct date formats
reddit_data['date'] = pd.to_datetime(reddit_data['date'], errors='coerce')
stock_data['Date'] = pd.to_datetime(stock_data['Date'], errors='coerce')

# Merge the datasets based on the 'date' column (adjust if column names are different)
merged_data = pd.merge(reddit_data, stock_data, left_on='date', right_on='Date', how='left')

# Check the merged data
print(merged_data.head())


# In[48]:


import praw
import pandas as pd

# Initialize Reddit API client
reddit = praw.Reddit( client_id='zUJLqNK8XQqLsvONP-3g3w',          # Replace with your Reddit client ID
    client_secret='7XoR0bgARdBVgy_L5q95MutJzFLmzw',  # Replace with your Reddit client secret
    user_agent='SentimentAnalyzer/1.0 by MadhavPaluru')   # Replace with your app's user agent

# Define subreddits and keywords for scraping
subreddits = ['stocks', 'investing', 'stockmarket', 'wallstreetbets']  # Modify as needed
keywords = ['stock', 'investment', 'market', 'prediction', 'forecast']  # Modify as needed

# Load existing cleaned data
existing_data = pd.read_csv('cleaned_reddit_stock_data.csv')

# Define columns for new data
columns = ['date', 'title', 'post_text', 'comments', 'score']

# Create a list to store new posts data
new_data = []

# Set to track already fetched posts (based on post title or unique identifier)
existing_post_titles = set(existing_data['title'].values)

# Scraping new posts to ensure uniqueness
for subreddit in subreddits:
    for submission in reddit.subreddit(subreddit).search(' OR '.join(keywords), limit=100):  # Adjust limit as needed
        post_title = submission.title
        
        # Check if the post is already in the existing dataset (based on title)
        if post_title not in existing_post_titles:
            post_date = pd.to_datetime(submission.created_utc, unit='s')  # Convert UTC timestamp to datetime
            post_text = submission.selftext
            comments = [comment.body for comment in submission.comments[:5]]  # Top 5 comments
            score = submission.score
            
            # Append the unique post data
            new_data.append([post_date, title, post_text, ' | '.join(comments), score])
            existing_post_titles.add(title)  # Add to set to avoid duplicates

# If there is new data, store it in a new file and clean it
if new_data:
    # Create DataFrame for the new data
    new_df = pd.DataFrame(new_data, columns=columns)
    
    # Save the newly fetched data to a new CSV file
    new_df.to_csv('new_reddit_data.csv', index=False)
    
    # Cleaning the newly fetched data (same process as before)
    cleaned_new_df = new_df.dropna(subset=['post_title', 'post_text', 'comments'])
    cleaned_new_df = cleaned_new_df[cleaned_new_df['post_text'].apply(lambda x: len(x.split()) > 5)]  # Keep longer posts
    cleaned_new_df['date'] = pd.to_datetime(cleaned_new_df['date'])
    
    # Save the cleaned new data to another CSV file
    cleaned_new_df.to_csv('cleaned_new_reddit_data.csv', index=False)

    print(f"Fetched {len(new_data)} new unique posts and saved them to 'new_reddit_data.csv' and cleaned data to 'cleaned_new_reddit_data.csv'.")
else:
    print("No new unique posts were found.")

# Now merge the cleaned datasets
old_cleaned_data = pd.read_csv('cleaned_reddit_stock_data.csv')
new_cleaned_data = pd.read_csv('cleaned_new_reddit_data.csv')

# Concatenate the old and new cleaned data
combined_data = pd.concat([old_cleaned_data, new_cleaned_data], ignore_index=True)

# Optionally, remove duplicates after merging
combined_data = combined_data.drop_duplicates(subset=['title'], keep='first')

# Save the final combined and cleaned data
combined_data.to_csv('final_cleaned_reddit_stock_data.csv', index=False)

print("Combined and cleaned data has been saved to 'final_cleaned_reddit_stock_data.csv'.")


# In[49]:


# Load the existing cleaned data
existing_data = pd.read_csv('cleaned_reddit_stock_data.csv')

# Print the column names to verify their names
print(existing_data.columns)


# In[51]:


# Adjust the column name here if it's different
existing_post_titles = set(existing_data['Title'].values)  # Change 'post_title' to 'title' or the correct name

# In the scraping loop, we also use the correct column names, for example:
post_title = submission.title  # Assuming the title is retrieved this way


# In[52]:


import praw
import pandas as pd

# Initialize Reddit API client
reddit = praw.Reddit(client_id='YOUR_CLIENT_ID',
                     client_secret='YOUR_CLIENT_SECRET',
                     user_agent='YOUR_USER_AGENT')

# Define subreddits and keywords for scraping
subreddits = ['stocks', 'investing', 'stockmarket', 'wallstreetbets']  # Modify as needed
keywords = ['stock', 'investment', 'market', 'prediction', 'forecast']  # Modify as needed

# Load existing cleaned data
existing_data = pd.read_csv('cleaned_reddit_stock_data.csv')

# Print the columns to confirm the correct names
print(existing_data.columns)

# Assuming 'title' is the correct column name for Reddit post titles
# Adjust the column name if necessary based on the output above
existing_post_titles = set(existing_data['Title'].values)  # Adjust 'title' if necessary

# Define columns for new data
columns = ['date', 'post_title', 'post_text', 'comments', 'score']

# Create a list to store new posts data
new_data = []

# Scraping new posts to ensure uniqueness
for subreddit in subreddits:
    for submission in reddit.subreddit(subreddit).search(' OR '.join(keywords), limit=100):  # Adjust limit as needed
        post_title = submission.title
        
        # Check if the post is already in the existing dataset (based on title)
        if post_title not in existing_post_titles:
            post_date = pd.to_datetime(submission.created_utc, unit='s')  # Convert UTC timestamp to datetime
            post_text = submission.selftext
            comments = [comment.body for comment in submission.comments[:5]]  # Top 5 comments
            score = submission.score
            
            # Append the unique post data
            new_data.append([post_date, post_title, post_text, ' | '.join(comments), score])
            existing_post_titles.add(post_title)  # Add to set to avoid duplicates

# If there is new data, store it in a new file and clean it
if new_data:
    # Create DataFrame for the new data
    new_df = pd.DataFrame(new_data, columns=columns)
    
    # Save the newly fetched data to a new CSV file
    new_df.to_csv('new_reddit_data.csv', index=False)
    
    # Cleaning the newly fetched data (same process as before)
    cleaned_new_df = new_df.dropna(subset=['post_title', 'post_text', 'comments'])
    cleaned_new_df = cleaned_new_df[cleaned_new_df['post_text'].apply(lambda x: len(x.split()) > 5)]  # Keep longer posts
    cleaned_new_df['date'] = pd.to_datetime(cleaned_new_df['date'])
    
    # Save the cleaned new data to another CSV file
    cleaned_new_df.to_csv('cleaned_new_reddit_data.csv', index=False)

    print(f"Fetched {len(new_data)} new unique posts and saved them to 'new_reddit_data.csv' and cleaned data to 'cleaned_new_reddit_data.csv'.")
else:
    print("No new unique posts were found.")

# Now merge the cleaned datasets
old_cleaned_data = pd.read_csv('cleaned_reddit_stock_data.csv')
new_cleaned_data = pd.read_csv('cleaned_new_reddit_data.csv')

# Concatenate the old and new cleaned data
combined_data = pd.concat([old_cleaned_data, new_cleaned_data], ignore_index=True)

# Optionally, remove duplicates after merging
combined_data = combined_data.drop_duplicates(subset=['post_title'], keep='first')

# Save the final combined and cleaned data
combined_data.to_csv('final_cleaned_reddit_stock_data.csv', index=False)

print("Combined and cleaned data has been saved to 'final_cleaned_reddit_stock_data.csv'.")


# In[53]:


import praw

# Initialize the Reddit API client with your credentials
reddit = praw.Reddit(client_id='zUJLqNK8XQqLsvONP-3g3w',          # Replace with your Reddit client ID
    client_secret='7XoR0bgARdBVgy_L5q95MutJzFLmzw',  # Replace with your Reddit client secret
    user_agent='SentimentAnalyzer/1.0 by MadhavPaluru')

# Test authentication by printing your Reddit username
print(f"Authenticated as: {reddit.user.me()}")


# In[54]:


import praw

# Replace with your actual credentials
client_id = 'zUJLqNK8XQqLsvONP-3g3w'
client_secret = '7XoR0bgARdBVgy_L5q95MutJzFLmzw'
user_agent = 'SentimentAnalyzer/1.0 by MadhavPaluru'

# Initialize the Reddit API client with your credentials
reddit = praw.Reddit(client_id=client_id,
                     client_secret=client_secret,
                     user_agent=user_agent)

# Try fetching a popular subreddit as a test
try:
    subreddit = reddit.subreddit('python')  # You can use any subreddit name here
    print(f"Successfully connected to the subreddit: {subreddit.display_name}")
except Exception as e:
    print(f"Error occurred: {e}")


# In[ ]:


import praw
import pandas as pd
import datetime
from textblob import TextBlob  # Simple sentiment analysis library

# Your Reddit API credentials
client_id = 'zUJLqNK8XQqLsvONP-3g3w'
client_secret = '7XoR0bgARdBVgy_L5q95MutJzFLmzw'
user_agent = 'SentimentAnalyzer/1.0 by MadhavPaluru'

# Initialize Reddit API client
reddit = praw.Reddit(client_id=client_id,
                     client_secret=client_secret,
                     user_agent=user_agent)

# Specify the subreddit and search keywords (you can customize these)
subreddits = ['stocks', 'investing', 'stockmarket', 'WallStreetBets']
keywords = ['stock', 'stocks', 'investing', 'market', 'stock prices', 'trading', 'investment']

# Function for basic sentiment analysis using TextBlob
def get_sentiment(text):
    analysis = TextBlob(text)
    # Returns polarity: positive, negative or neutral based on score
    if analysis.sentiment.polarity > 0:
        return 'Positive'
    elif analysis.sentiment.polarity < 0:
        return 'Negative'
    else:
        return 'Neutral'

# List to store the scraped data
posts_data = []

# Loop through subreddits and fetch posts with relevant keywords
for subreddit in subreddits:
    for keyword in keywords:
        print(f"Scraping posts from r/{subreddit} with keyword '{keyword}'...")
        try:
            # Perform search in the subreddit for the given keyword
            for submission in reddit.subreddit(subreddit).search(keyword, limit=100):  # Adjust the limit if needed
                post_title = submission.title
                post_score = submission.score
                post_url = submission.url
                post_text = submission.selftext
                post_date = datetime.datetime.utcfromtimestamp(submission.created_utc)
                
                # Get top comments and perform sentiment analysis on the post and comments
                top_comments = []
                for comment in submission.comments[:5]:  # Get top 5 comments
                    if isinstance(comment, praw.models.Comment):  # Avoid handling comment replies
                        top_comments.append(comment.body)
                
                post_sentiment = get_sentiment(post_text)
                
                # Analyzing the sentiment of the top comments
                comments_sentiment = []
                for comment in top_comments:
                    comments_sentiment.append(get_sentiment(comment))
                
                # Add the post data into the list
                posts_data.append({
                    'Title': post_title,
                    'Score': post_score,
                    'URL': post_url,
                    'Post Text': post_text,
                    'Date': post_date,
                    'Keyword': keyword,
                    'Top Comments': top_comments,
                    'Post Sentiment': post_sentiment,
                    'Comments Sentiment': comments_sentiment
                })
        except Exception as e:
            print(f"Error scraping posts from r/{subreddit} with keyword '{keyword}': {e}")

# Convert the collected data into a pandas DataFrame
df = pd.DataFrame(posts_data)

# Save the scraped data into a CSV file|
df.to_csv('new_scraped_reddit_data.csv', index=False)
print(f"Scraped data saved to 'new_scraped_reddit_data.csv'.")


# In[ ]:





# import praw
# import pandas as pd
# import datetime
# from textblob import TextBlob  # Simple sentiment analysis library
# 
# # Your Reddit API credentials
# client_id = 'zUJLqNK8XQqLsvONP-3g3w'
# client_secret = '7XoR0bgARdBVgy_L5q95MutJzFLmzw'
# user_agent = 'SentimentAnalyzer/1.0 by MadhavPaluru'
# 
# # Initialize Reddit API client
# reddit = praw.Reddit(client_id=client_id,
#                      client_secret=client_secret,
#                      user_agent=user_agent)
# 
# # Specify the subreddit and search keywords (you can customize these)
# subreddits = ['stocks', 'investing', 'stockmarket', 'WallStreetBets']
# keywords = ['stock', 'stocks', 'investing', 'market', 'stock prices', 'trading', 'investment']
# 
# # Function for basic sentiment analysis using TextBlob
# def get_sentiment(text):
#     analysis = TextBlob(text)
#     # Returns polarity: positive, negative or neutral based on score
#     if analysis.sentiment.polarity > 0:
#         return 'Positive'
#     elif analysis.sentiment.polarity < 0:
#         return 'Negative'
#     else:
#         return 'Neutral'
# 
# # List to store the scraped data
# posts_data = []
# 
# # Loop through subreddits and fetch posts with relevant keywords
# for subreddit in subreddits:
#     for keyword in keywords:
#         print(f"Scraping posts from r/{subreddit} with keyword '{keyword}'...")
# 
#         try:
#             # Perform search in the subreddit for the given keyword
#             for submission in reddit.subreddit(subreddit).search(keyword, limit=100):  # Adjust the limit if needed
#                 print(f"Scraping Post: {submission.title[:50]}...")  # Display first 50 characters of the post title
#                 post_title = submission.title
#                 post_score = submission.score
#                 post_url = submission.url
#                 post_text = submission.selftext
#                 post_date = datetime.datetime.utcfromtimestamp(submission.created_utc)
#                 
#                 # Get top comments and perform sentiment analysis on the post and comments
#                 top_comments = []
#                 for comment in submission.comments[:5]:  # Get top 5 comments
#                     if isinstance(comment, praw.models.Comment):  # Avoid handling comment replies
#                         top_comments.append(comment.body)
#                 
#                 post_sentiment = get_sentiment(post_text)
#                 
#                 # Analyzing the sentiment of the top comments
#                 comments_sentiment = []
#                 for comment in top_comments:
#                     comments_sentiment.append(get_sentiment(comment))
#                 
#                 # Add the post data into the list
#                 posts_data.append({
#                     'Title': post_title,
#                     'Score': post_score,
#                     'URL': post_url,
#                     'Post Text': post_text,
#                     'Date': post_date,
#                     'Keyword': keyword,
#                     'Top Comments': top_comments,
#                     'Post Sentiment': post_sentiment,
#                     'Comments Sentiment': comments_sentiment
#                 })
#             
#             print(f"Finished scraping posts from r/{subreddit} with keyword '{keyword}'.")
# 
#         except Exception as e:
#             print(f"Error scraping posts from r/{subreddit} with keyword '{keyword}': {e}")
# 
# # Convert the collected data into a pandas DataFrame
# df = pd.DataFrame(posts_data)
# 
# # Save the scraped data into a CSV file
# df.to_csv('new_scraped_reddit_data.csv', index=False)
# print(f"Scraped data saved to 'new_scraped_reddit_data.csv'.")
# 

# In[ ]:


import praw
import pandas as pd
import datetime
from textblob import TextBlob  # Simple sentiment analysis library

# Your Reddit API credentials
client_id = 'YOUR_CLIENT_ID'
client_secret = 'YOUR_CLIENT_SECRET'
user_agent = 'YOUR_USER_AGENT'

# Initialize Reddit API client
reddit = praw.Reddit(client_id=client_id,
                     client_secret=client_secret,
                     user_agent=user_agent)

# Specify the subreddit and search keywords (you can customize these)
subreddits = ['stocks', 'investing', 'stockmarket', 'WallStreetBets']
keywords = ['stock', 'stocks', 'investing', 'market', 'stock prices', 'trading', 'investment']

# Function for basic sentiment analysis using TextBlob
def get_sentiment(text):
    analysis = TextBlob(text)
    # Returns polarity: positive, negative or neutral based on score
    if analysis.sentiment.polarity > 0:
        return 'Positive'
    elif analysis.sentiment.polarity < 0:
        return 'Negative'
    else:
        return 'Neutral'

# List to store the scraped data
posts_data = []

# Loop through subreddits and fetch posts with relevant keywords
for subreddit in subreddits:
    for keyword in keywords:
        print(f"Scraping posts from r/{subreddit} with keyword '{keyword}'...")

        try:
            # Perform search in the subreddit for the given keyword
            for submission in reddit.subreddit(subreddit).search(keyword, limit=100):  # Adjust the limit if needed
                print(f"Scraping Post: {submission.title[:50]}...")  # Display first 50 characters of the post title
                post_title = submission.title
                post_score = submission.score
                post_url = submission.url
                post_text = submission.selftext
                post_date = datetime.datetime.utcfromtimestamp(submission.created_utc)
                
                # Get top comments and perform sentiment analysis on the post and comments
                top_comments = []
                for comment in submission.comments[:5]:  # Get top 5 comments
                    if isinstance(comment, praw.models.Comment):  # Avoid handling comment replies
                        top_comments.append(comment.body)
                
                post_sentiment = get_sentiment(post_text)
                
                # Analyzing the sentiment of the top comments
                comments_sentiment = []
                for comment in top_comments:
                    comments_sentiment.append(get_sentiment(comment))
                
                # Add the post data into the list
                posts_data.append({
                    'Title': post_title,
                    'Score': post_score,
                    'URL': post_url,
                    'Post Text': post_text,
                    'Date': post_date,
                    'Keyword': keyword,
                    'Top Comments': top_comments,
                    'Post Sentiment': post_sentiment,
                    'Comments Sentiment': comments_sentiment
                })
            
            print(f"Finished scraping posts from r/{subreddit} with keyword '{keyword}'.")

        except Exception as e:
            print(f"Error scraping posts from r/{subreddit} with keyword '{keyword}': {e}")

# Convert the collected data into a pandas DataFrame
df = pd.DataFrame(posts_data)

# Save the scraped data into a CSV file
df.to_csv('new_scraped_reddit_data.csv', index=False)
print(f"Scraped data saved to 'new_scraped_reddit_data.csv'.")


# In[ ]:


import praw
import pandas as pd
import datetime
from textblob import TextBlob  # Simple sentiment analysis library

# Your Reddit API credentials
client_id = 'zUJLqNK8XQqLsvONP-3g3w'
client_secret = '7XoR0bgARdBVgy_L5q95MutJzFLmzw'
user_agent = 'SentimentAnalyzer/1.0 by MadhavPaluru'

# Initialize Reddit API client
reddit = praw.Reddit(client_id=client_id,
                     client_secret=client_secret,
                     user_agent=user_agent)

# Specify the subreddit and search keywords (you can customize these)
subreddits = ['stocks', 'investing', 'stockmarket', 'WallStreetBets']
keywords = ['stock', 'stocks', 'investing', 'market', 'stock prices', 'trading', 'investment']

# Function for basic sentiment analysis using TextBlob
def get_sentiment(text):
    analysis = TextBlob(text)
    # Returns polarity: positive, negative or neutral based on score
    if analysis.sentiment.polarity > 0:
        return 'Positive'
    elif analysis.sentiment.polarity < 0:
        return 'Negative'
    else:
        return 'Neutral'

# List to store the scraped data
posts_data = []

# Loop through subreddits and fetch posts with relevant keywords
for subreddit in subreddits:
    for keyword in keywords:
        print(f"Scraping posts from r/{subreddit} with keyword '{keyword}'...")

        try:
            # Perform search in the subreddit for the given keyword
            for submission in reddit.subreddit(subreddit).search(keyword, limit=100):  # Adjust the limit if needed
                print(f"Scraping Post: {submission.title[:50]}...")  # Display first 50 characters of the post title
                post_title = submission.title
                post_score = submission.score
                post_url = submission.url
                post_text = submission.selftext
                post_date = datetime.datetime.utcfromtimestamp(submission.created_utc)
                
                # Get top comments and perform sentiment analysis on the post and comments
                top_comments = []
                for comment in submission.comments[:5]:  # Get top 5 comments
                    if isinstance(comment, praw.models.Comment):  # Avoid handling comment replies
                        top_comments.append(comment.body)
                
                post_sentiment = get_sentiment(post_text)
                
                # Analyzing the sentiment of the top comments
                comments_sentiment = []
                for comment in top_comments:
                    comments_sentiment.append(get_sentiment(comment))
                
                # Add the post data into the list
                posts_data.append({
                    'Title': post_title,
                    'Score': post_score,
                    'URL': post_url,
                    'Post Text': post_text,
                    'Date': post_date,
                    'Keyword': keyword,
                    'Top Comments': top_comments,
                    'Post Sentiment': post_sentiment,
                    'Comments Sentiment': comments_sentiment
                })
            
            print(f"Finished scraping posts from r/{subreddit} with keyword '{keyword}'.")

        except Exception as e:
            print(f"Error scraping posts from r/{subreddit} with keyword '{keyword}': {e}")

# Convert the collected data into a pandas DataFrame
df = pd.DataFrame(posts_data)

# Save the scraped data into a CSV file
df.to_csv('new_scraped_reddit_data.csv', index=False)
print(f"Scraped data saved to 'new_scraped_reddit_data.csv'.")


# In[58]:


import praw
import pandas as pd
import datetime
from textblob import TextBlob

# Your Reddit API credentials
client_id = 'zUJLqNK8XQqLsvONP-3g3w'
client_secret = '7XoR0bgARdBVgy_L5q95MutJzFLmzw'
user_agent = 'SentimentAnalyzer/1.0 by MadhavPaluru'

# Initialize Reddit API client
reddit = praw.Reddit(client_id=client_id,
                     client_secret=client_secret,
                     user_agent=user_agent)

# Subreddits and keywords
subreddits = ['stocks', 'investing', 'stockmarket', 'WallStreetBets']
keywords = ['stock', 'stocks', 'investing', 'market', 'stock prices', 'trading', 'investment']

# Function for sentiment analysis
def get_sentiment(text):
    analysis = TextBlob(text)
    if analysis.sentiment.polarity > 0:
        return 'Positive'
    elif analysis.sentiment.polarity < 0:
        return 'Negative'
    else:
        return 'Neutral'

# Load existing data if available (to avoid duplication)
try:
    df_existing = pd.read_csv('new_scraped_reddit_data.csv')
    existing_titles = df_existing['Title'].tolist()  # List of titles in the existing file
    print(f"Found {len(df_existing)} existing posts in the file.")
except FileNotFoundError:
    df_existing = pd.DataFrame()
    existing_titles = []
    print("No existing file found. Starting fresh.")

posts_data = []

# Scrape data
for subreddit in subreddits:
    for keyword in keywords:
        print(f"Starting to scrape posts from r/{subreddit} with keyword '{keyword}'...")

        try:
            for submission in reddit.subreddit(subreddit).search(keyword, limit=100):
                # Skip posts already scraped (based on title)
                if submission.title in existing_titles:
                    print(f"Skipping already scraped post: {submission.title[:50]}...")
                    continue

                print(f"Scraping Post: {submission.title[:50]}...")

                # Extract post data
                post_title = submission.title
                post_score = submission.score
                post_url = submission.url
                post_text = submission.selftext
                post_date = datetime.datetime.utcfromtimestamp(submission.created_utc)
                
                # Get top comments and perform sentiment analysis
                top_comments = []
                for comment in submission.comments[:5]:
                    if isinstance(comment, praw.models.Comment):
                        top_comments.append(comment.body)
                
                post_sentiment = get_sentiment(post_text)
                comments_sentiment = [get_sentiment(comment) for comment in top_comments]
                
                posts_data.append({
                    'Title': post_title,
                    'Score': post_score,
                    'URL': post_url,
                    'Post Text': post_text,
                    'Date': post_date,
                    'Keyword': keyword,
                    'Top Comments': top_comments,
                    'Post Sentiment': post_sentiment,
                    'Comments Sentiment': comments_sentiment
                })
            
            # Save data periodically
            if posts_data:
                print(f"Saving {len(posts_data)} new posts to 'new_scraped_reddit_data.csv'...")
                df_new = pd.DataFrame(posts_data)
                df_existing = pd.concat([df_existing, df_new], ignore_index=True)
                df_existing.to_csv('new_scraped_reddit_data.csv', index=False)
                print(f"Saved {len(posts_data)} new posts.")
                posts_data = []  # Reset the posts data for the next batch
                
        except Exception as e:
            print(f"Error scraping posts from r/{subreddit} with keyword '{keyword}': {e}")
        
        print(f"Finished scraping posts from r/{subreddit} with keyword '{keyword}'.")

print("Scraping completed!")


# In[59]:


import os

# Print the current working directory
print(f"File saved at: {os.path.abspath('new_scraped_reddit_data.csv')}")


# In[60]:


import pandas as pd
df=pd.read_csv(r'C:\Users\hp\new_scraped_reddit_data.csv')
df.head(10)


# In[61]:


df


# In[2]:


# Drop rows with any missing values
cleaned_data = new_data.dropna()

# Alternatively, fill missing values for specific columns
cleaned_data = new_data.fillna({
    'Title': 'Unknown Title',
    'Post Text': 'No Text',
    'Top Comments': 'No Comments',
    'Post Sentiment': 0,  # Assuming 0 is a neutral sentiment
    'Comments Sentiment': 0  # Neutral sentiment
})


# In[3]:


# Drop rows with any missing values
cleaned_data = new_scraped_reddit_data.dropna()

# Alternatively, fill missing values for specific columns
cleaned_data =  new_scraped_reddit_data.fillna({
    'Title': 'Unknown Title',
    'Post Text': 'No Text',
    'Top Comments': 'No Comments',
    'Post Sentiment': 0,  # Assuming 0 is a neutral sentiment
    'Comments Sentiment': 0  # Neutral sentiment
})


# In[5]:


import pandas as pd

df=pd.read_csv(r'C:\Users\hp\new_scraped_reddit_data.csv')
df.head()               


# In[6]:


cleaned_data = new_scraped_reddit_data.dropna()

# Alternatively, fill missing values for specific columns
cleaned_data = new_scraped_reddit_data.fillna({
    'Title': 'Unknown Title',
    'Post Text': 'No Text',
    'Top Comments': 'No Comments',
    'Post Sentiment': 0,  # Assuming 0 is a neutral sentiment
    'Comments Sentiment': 0  # Neutral sentiment
})


# In[9]:


cleaned_data = df.dropna()

# Alternatively, fill missing values for specific columns
cleaned_data = df.fillna({
    'Title': 'Unknown Title',
    'Post Text': 'No Text',
    'Top Comments': 'No Comments',
    'Post Sentiment': 0,  # Assuming 0 is a neutral sentiment
    'Comments Sentiment': 0  # Neutral sentiment
})
cleaned_data


# In[10]:


# Convert text columns to lowercase and strip extra spaces
text_columns = ['Title', 'Post Text', 'Top Comments']
for col in text_columns:
    cleaned_data[col] = cleaned_data[col].str.lower().str.strip()

# Remove special characters from text fields
for col in text_columns:
    cleaned_data[col] = cleaned_data[col].str.replace(r'[^a-zA-Z0-9\s]', '', regex=True)
cleaned_data


# In[11]:


# Remove duplicates based on all columns
cleaned_data = cleaned_data.drop_duplicates()

# Remove duplicates based on the 'URL' column (if unique identifiers are URLs)
cleaned_data = cleaned_data.drop_duplicates(subset=['URL'])


# In[12]:


# Keep rows where 'Score' is above 10
cleaned_data = cleaned_data[cleaned_data['Score'] > 10]

# Remove rows with neutral sentiment in 'Post Sentiment'
cleaned_data = cleaned_data[cleaned_data['Post Sentiment'] != 0]


# In[13]:


# Load the old cleaned dataset
old_data = pd.read_csv("cleaned_reddit_stock_data.csv")

# Concatenate the two datasets
merged_data = pd.concat([old_data, cleaned_data], ignore_index=True)

# Remove duplicates across the merged dataset
merged_data = merged_data.drop_duplicates(subset=['URL'])


# In[14]:


import pandas as pd

# Load datasets
new_data = pd.read_csv("new_scraped_reddit_data.csv")
old_data = pd.read_csv("cleaned_reddit_stock_data.csv")

# Step 1: Handle missing values
print("Handling missing values...")
cleaned_data = new_data.dropna()

# Step 2: Normalize text
print("Normalizing text columns...")
text_columns = ['Title', 'Post Text', 'Top Comments']
for col in text_columns:
    cleaned_data[col] = cleaned_data[col].str.lower().str.strip()

# Step 3: Remove duplicates
print("Removing duplicates...")
cleaned_data = cleaned_data.drop_duplicates()

# Step 4: Filter rows
print("Filtering rows with low scores...")
cleaned_data = cleaned_data[cleaned_data['Score'] > 10]

# Step 5: Merge datasets
print("Merging datasets...")
merged_data = pd.concat([old_data, cleaned_data], ignore_index=True)

# Step 6: Save the final dataset
merged_data.to_csv("final_cleaned_dataset.csv", index=False)
print("Saved final dataset as 'final_cleaned_dataset.csv'")


# In[15]:


import pandas as pd

df=pd.read_csv(r'C:\Users\hp\final_cleaned_dataset.csv')
df.head(20)


# In[16]:


df


# In[17]:


from yahoo_fin import stock_info

# Fetch historical data for a stock (e.g., Apple)
stock_symbol = "AAPL"
stock_data = stock_info.get_data(stock_symbol, start_date="2022-01-01", end_date="2024-01-01")

# Display the first few rows of data
print(stock_data.head())

# Save the data to CSV
stock_data.to_csv(f"{stock_symbol}_stock_data.csv")
print(f"Saved {stock_symbol} stock data to CSV.")


# In[18]:


import pandas as pd
df=pd.read_csv(r'C:\Users\hp\AAPL_stock_data.csv')
df


# In[19]:


df[1]=Date


# In[20]:


df[1]='Date'


# In[21]:


df


# In[22]:


df[1].drop()


# In[23]:


import pandas as pd
df=pd.read_csv(r'C:\Users\hp\AAPL_stock_data.csv')
df


# In[24]:


import requests
from bs4 import BeautifulSoup

# URL of a stock-related page (e.g., Apple stock)
url = "https://finance.yahoo.com/quote/AAPL/community"

# Send a request to the page
response = requests.get(url)
soup = BeautifulSoup(response.text, "html.parser")

# Extract the comments section
comments = soup.find_all('div', {'class': 'Mb(5px)'})

# Print the first 5 comments
for comment in comments[:5]:
    print(comment.get_text().strip())

# Save the comments to a CSV file
comments_data = [comment.get_text().strip() for comment in comments]
comments_df = pd.DataFrame(comments_data, columns=["User Comments"])
comments_df.to_csv("yahoo_finance_comments.csv", index=False)
print("Saved Yahoo Finance comments to 'yahoo_finance_comments.csv'")


# In[25]:


from textblob import TextBlob
import pandas as pd

# Load the comments
comments_df = pd.read_csv("yahoo_finance_comments.csv")

# Perform sentiment analysis on each comment
def get_sentiment(text):
    analysis = TextBlob(text)
    # Polarity ranges from -1 (negative) to 1 (positive)
    return analysis.sentiment.polarity

comments_df["Sentiment"] = comments_df["User Comments"].apply(get_sentiment)

# Display the sentiment scores
print(comments_df.head())

# Save the results
comments_df.to_csv("sentiment_analyzed_comments.csv", index=False)
print("Saved sentiment-analyzed comments to 'sentiment_analyzed_comments.csv'")


# In[26]:


import pandas as pd

# Load the stock data
stock_data = pd.read_csv("AAPL_stock_data.csv")

# Rename the 'Unnamed: 0' column to 'Date'
stock_data.rename(columns={'Unnamed: 0': 'Date'}, inplace=True)

# Display the first few rows to verify the change
print(stock_data.head())

# Save the updated stock data
stock_data.to_csv("AAPL_stock_data_updated.csv", index=False)
print("Saved the updated stock data with 'Date' as the column name.")


# In[28]:


# Convert the 'Date' column to datetime format
stock_data['Date'] = pd.to_datetime(stock_data['Date'])

# Remove rows with missing values (if any)
stock_data.dropna(inplace=True)

# Display the cleaned data
print(stock_data.head())

# Save the cleaned data
stock_data.to_csv("cleaned_AAPL_stock_data.csv", index=False)
print("Saved the cleaned stock data from yahooo.")


# In[29]:


# Load the cleaned stock data and the sentiment dataset
stock_data = pd.read_csv("cleaned_AAPL_stock_data.csv")
sentiment_data = pd.read_csv("sentiment_analyzed_comments.csv")

# Assuming both datasets have a 'Date' column, merge on 'Date'
merged_data = pd.merge(stock_data, sentiment_data, on="Date", how="outer")

# Display the merged data
print(merged_data.head())

# Save the merged dataset
merged_data.to_csv("merged_stock_sentiment_data.csv", index=False)
print("Saved the merged dataset as 'merged_stock_sentiment_data.csv'")


# In[30]:


# Load the stock data
stock_data = pd.read_csv("AAPL_stock_data.csv")

# Print column names to check their exact names
print(stock_data.columns)


# In[31]:


# Rename the 'Unnamed: 0' column to 'Date' (even if there are extra spaces)
stock_data.rename(columns={stock_data.columns[0]: 'Date'}, inplace=True)

# Display the first few rows to verify the change
print(stock_data.head())


# In[32]:


# Strip extra spaces from column names
stock_data.columns = stock_data.columns.str.strip()

# Display cleaned column names
print(stock_data.columns)


# In[34]:


# Load the cleaned stock data and the sentiment dataset
stock_data = pd.read_csv("stock_data.csv")
sentiment_data = pd.read_csv("sentiment_analyzed_comments.csv")

# Assuming both datasets have a 'Date' column, merge on 'Date'
merged_data = pd.merge(stock_data, sentiment_data, on="Date", how="outer")

# Display the merged data
print(merged_data.head())

# Save the merged dataset
merged_data.to_csv("merged_stock_sentiment_data.csv", index=False)
print("Saved the merged dataset as 'merged_stock_sentiment_data.csv'")


# In[35]:


sentiment_data = pd.read_csv("sentiment_analyzed_comments.csv")


# In[36]:


sentiment_data = pd.read_csv("sentiment_analyzed_comments.csv")
sentiment_data.head()


# In[37]:


import requests
from bs4 import BeautifulSoup

# URL of the Yahoo Finance stock page (e.g., Apple)
url = "https://finance.yahoo.com/quote/AAPL/community"

# Send a request to the page
response = requests.get(url)
soup = BeautifulSoup(response.text, "html.parser")

# Extract comment sections (adjust the selector to match the HTML structure)
comments = soup.find_all('div', {'class': 'Mb(5px)'})  # Update with the correct class if necessary
dates = soup.find_all('span', {'class': 'C(#959595) Fz(12px)'})  # Example for date span class

# Combine comments and dates (if available)
comment_date_pairs = []
for comment, date in zip(comments, dates):
    comment_text = comment.get_text().strip()
    date_text = date.get_text().strip()  # Date text (could be formatted differently)
    comment_date_pairs.append((comment_text, date_text))

# Display the first few comment-date pairs
for pair in comment_date_pairs[:5]:
    print(pair)

# Save the comment-date pairs to a CSV
import pandas as pd
comment_data = pd.DataFrame(comment_date_pairs, columns=["Comment", "Date"])
comment_data.to_csv("yahoo_finance_comments_with_date.csv", index=False)
print("Saved comments with dates to 'yahoo_finance_comments_with_date.csv'")


# In[38]:


import pandas as pd
from textblob import TextBlob

# Load the Yahoo Finance comments data with dates (already retrieved earlier)
comments_data = pd.read_csv("yahoo_finance_comments_with_date.csv")

# Function to get sentiment polarity and subjectivity
def get_sentiment(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity  # Sentiment polarity (-1 to 1)
    subjectivity = blob.sentiment.subjectivity  # Subjectivity (0 to 1)
    return polarity, subjectivity

# Apply sentiment analysis to each comment
comments_data[['Sentiment_Polarity', 'Sentiment_Subjectivity']] = comments_data['Comment'].apply(
    lambda comment: pd.Series(get_sentiment(comment))
)

# Display the first few rows to check the results
print(comments_data.head())

# Save the data with sentiments to a CSV file
comments_data.to_csv("yahoo_finance_comments_with_sentiment.csv", index=False)
print("Saved the comments with sentiment analysis results to 'yahoo_finance_comments_with_sentiment.csv'")


# In[39]:


import pandas as pd
from textblob import TextBlob

# Load the Yahoo Finance comments data with dates (already retrieved earlier)
comments_data = pd.read_csv("yahoo_finance_comments_with_date.csv")

# Function to get sentiment polarity
def get_sentiment(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity  # Sentiment polarity (-1 to 1)
    return polarity

# Apply sentiment analysis to each comment
comments_data['Sentiment'] = comments_data['Comment'].apply(lambda comment: get_sentiment(comment))

# Create a new DataFrame with only User Comments, Sentiment, and Date
final_data = comments_data[['Comment', 'Sentiment', 'Date']]

# Rename columns for clarity if needed
final_data.columns = ['User Comments', 'Sentiment', 'Date']

# Display the first few rows to check
print(final_data.head())

# Save the filtered data to a new CSV file
final_data.to_csv("yahoo_finance_comments_sentiment.csv", index=False)
print("Saved the User Comments, Sentiment, and Date to 'yahoo_finance_comments_sentiment.csv'")


# In[40]:


import pandas as pd

# Load the sentiment data with comments and sentiment
sentiment_data = pd.read_csv("yahoo_finance_comments_sentiment.csv")

# Load the cleaned AAPL stock data
stock_data = pd.read_csv("cleaned_AAPL_stock_data.csv")

# Merge the sentiment data with stock data on the 'Date' column
merged_data = pd.merge(stock_data, sentiment_data, on="Date", how="outer")  # 'outer' keeps all rows, change to 'inner' if you only want matching dates

# Display the first few rows of the merged data
print(merged_data.head())

# Save the merged data to a CSV file
merged_data.to_csv("merged_stock_sentiment_data.csv", index=False)
print("Saved the merged data to 'merged_stock_sentiment_data.csv'")


# In[41]:


import pandas as pd

# Load the merged stock sentiment data
merged_data = pd.read_csv("merged_stock_sentiment_data.csv")

# Display the first few rows of the merged data
print("Displaying the first few rows of the merged data:")
print(merged_data.head())



# In[42]:


import requests
from bs4 import BeautifulSoup
import pandas as pd

# Function to fetch Yahoo Finance page and extract comments
def fetch_comments(url):
    # Send a GET request to the Yahoo Finance page
    response = requests.get(url)
    
    # Check if the request was successful
    if response.status_code == 200:
        page_content = response.content
        
        # Parse the page content using BeautifulSoup
        soup = BeautifulSoup(page_content, 'html.parser')
        
        # Find the comments section (you may need to inspect the HTML structure to find the exact class names or tags)
        comments = []
        for comment in soup.find_all('div', class_='comment'):
            text = comment.find('p', class_='comment-text').get_text(strip=True)
            if text:
                comments.append(text)
        
        return comments
    else:
        print(f"Failed to fetch page: {url}")
        return []

# Example Yahoo Finance article URL (replace with actual URL)
url = "https://finance.yahoo.com/quote/AAPL/community?p=AAPL"

# Fetch the comments for the given URL
user_comments = fetch_comments(url)

# Create a DataFrame with the extracted comments
comments_df = pd.DataFrame(user_comments, columns=['User Comments'])

# Add the Date column (using current date for simplicity, replace with actual date extraction if possible)
comments_df['Date'] = pd.to_datetime('today').strftime('%Y-%m-%d')

# Save the comments to a CSV file
comments_df.to_csv("yahoo_finance_comments.csv", index=False)
print("Saved the extracted user comments to 'yahoo_finance_comments.csv'")


# In[43]:


import pandas as pd
df=pd.read_csv(r"C:\Users\hp\yahoo_finance_comments.csv")
df


# In[44]:


headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36'
}
response = requests.get(url, headers=headers)


# In[45]:


from selenium import webdriver
from selenium.webdriver.common.by import By
import pandas as pd

# Set up the Selenium WebDriver (use a browser driver like ChromeDriver)
driver = webdriver.Chrome()  # Ensure you have the ChromeDriver installed

# Open the Yahoo Finance community page for a specific stock
url = "https://finance.yahoo.com/quote/AAPL/community?p=AAPL"
driver.get(url)

# Wait for the comments section to load (adjust based on your internet speed)
driver.implicitly_wait(10)

# Extract comments
comments = driver.find_elements(By.CSS_SELECTOR, 'div.comment div.comment-text')
comment_list = [comment.text for comment in comments]

# Close the driver
driver.quit()

# Create a DataFrame with comments
comments_df = pd.DataFrame(comment_list, columns=['User Comments'])
comments_df['Date'] = pd.to_datetime('today').strftime('%Y-%m-%d')

# Save to a CSV file
comments_df.to_csv("yahoo_finance_comments.csv", index=False)
print("Saved the extracted user comments to 'yahoo_finance_comments.csv'")


# In[51]:


import praw
import pandas as pd
from textblob import TextBlob

# Reddit API credentials
reddit = praw.Reddit(
 client_id='zUJLqNK8XQqLsvONP-3g3w',          # Replace with your Reddit client ID
    client_secret='7XoR0bgARdBVgy_L5q95MutJzFLmzw',  # Replace with your Reddit client secret
    user_agent='SentimentAnalyzer/1.0 by MadhavPaluru'   # Replace with your app's user agent
)

# New subreddits to scrape
subreddits = ['ValueInvesting', 'DividendGrowth', 'StocksAndTrading', 'PennyStocks']

# List to store scraped data
data = []

# Function to calculate sentiment
def calculate_sentiment(text):
    if text:  # Check if text is not empty
        sentiment = TextBlob(text).sentiment.polarity
        return sentiment
    return 0.0  # Neutral sentiment for empty text

# Scrape data
for subreddit in subreddits:
    print(f"Scraping posts from r/{subreddit}...")
    for post in reddit.subreddit(subreddit).hot(limit=100):  # Adjust the limit as needed
        top_comments = [comment.body for comment in post.comments if hasattr(comment, "body")][:5]  # Limit top comments
        post_data = {
            "Title": post.title,
            "Score": post.score,
            "URL": post.url,
            "Post Text": post.selftext,
            "Date": pd.to_datetime(post.created_utc, unit='s'),
            "Keyword": subreddit,
            "Top Comments": " ".join(top_comments),  # Combine top comments into one string
            "Post Sentiment": calculate_sentiment(post.selftext),
            "Comments Sentiment": calculate_sentiment(" ".join(top_comments))
        }
        data.append(post_data)

# Convert to DataFrame
new_data_df = pd.DataFrame(data)

# Remove duplicates based on title and text (ensures uniqueness)
print("Removing duplicate entries...")
new_data_df = new_data_df.drop_duplicates(subset=["Title", "Post Text"], keep="first")

# Load previously collected data to ensure no overlap
existing_data = pd.read_csv("final_cleaned_dataset.csv")
print("Filtering out overlapping data with previously collected dataset...")
unique_data_df = new_data_df[~new_data_df["Title"].isin(existing_data["Title"])]

# Save the unique data to a new CSV file
unique_data_df.to_csv("unique_reddit_sentiment_data.csv", index=False)
print(f"Saved unique data with sentiment analysis to 'unique_reddit_sentiment_data.csv'. Total rows: {len(unique_data_df)}")


# In[52]:


import pandas as pd
df=pd.read_csv(r"C:\Users\hp\final_cleaned_dataset.csv")
df


# In[53]:


import pandas as pd

# Load the newly collected data
new_data_df = pd.read_csv("unique_reddit_sentiment_data.csv")
print(f"Loaded data with {len(new_data_df)} rows.")

# 1. Remove duplicates
print("Removing duplicate entries...")
new_data_df = new_data_df.drop_duplicates(subset=["Title", "Post Text"], keep="first")
print(f"Data after removing duplicates: {len(new_data_df)} rows.")

# 2. Handle missing values
print("Handling missing values...")
new_data_df = new_data_df.dropna(subset=["Post Text", "Top Comments"])  # Drop rows with missing text or comments
new_data_df["Keyword"] = new_data_df["Keyword"].fillna("Unknown")  # Fill missing keywords with 'Unknown'
print(f"Data after handling missing values: {len(new_data_df)} rows.")

# 3. Filter irrelevant data (e.g., short posts/comments)
print("Filtering short posts and comments...")
new_data_df = new_data_df[
    (new_data_df["Post Text"].str.len() > 20) &  # Keep posts with more than 20 characters
    (new_data_df["Top Comments"].str.len() > 10)  # Keep comments with more than 10 characters
]
print(f"Data after filtering short text: {len(new_data_df)} rows.")

# 4. Reformat the Date column
print("Reformatting Date column...")
new_data_df["Date"] = pd.to_datetime(new_data_df["Date"], errors="coerce")  # Convert to datetime, handle errors
new_data_df = new_data_df.dropna(subset=["Date"])  # Drop rows with invalid dates
print(f"Data after reformatting dates: {len(new_data_df)} rows.")

# 5. Validate sentiment values
print("Validating sentiment values...")
new_data_df["Post Sentiment"] = pd.to_numeric(new_data_df["Post Sentiment"], errors="coerce").fillna(0.0)
new_data_df["Comments Sentiment"] = pd.to_numeric(new_data_df["Comments Sentiment"], errors="coerce").fillna(0.0)
print("Sentiment values validated.")

# Save the cleaned data
cleaned_new_data_file = "cleaned_new_reddit_sentiment_data.csv"
new_data_df.to_csv(cleaned_new_data_file, index=False)
print(f"Saved cleaned data to '{cleaned_new_data_file}'. Total rows: {len(new_data_df)}")


# In[54]:


import pandas as pd

# Load the existing cleaned dataset
final_cleaned_dataset = pd.read_csv("final_cleaned_dataset.csv")
print(f"Loaded 'final_cleaned_dataset.csv' with {len(final_cleaned_dataset)} rows.")

# Load the newly cleaned data
new_cleaned_data = pd.read_csv("cleaned_new_reddit_sentiment_data.csv")
print(f"Loaded 'cleaned_new_reddit_sentiment_data.csv' with {len(new_cleaned_data)} rows.")

# Merge the two datasets
print("Merging datasets...")
merged_data = pd.concat([final_cleaned_dataset, new_cleaned_data], ignore_index=True)

# Remove duplicates after merging to ensure uniqueness
print("Removing duplicates after merging...")
merged_data = merged_data.drop_duplicates(subset=["Title", "Post Text"], keep="first")
print(f"Final merged dataset contains {len(merged_data)} rows.")

# Save the merged dataset to a new CSV file
merged_file_name = "merged_reddit_dataset_to_Machine_Learning.csv"
merged_data.to_csv(merged_file_name, index=False)
print(f"Saved merged dataset to '{merged_file_name}'.")


# In[55]:


import pandas as pd
df=pd.read_csv(r"C:\Users\hp\merged_reddit_dataset_to_Machine_Learning.csv")
df


# In[56]:


import pandas as pd
from sklearn.model_selection import train_test_split

# Load the cleaned dataset (both sentiment and stock data)
data = pd.read_csv('merged_reddit_dataset_to_Machine_Learning.csv')

# Ensure no missing values in key columns
data = data.dropna(subset=['Post Sentiment', 'Stock Price'])

# Example: Adding a new feature for stock price change (closing - opening price)
data['Price Change'] = data['close'] - data['open']

# Normalize sentiment score (if needed)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
data[['Post Sentiment']] = scaler.fit_transform(data[['Post Sentiment']])

# Split the data into features (X) and target (y)
X = data[['Post Sentiment', 'Price Change']]  # Features: sentiment score and price change
y = data['close']  # Target: stock closing price (or you can use the next day's price for forecasting)

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training data size: {len(X_train)}, Test data size: {len(X_test)}")


# In[59]:


import pandas as pd

# Load the stock price data (assumed that you have fetched this from Yahoo Finance)
stock_data = pd.read_csv('cleaned_AAPL_stock_data.csv')

# Convert the 'Date' column to datetime for proper alignment
stock_data['Date'] = pd.to_datetime(stock_data['Date'])
stock_data = stock_data.sort_values(by='Date')

# Create a new column 'Price Movement' based on close price
stock_data['Price Movement'] = stock_data['close'].pct_change().apply(lambda x: 1 if x > 0 else 0)

# Ensure 'Date' is also present in the comments dataset to merge with stock data
comments_data = pd.read_csv('final_cleaned_dataset.csv')
comments_data['Date'] = pd.to_datetime(comments_data['Date'])

# Merge sentiment data with stock price data on Date
merged_data = pd.merge(comments_data, stock_data[['Date', 'Price Movement']], on='Date', how='left')

# Check the merged data
merged_file_name='Machine_learning_data.csv'
merged_data.to_csv(merged_file_name, index=False)
import pandas as pd
df=pd.read_csv(r"C:\Users\hp\Machine_learning_data.csv")
df


# In[60]:


import pandas as pd

# Assuming you already have stock price data loaded
stock_data = pd.read_csv('cleaned_AAPL_stock_data.csv')

# Convert the 'Date' column to datetime for proper alignment
stock_data['Date'] = pd.to_datetime(stock_data['Date'])
stock_data = stock_data.sort_values(by='Date')

# Calculate the price movement (percentage change)
stock_data['Price Movement'] = stock_data['close'].pct_change().apply(lambda x: 1 if x > 0 else 0)

# Remove rows where Price Movement is NaN (the first row)
stock_data = stock_data.dropna(subset=['Price Movement'])

# Now you should have a 'Price Movement' column without NaN values
print(stock_data.head())


# In[61]:


# Load the comments dataset
comments_data = pd.read_csv('final_cleaned_dataset.csv')
comments_data['Date'] = pd.to_datetime(comments_data['Date'])

# Merge the sentiment data with stock price data on the Date column
merged_data = pd.merge(comments_data, stock_data[['Date', 'Price Movement']], on='Date', how='left')

# Check the merged data
print(merged_data.head())


# In[62]:


import pandas as pd

# Load stock data
stock_data = pd.read_csv('cleaned_AAPL_stock_data.csv')

# Ensure the Date column is in datetime format and sorted
stock_data['Date'] = pd.to_datetime(stock_data['Date'])
stock_data = stock_data.sort_values(by='Date')

# Check the data
print(stock_data[['Date', 'close']].head())  # Ensure there is no missing data in 'close' and 'Date'


# In[63]:


# Calculate the price movement based on percentage change
stock_data['Price Movement'] = stock_data['close'].pct_change()

# Check the first few rows
print(stock_data[['Date', 'close', 'Price Movement']].head())


# In[64]:


# Remove rows where 'Price Movement' is NaN (which will only be the first row)
stock_data = stock_data.dropna(subset=['Price Movement'])

# Ensure the data is clean and ready for merging
print(stock_data[['Date', 'Price Movement']].head())


# In[65]:


# Load comments dataset
comments_data = pd.read_csv('final_cleaned_dataset.csv')
comments_data['Date'] = pd.to_datetime(comments_data['Date'])

# Merge the sentiment data with stock price data on the Date column
merged_data = pd.merge(comments_data, stock_data[['Date', 'Price Movement']], on='Date', how='left')

# Check the merged data
print(merged_data[['Date', 'Price Movement']].head())


# In[66]:


# Print unique dates from both datasets to ensure they are aligned
print("Unique Dates in Stock Data:", stock_data['Date'].unique()[:10])
print("Unique Dates in Comments Data:", comments_data['Date'].unique()[:10])


# In[67]:


# Merge the sentiment data with stock price data on the Date column
merged_data = pd.merge(comments_data, stock_data[['Date', 'Price Movement']], on='Date', how='left')

# Check the merged data to ensure correct alignment
print(merged_data[['Date', 'Price Movement']].head())


# In[68]:


# Drop rows where 'Price Movement' is NaN
merged_data = merged_data.dropna(subset=['Price Movement'])

# Check the cleaned merged data
print(merged_data[['Date', 'Price Movement']].head())


# In[69]:


# Forward-fill missing values in the 'Price Movement' column
merged_data['Price Movement'] = merged_data['Price Movement'].fillna(method='ffill')

# Check the merged data after filling missing values
print(merged_data[['Date', 'Price Movement']].head())


# In[70]:


# Forward-fill missing values in the 'Price Movement' column
merged_data['Price Movement'] = merged_data['Price Movement'].fillna(method='ffill')

# Check the merged data after filling missing values
print(merged_data[['Date', 'Price Movement']].head())


# In[71]:


# Calculate the price movement again, just to be sure
stock_data['Price Movement'] = stock_data['close'].pct_change()

# Check for correct price movement
print(stock_data[['Date', 'close', 'Price Movement']].head())


# In[ ]:





# In[72]:


# Final merged dataset
print(merged_data.head())


# In[73]:


# Merge the datasets with a left join (keeps all comments data)
merged_data = pd.merge(comments_data, stock_data[['Date', 'Price Movement']], on='Date', how='left')

# Forward-fill missing 'Price Movement' values (carry the previous valid value forward)
merged_data['Price Movement'] = merged_data['Price Movement'].fillna(method='ffill')

# Check the first few rows of the merged data
print(merged_data[['Date', 'Price Movement']].head())


# In[74]:


# Ensure both datasets are sorted by Date
comments_data['Date'] = pd.to_datetime(comments_data['Date'])
stock_data['Date'] = pd.to_datetime(stock_data['Date'])

comments_data = comments_data.sort_values(by='Date')
stock_data = stock_data.sort_values(by='Date')

# Perform an as-of merge: match each comment to the closest stock price
merged_data = pd.merge_asof(comments_data, stock_data[['Date', 'Price Movement']], on='Date')

# Check the first few rows of the merged data
print(merged_data[['Date', 'Price Movement']].head())


# In[75]:


# Ensure that the 'Date' columns are in datetime format
comments_data['Date'] = pd.to_datetime(comments_data['Date'])
stock_data['Date'] = pd.to_datetime(stock_data['Date'])

# Filter out dates that do not appear in both datasets
common_dates = comments_data[comments_data['Date'].isin(stock_data['Date'])]

# Merge the filtered data on the 'Date' column
merged_data = pd.merge(common_dates, stock_data[['Date', 'Price Movement']], on='Date')

# Check the merged data
print(merged_data[['Date', 'Price Movement']].head())


# In[76]:


# Filter stock data to match the time period of comments data (2022)
stock_data_2022 = stock_data[stock_data['Date'].dt.year == 2022]

# Check the first few rows after filtering
print(stock_data_2022.head())


# In[77]:


# Merge filtered stock data with comments data
merged_data = pd.merge(comments_data, stock_data_2022[['Date', 'Price Movement']], on='Date', how='left')

# Check for missing values
print(merged_data.isna().sum())


# In[79]:


# Resample the stock data to monthly average
stock_data_monthly = stock_data.resample('ME', on='Date').mean()

# Check the first few rows after resampling
print(stock_data_monthly.head())


# In[80]:


# Resample the stock data to weekly average
stock_data_weekly = stock_data.resample('W', on='Date').mean()

# Check the first few rows after resampling
print(stock_data_weekly.head())


# In[81]:


# Merge resampled stock data with comments data (based on nearest date)
merged_data = pd.merge_asof(comments_data.sort_values('Date'), 
                            stock_data_monthly.sort_values('Date'), 
                            on='Date', 
                            direction='nearest')

# Check for missing values and inspect merged data
print(merged_data.isna().sum())
print(merged_data.head())


# In[82]:


# Resample stock data by month and calculate the average values
stock_data_monthly = stock_data.resample('M', on='Date').mean()

# Or, you can resample by week
stock_data_weekly = stock_data.resample('W', on='Date').mean()


# In[83]:


# Resample stock data by month and calculate the average values, considering only numeric columns
stock_data_numeric = stock_data.select_dtypes(include=['number'])  # Keep only numeric columns
stock_data_monthly = stock_data_numeric.resample('M', on='Date').mean()


# In[84]:


print(stock_data.columns)



# In[85]:


# Rename the column to 'Date' if it has a different name
stock_data.rename(columns={'YourColumnName': 'Date'}, inplace=True)


# In[86]:


# Rename the column to 'Date' if it has a different name
stock_data.rename(columns={'Date': 'Date'}, inplace=True)


# In[87]:


# Strip any leading/trailing spaces from column names
stock_data.columns = stock_data.columns.str.strip()


# In[88]:


# Convert 'Date' column to datetime
stock_data['Date'] = pd.to_datetime(stock_data['Date'], errors='coerce')


# In[89]:


# Resample stock data by month and calculate the average values
stock_data_numeric = stock_data.select_dtypes(include=['number'])  # Keep only numeric columns
stock_data_monthly = stock_data_numeric.resample('M', on='Date').mean()


# In[90]:


# Resample stock data by month and calculate the average values
stock_data_numeric = stock_data.select_dtypes(include=['number'])  # Keep only numeric columns
stock_data_monthly = stock_data_numeric.resample('M', on='Date').mean()


# In[91]:


# Strip leading/trailing spaces from column names
stock_data.columns = stock_data.columns.str.strip()

# Check the column names again
print(stock_data.columns)


# In[92]:


# Convert 'Date' column to datetime if not already in datetime format
stock_data['Date'] = pd.to_datetime(stock_data['Date'], errors='coerce')

# Verify the data type
print(stock_data['Date'].dtype)


# In[93]:


# Check for any missing/invalid values in 'Date'
print(stock_data['Date'].isna().sum())


# In[94]:


# Drop rows with invalid dates
stock_data = stock_data.dropna(subset=['Date'])

# Alternatively, you can fill missing dates with a default value
# stock_data['Date'] = stock_data['Date'].fillna('your_default_value')


# In[95]:


# Set 'Date' as index
stock_data.set_index('Date', inplace=True)

# Perform the resampling operation
stock_data_monthly = stock_data.resample('M').mean()


# In[96]:


# Using asfreq() if resampling with mean fails
stock_data_monthly = stock_data.asfreq('M')


# In[97]:


print(stock_data.head())


# In[98]:


# Check the frequency of the datetime index
print(stock_data.index.freq)


# In[99]:


# Set frequency to daily if it's not set
stock_data = stock_data.asfreq('D')


# In[100]:


# Resample the data by month, calculating the mean for each column
stock_data_monthly = stock_data.resample('M').mean()

# Check the first few rows of the resampled data
print(stock_data_monthly.head())


# In[101]:


# Check the data types of each column
print(stock_data.dtypes)


# In[102]:


# Convert columns to numeric, forcing errors to NaN (if any non-numeric values exist)
stock_data[['open', 'high', 'low', 'close', 'adjclose', 'volume']] = stock_data[['open', 'high', 'low', 'close', 'adjclose', 'volume']].apply(pd.to_numeric, errors='coerce')

# Check the data types again
print(stock_data.dtypes)


# In[103]:


# Fill missing values using forward fill (or backward fill)
stock_data = stock_data.fillna(method='ffill')

# Alternatively, drop rows with missing values
# stock_data = stock_data.dropna()


# In[104]:


# Resample the data by month, calculating the mean for each column
stock_data_monthly = stock_data.resample('M').mean()

# Check the resampled data
print(stock_data_monthly.head())


# In[105]:


# Check for non-numeric values in the columns
for column in ['open', 'high', 'low', 'close', 'adjclose', 'volume']:
    non_numeric = stock_data[column].apply(pd.to_numeric, errors='coerce').isna()
    if non_numeric.any():
        print(f"Non-numeric values found in {column}:")
        print(stock_data[non_numeric][['Date', column]])


# In[106]:


# Convert columns to numeric (forcefully replace invalid values with NaN)
stock_data[['open', 'high', 'low', 'close', 'adjclose', 'volume']] = stock_data[['open', 'high', 'low', 'close', 'adjclose', 'volume']].apply(pd.to_numeric, errors='coerce')

# Check if any columns contain NaN after conversion
print(stock_data.isna().sum())


# In[107]:


# Fill missing values with forward fill
stock_data = stock_data.fillna(method='ffill')

# Alternatively, drop rows with missing values
# stock_data = stock_data.dropna()


# In[108]:


# Convert Date column to datetime format (if not already)
stock_data['Date'] = pd.to_datetime(stock_data['Date'], errors='coerce')

# Set Date as index
stock_data = stock_data.set_index('Date')

# Ensure the index is a datetime type
print(stock_data.index)


# In[109]:


# Check the column names
print(stock_data.columns)


# In[110]:


# Strip any leading/trailing whitespaces from the column names
stock_data.columns = stock_data.columns.str.strip()


# In[111]:


# Inspect the first few rows of the DataFrame
print(stock_data.head())


# In[112]:


# Convert the index to datetime if it's the Date index
stock_data.index = pd.to_datetime(stock_data.index, errors='coerce')


# In[113]:


# Set Date column as the index
stock_data = stock_data.set_index('Date')

# Convert index (Date) to datetime format
stock_data.index = pd.to_datetime(stock_data.index, errors='coerce')


# In[114]:


# Check the name of the index
print(stock_data.index.name)


# In[115]:


# Check the first few rows again
print(stock_data.head())


# In[116]:


# Ensure the Date index is in datetime format
stock_data.index = pd.to_datetime(stock_data.index, errors='coerce')


# In[117]:


# Extract year, month, and day from the Date index
stock_data['Year'] = stock_data.index.year
stock_data['Month'] = stock_data.index.month
stock_data['Day'] = stock_data.index.day


# In[118]:


# Calculate a 5-day moving average for the 'close' price
stock_data['5-day MA'] = stock_data['close'].rolling(window=5).mean()


# In[119]:


# Check for missing values in the DataFrame
print(stock_data.isnull().sum())


# In[120]:


import matplotlib.pyplot as plt

# Plot the 'close' price
stock_data['close'].plot(figsize=(10, 6))
plt.title('Stock Price Movement')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.grid(True)
plt.show()


# In[121]:


# Calculate price movement (percentage change in 'close')
stock_data['Price Movement'] = stock_data['close'].pct_change()


# In[122]:


import seaborn as sns
# Correlation heatmap
sns.heatmap(stock_data.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.show()


# In[123]:


ValueError: could not convert string to float: 'AAPL'


# In[124]:


stock_data = stock_data.drop(columns=['ticker'])


# In[125]:


from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
stock_data['ticker'] = label_encoder.fit_transform(stock_data['ticker'])


# In[126]:


print(stock_data.columns)


# In[127]:


# Check the column names in the dataset to verify available columns
print(stock_data.columns)

# Drop any unwanted columns (if necessary)
# stock_data = stock_data.drop(columns=['ticker'])  # No 'ticker' column, so skip this step

# Prepare features (X) and target (y)
X = stock_data[['open', 'high', 'low', 'close', 'adjclose', 'volume']]  # Features
y = stock_data['Price Movement']  # Target

# Train-test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Model training
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Model evaluation
from sklearn.metrics import mean_absolute_error
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error: {mae}")


# In[128]:


print(stock_data['Price Movement'].isna().sum())  # Count NaN values in 'Price Movement'


# In[129]:


stock_data = stock_data.dropna(subset=['Price Movement'])


# In[130]:


stock_data['Price Movement'] = stock_data['Price Movement'].fillna(0)  # Replace NaN with 0
# Or you can fill with the mean or median
# stock_data['Price Movement'] = stock_data['Price Movement'].fillna(stock_data['Price Movement'].mean())


# In[131]:


# Check for NaN values in the target variable 'Price Movement'
print(stock_data['Price Movement'].isna().sum())

# Remove rows with NaN in 'Price Movement'
stock_data = stock_data.dropna(subset=['Price Movement'])

# Prepare features (X) and target (y)
X = stock_data[['open', 'high', 'low', 'close', 'adjclose', 'volume']]  # Features
y = stock_data['Price Movement']  # Target

# Train-test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Model training
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Model evaluation
from sklearn.metrics import mean_absolute_error
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error: {mae}")


# In[132]:


from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_pred)
print(f"R-Squared: {r2}")


# In[133]:


import matplotlib.pyplot as plt
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Price Movement')
plt.ylabel('Predicted Price Movement')
plt.title('Actual vs Predicted')
plt.show()


# In[134]:


from sklearn.model_selection import cross_val_score
cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_absolute_error').mean()


# In[135]:


from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_pred)
print(f"R-Squared: {r2}")


# In[136]:


import matplotlib.pyplot as plt
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Price Movement')
plt.ylabel('Predicted Price Movement')
plt.title('Actual vs Predicted')
plt.show()


# In[137]:


from sklearn.model_selection import GridSearchCV
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(estimator=RandomForestRegressor(), param_grid=param_grid, cv=5, scoring='neg_mean_absolute_error')
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_


# In[138]:


from xgboost import XGBRegressor
model = XGBRegressor(n_estimators=100, learning_rate=0.05)
model.fit(X_train, y_train)


# In[ ]:


from xgboost import XGBRegressor
model = XGBRegressor(n_estimators=100, learning_rate=0.05)
model.fit(X_train, y_train)


# In[ ]:


# Calculate moving averages
stock_data['SMA_5'] = stock_data['close'].rolling(window=5).mean()
stock_data['SMA_10'] = stock_data['close'].rolling(window=10).mean()
stock_data['SMA_30'] = stock_data['close'].rolling(window=30).mean()


# In[ ]:


# Calculate exponential moving averages
stock_data['EMA_5'] = stock_data['close'].ewm(span=5, adjust=False).mean()
stock_data['EMA_10'] = stock_data['close'].ewm(span=10, adjust=False).mean()
stock_data['EMA_30'] = stock_data['close'].ewm(span=30, adjust=False).mean()


# In[ ]:


# Calculate RSI
delta = stock_data['close'].diff(1)
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()

rs = gain / loss
stock_data['RSI'] = 100 - (100 / (1 + rs))


# In[ ]:


# Calculate Bollinger Bands
stock_data['Bollinger_Mid'] = stock_data['SMA_20']  # 20-period simple moving average
stock_data['Bollinger_Upper'] = stock_data['Bollinger_Mid'] + 2 * stock_data['close'].rolling(window=20).std()
stock_data['Bollinger_Lower'] = stock_data['Bollinger_Mid'] - 2 * stock_data['close'].rolling(window=20).std()


# In[ ]:


# Price change from previous day
stock_data['Price_Change'] = stock_data['close'].pct_change()


# In[ ]:


# Volume percentage change
stock_data['Volume_Change'] = stock_data['volume'].pct_change()


# In[ ]:


# Rolling mean and std dev for close and volume
stock_data['Rolling_Mean_Close'] = stock_data['close'].rolling(window=5).mean()
stock_data['Rolling_Std_Close'] = stock_data['close'].rolling(window=5).std()
stock_data['Rolling_Mean_Volume'] = stock_data['volume'].rolling(window=5).mean()
stock_data['Rolling_Std_Volume'] = stock_data['volume'].rolling(window=5).std()


# In[ ]:


# Moving averages
stock_data['SMA_5'] = stock_data['close'].rolling(window=5).mean()
stock_data['SMA_10'] = stock_data['close'].rolling(window=10).mean()
stock_data['SMA_30'] = stock_data['close'].rolling(window=30).mean()

# Exponential moving averages
stock_data['EMA_5'] = stock_data['close'].ewm(span=5, adjust=False).mean()
stock_data['EMA_10'] = stock_data['close'].ewm(span=10, adjust=False).mean()
stock_data['EMA_30'] = stock_data['close'].ewm(span=30, adjust=False).mean()

# RSI
delta = stock_data['close'].diff(1)
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / loss
stock_data['RSI'] = 100 - (100 / (1 + rs))

# Bollinger Bands
stock_data['Bollinger_Mid'] = stock_data['SMA_20']
stock_data['Bollinger_Upper'] = stock_data['Bollinger_Mid'] + 2 * stock_data['close'].rolling(window=20).std()
stock_data['Bollinger_Lower'] = stock_data['Bollinger_Mid'] - 2 * stock_data['close'].rolling(window=20).std()

# Price Change (Lag Feature)
stock_data['Price_Change'] = stock_data['close'].pct_change()

# Volume indicators
stock_data['Volume_Change'] = stock_data['volume'].pct_change()

# Rolling Window Stats
stock_data['Rolling_Mean_Close'] = stock_data['close'].rolling(window=5).mean()
stock_data['Rolling_Std_Close'] = stock_data['close'].rolling(window=5).std()
stock_data['Rolling_Mean_Volume'] = stock_data['volume'].rolling(window=5).mean()
stock_data['Rolling_Std_Volume'] = stock_data['volume'].rolling(window=5).std()


# In[ ]:


# Check the first few rows of the dataset to confirm the new features
print(stock_data.head())


# In[ ]:


# Check the first few rows of the dataset to confirm the new features
print(stock_data.head())


# In[ ]:


# Check if the dataset is loaded correctly and contains data
print(f"Dataset shape: {stock_data.shape}")
print(f"Columns in dataset: {stock_data.columns}")


# In[ ]:


# Reload the dataset if necessary
import pandas as pd

# If you already have the file saved locally
stock_data = pd.read_csv("path_to_your_data.csv", index_col="Date")

# Ensure correct column names and convert 'Date' to datetime if needed
stock_data['Date'] = pd.to_datetime(stock_data.index)


# In[ ]:


# Reload the dataset if necessary
import pandas as pd

# If you already have the file saved locally
stock_data = pd.read_csv(r"path_to_your_data.csv", index_col="Date")

# Ensure correct column names and convert 'Date' to datetime if needed
stock_data['Date'] = pd.to_datetime(stock_data.index)


# In[140]:


import yfinance as yf
import pandas as pd

# List of tickers
tickers = [
    "AAPL", "GOOG", "MSFT", "AMZN", "TSLA", "NFLX", "META", "NVDA", "SPY",
    "DIS", "BA", "V", "GS", "IBM", "INTC", "AMD", "WMT", "JNJ", "PG"
]

# Define the start and end dates
start_date = "2010-01-01"
end_date = "2024-11-21"

# Create an empty list to store data
stock_data_list = []

# Download data for each ticker
for ticker in tickers:
    print(f"Downloading data for {ticker}...")
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    stock_data['Ticker'] = ticker  # Add ticker as a column to identify each stock
    stock_data_list.append(stock_data)

# Combine all stock data into a single DataFrame
all_stock_data = pd.concat(stock_data_list)

# Remove duplicate index entries (if any)
all_stock_data = all_stock_data[~all_stock_data.index.duplicated(keep='first')]

# Reset index (optional)
all_stock_data.reset_index(inplace=True)

# Check the first few rows of the combined data
print(all_stock_data.head())

# Export the combined data to a CSV file
all_stock_data.to_csv("all_stock_data.csv", index=False)


# In[141]:


import yfinance as yf
import pandas as pd

# List of tickers
tickers = [
    "AAPL", "GOOG", "MSFT", "AMZN", "TSLA", "NFLX", "META", "NVDA", "SPY",
    "DIS", "BA", "V", "GS", "IBM", "INTC", "AMD", "WMT", "JNJ", "PG"
]

# Define the start and end dates
start_date = "2010-01-01"
end_date = "2024-11-21"

# Create an empty list to store data
stock_data_list = []

# Download data for each ticker
for ticker in tickers:
    print(f"Downloading data for {ticker}...")
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    stock_data['Ticker'] = ticker  # Add ticker as a column to identify each stock
    stock_data_list.append(stock_data)

# Combine all stock data into a single DataFrame
all_stock_data = pd.concat(stock_data_list)

# Check the first few rows of the combined data
print(all_stock_data.head())

# If needed, reset index for clean DataFrame
all_stock_data.reset_index(inplace=True)

# Check the combined DataFrame
print(all_stock_data.head())

# Export the combined data to a CSV file
all_stock_data.to_csv("all_stock_data.csv", index=False)


# In[142]:


import yfinance as yf
import pandas as pd

# List of tickers
tickers = [
    "AAPL", "GOOG", "MSFT", "AMZN", "TSLA", "NFLX", "META", "NVDA", "SPY",
    "DIS", "BA", "V", "GS", "IBM", "INTC", "AMD", "WMT", "JNJ", "PG"
]

# Define the start and end dates
start_date = "2010-01-01"
end_date = "2024-11-21"

# Create an empty list to store data
stock_data_list = []

# Download data for each ticker
for ticker in tickers:
    print(f"Downloading data for {ticker}...")
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    
    # Rename columns to include the ticker symbol to avoid duplication
    stock_data.columns = [f"{col}_{ticker}" for col in stock_data.columns]
    
    # Add the ticker symbol as a column
    stock_data['Ticker'] = ticker
    
    # Append the data to the list
    stock_data_list.append(stock_data)

# Combine all stock data into a single DataFrame
all_stock_data = pd.concat(stock_data_list, axis=0, join='outer')

# Reset index to get the Date as a column again
all_stock_data.reset_index(inplace=True)

# Check the combined data
print(all_stock_data.head())

# Export the combined data to a CSV file
all_stock_data.to_csv("all_stock_data_combined.csv", index=False)


# In[143]:


import yfinance as yf
import pandas as pd

# List of tickers
tickers = [
    "AAPL", "GOOG", "MSFT", "AMZN", "TSLA", "NFLX", "META", "NVDA", "SPY",
    "DIS", "BA", "V", "GS", "IBM", "INTC", "AMD", "WMT", "JNJ", "PG"
]

# Define the start and end dates
start_date = "2010-01-01"
end_date = "2024-11-21"

# Create an empty list to store data
stock_data_list = []

# Download data for each ticker
for ticker in tickers:
    print(f"Downloading data for {ticker}...")
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    
    # Rename columns to include the ticker symbol to avoid duplication
    stock_data.columns = [f"{col}_{ticker}" for col in stock_data.columns]
    
    # Add the ticker symbol as a column
    stock_data['Ticker'] = ticker
    
    # Append the data to the list
    stock_data_list.append(stock_data)

# Combine all stock data into a single DataFrame
all_stock_data = pd.concat(stock_data_list, axis=0, join='outer')

# Reset index to get the Date as a column again
all_stock_data.reset_index(inplace=True)

# Clean up column names (removing tuple formatting)
all_stock_data.columns = [col[0] if isinstance(col, tuple) else col for col in all_stock_data.columns]

# Check the combined data
print(all_stock_data.head())

# Export the combined data to a CSV file
all_stock_data.to_csv("all_stock_data_combined.csv", index=False)


# In[144]:


import yfinance as yf
import pandas as pd

# List of tickers
tickers = [
    "AAPL", "GOOG", "MSFT", "AMZN", "TSLA", "NFLX", "META", "NVDA", "SPY",
    "DIS", "BA", "V", "GS", "IBM", "INTC", "AMD", "WMT", "JNJ", "PG"
]

# Define the start and end dates
start_date = "2010-01-01"
end_date = "2024-11-21"

# Create an empty list to store data
stock_data_list = []

# Download data for each ticker
for ticker in tickers:
    print(f"Downloading data for {ticker}...")
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    
    # Rename columns to include the ticker symbol to avoid duplication
    stock_data.columns = [f"{col}_{ticker}" for col in stock_data.columns]
    
    # Add the ticker symbol as a column
    stock_data['Ticker'] = ticker
    
    # Append the data to the list
    stock_data_list.append(stock_data)

# Combine all stock data into a single DataFrame
all_stock_data = pd.concat(stock_data_list, axis=0, join='outer')

# Reset index to get the Date as a column again
all_stock_data.reset_index(inplace=True)

# Clean up column names (remove tuple structure)
# Extract only the first part of the column name in case of a tuple
all_stock_data.columns = [col[0] if isinstance(col, tuple) else col for col in all_stock_data.columns]

# Now, rename the columns for clarity, if needed
all_stock_data.columns = [col.split(',')[0].strip().replace("'", "") for col in all_stock_data.columns]

# Check the combined data
print(all_stock_data.head())

# Export the combined data to a CSV file
all_stock_data.to_csv("all_stock_data_combined.csv", index=False)


# In[145]:


import yfinance as yf
import pandas as pd

# List of tickers
tickers = [
    "AAPL", "GOOG", "MSFT", "AMZN", "TSLA", "NFLX", "META", "NVDA", "SPY",
    "DIS", "BA", "V", "GS", "IBM", "INTC", "AMD", "WMT", "JNJ", "PG"
]

# Define the start and end dates
start_date = "2010-01-01"
end_date = "2024-11-21"

# Create an empty list to store data
stock_data_list = []

# Download data for each ticker
for ticker in tickers:
    print(f"Downloading data for {ticker}...")
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    
    # Rename columns to include the ticker symbol to avoid duplication
    stock_data.columns = [f"{col}_{ticker}" for col in stock_data.columns]
    
    # Add the ticker symbol as a column
    stock_data['Ticker'] = ticker
    
    # Append the data to the list
    stock_data_list.append(stock_data)

# Combine all stock data into a single DataFrame
all_stock_data = pd.concat(stock_data_list, axis=0, join='outer')

# Reset index to get the Date as a column again
all_stock_data.reset_index(inplace=True)

# Clean up column names by removing unwanted characters and ensuring consistency
all_stock_data.columns = [col.replace("(", "").replace(")", "").replace("'", "") for col in all_stock_data.columns]

# Check the cleaned data
print(all_stock_data.head())

# Export the combined data to a CSV file
all_stock_data.to_csv("all_stock_data_combined.csv", index=False)


# In[146]:


import yfinance as yf
import pandas as pd

# List of tickers
tickers = [
    "AAPL", "GOOG", "MSFT", "AMZN", "TSLA", "NFLX", "META", "NVDA", "SPY",
    "DIS", "BA", "V", "GS", "IBM", "INTC", "AMD", "WMT", "JNJ", "PG"
]

# Define the start and end dates
start_date = "2010-01-01"
end_date = "2024-11-21"

# Create an empty list to store data
stock_data_list = []

# Download data for each ticker
for ticker in tickers:
    print(f"Downloading data for {ticker}...")
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    
    # Check if multi-level columns exist and flatten them
    if isinstance(stock_data.columns, pd.MultiIndex):
        stock_data.columns = [f"{col[0]}_{ticker}" for col in stock_data.columns]
    else:
        stock_data.columns = [f"{col}_{ticker}" for col in stock_data.columns]
    
    # Add the ticker symbol as a column for easy reference
    stock_data['Ticker'] = ticker
    
    # Append the data to the list
    stock_data_list.append(stock_data)

# Combine all stock data into a single DataFrame
all_stock_data = pd.concat(stock_data_list, axis=0, join='outer')

# Reset index to get the Date as a column again
all_stock_data.reset_index(inplace=True)

# Clean up column names by removing any unwanted characters
all_stock_data.columns = [col.replace("(", "").replace(")", "").replace("'", "") for col in all_stock_data.columns]

# Check the cleaned data
print(all_stock_data.head())

# Export the combined data to a CSV file
all_stock_data.to_csv("all_stock_data_cleaned.csv", index=False)


# In[147]:


import yfinance as yf
import pandas as pd

# List of tickers
tickers = [
    "AAPL", "GOOG", "MSFT", "AMZN", "TSLA", "NFLX", "META", "NVDA", "SPY",
    "DIS", "BA", "V", "GS", "IBM", "INTC", "AMD", "WMT", "JNJ", "PG"
]

# Define the start and end dates
start_date = "2010-01-01"
end_date = "2024-11-21"

# Create an empty DataFrame to store the combined data
all_stock_data = pd.DataFrame()

# Download data for each ticker and clean it
for ticker in tickers:
    print(f"Downloading data for {ticker}...")
    
    # Download stock data using yfinance
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    
    # Drop any rows with NaN values (optional based on your needs)
    stock_data = stock_data.dropna()
    
    # Rename the columns to include the ticker symbol for easy identification
    stock_data.columns = [f"{col}_{ticker}" for col in stock_data.columns]
    
    # Reset the index to have 'Date' as a column
    stock_data.reset_index(inplace=True)
    
    # Merge the stock data into the main DataFrame
    if all_stock_data.empty:
        all_stock_data = stock_data
    else:
        all_stock_data = pd.merge(all_stock_data, stock_data, on="Date", how="outer")

# Check the structure of the combined data
print(all_stock_data.head())

# Export the cleaned and merged data to a CSV file
all_stock_data.to_csv("cleaned_stock_data.csv", index=False)

print("Data processing complete and saved to 'cleaned_stock_data.csv'.")


# In[148]:


import pandas as pd
df=pd.read_csv(r"C:\Users\hp\cleaned_stock_data_for_another_machine_learning_model.csv")
df


# In[149]:


# Check for NaN values in the dataset
nan_summary = all_stock_data.isna().sum()

# Display the count of NaN values in each column
print("NaN Summary for each column:")
print(nan_summary)

# Remove rows with NaN values (if any remain)
# If you want to drop rows with NaN in any column, you can use:
all_stock_data_clean = all_stock_data.dropna()

# Alternatively, if you want to fill NaNs with a specific value (like the last available value)
# all_stock_data_clean = all_stock_data.fillna(method='ffill')  # Forward fill

# Ensure that there are no duplicate rows, especially for 'Date'
all_stock_data_clean = all_stock_data_clean.drop_duplicates(subset=['Date'])

# Reset the index to avoid gaps after cleaning
all_stock_data_clean.reset_index(drop=True, inplace=True)

# Verify the cleaned data
print(f"Cleaned data shape: {all_stock_data_clean.shape}")
print(all_stock_data_clean.head())

# Optionally, export the cleaned data to a new CSV file
all_stock_data_clean.to_csv("cleaned_stock_data_no_nans.csv", index=False)

print("Data cleaning complete and saved to 'cleaned_stock_data_no_nans.csv'.")


# In[150]:


import os

# Get current working directory
current_path = os.getcwd()
print("Current working directory:", current_path)


# In[151]:


import os

# Specify the file name
file_name = "cleaned_stock_data_no_nans.csv"

# Get the absolute path
file_path = os.path.abspath(file_name)
print("Absolute file path:", file_path)


# In[152]:


import pandas as pd
df=pd.read_csv(r'C:\Users\hp\cleaned_stock_data_no_nans.csv')
df


# In[153]:


# Check for duplicate rows
duplicates = df.duplicated().sum()
print(f"Duplicate rows: {duplicates}")

# Remove duplicates if any
df = df.drop_duplicates()


# In[154]:


# Verify that stock price columns do not contain negative values
stock_columns = ['Adj Close', 'Close', 'High', 'Low', 'Open']
for col in stock_columns:
    invalid_values = df[df[col] < 0]
    print(f"Invalid values in {col}:")
    print(invalid_values)


# In[155]:


# Rename columns if necessary
df = df.rename(columns={'OldColumnName': 'NewColumnName'})


# In[156]:


# Example for checking outliers using IQR
Q1 = df['Close'].quantile(0.25)
Q3 = df['Close'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outliers = df[(df['Close'] < lower_bound) | (df['Close'] > upper_bound)]
print(outliers)


# In[157]:


# Print column names to check for any inconsistencies
print(df.columns)


# In[158]:


# Remove leading and trailing spaces from column names
df.columns = df.columns.str.strip()


# In[159]:


# Check if the relevant columns are present
print('Adj Close' in df.columns)
print('Close' in df.columns)


# In[160]:


# Print the first few columns to inspect their names
print(df.columns[:20])  # Prints the first 20 column names



# In[161]:


# Flatten the multi-level column names
df.columns = ['_'.join(col) if isinstance(col, tuple) else col for col in df.columns]


# In[162]:


# Print the first few column names to verify the flattening
print(df.columns[:20])  # Check the first 20 column names


# In[163]:


# Flatten the column names manually
df.columns = [f"{col[0]}_{col[1]}" if isinstance(col, tuple) else col for col in df.columns]


# In[164]:


# Verify the column names after flattening
print(df.columns[:20])  # Print first 20 column names for verification


# In[165]:


# Flatten the multi-level columns explicitly by joining the tuple elements
df.columns = ['_'.join(col) if isinstance(col, tuple) else col for col in df.columns]


# In[166]:


# Check the first few column names after flattening
print(df.columns[:20])  # Print first 20 columns to check


# In[167]:


# If columns are multi-level, flatten using pd.MultiIndex
if isinstance(df.columns, pd.MultiIndex):
    df.columns = ['_'.join(col) for col in df.columns]

# Verify the first few column names after flattening
print(df.columns[:20])  # Print first 20 columns to check


# In[168]:


# To access a specific column (e.g., 'Adj Close' for 'AAPL')
df[('Adj Close', 'AAPL')_AAPL]


# In[169]:


if isinstance(df.columns, pd.MultiIndex):
    df.columns = ['_'.join(col) for col in df.columns]


# In[170]:


print(df.columns[:20])  # Print the first 20 columns to inspect


# In[171]:


import pandas as pd

# Load your dataset (if not already loaded)
# df = pd.read_csv('your_data.csv')

# Check for missing values
print("\nMissing values per column:\n", df.isnull().sum())

# Fill missing values if needed (e.g., forward fill for stock data or fill with 0)
df = df.fillna(method='ffill', axis=0)  # Forward fill to propagate previous valid value

# Check again for missing values
print("\nMissing values after fill:\n", df.isnull().sum())

# Check for duplicates
print("\nNumber of duplicate rows:", df.duplicated().sum())

# Remove duplicate rows if any
df = df.drop_duplicates()

# Convert 'Date' column to datetime
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

# Verify column types after conversion
print("\nData types:\n", df.dtypes)

# If necessary, convert numeric columns to float (stock data should be numeric)
numeric_columns = df.select_dtypes(include='object').columns
df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')

# Remove rows with invalid data (e.g., NaN after conversion)
df = df.dropna()

# Optionally, check for outliers (we'll skip this step for now but it can be added)
# For example, using Z-scores or IQR (Interquartile Range)

# Preview cleaned data
print("\nCleaned Data Sample:\n", df.head())

# Optionally save the cleaned data
# df.to_csv('cleaned_data.csv', index=False)


# In[172]:


import pandas as pd
from datetime import datetime

# Assume df is the cleaned dataframe
# Generate a unique file name based on the current timestamp
unique_filename = f"cleaned_stock_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

# Save the cleaned DataFrame to the CSV file
df.to_csv(unique_filename, index=False)

print(f"Cleaned data saved to: {unique_filename}")


# In[173]:


import pandas as pd
from datetime import datetime

# Assume df is the cleaned dataframe
# Generate a unique file name based on the current timestamp
unique_filename = f"cleaned_stock_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

# Save the cleaned DataFrame to the CSV file
df.to_csv(unique_filename, index=False)

print(f"Cleaned data saved to: {unique_filename}")


# In[174]:


# Show the first few rows to get an overview
print(df.head())

# Summary statistics for the numerical columns
print(df.describe())

# Check the data types of each column
print(df.dtypes)


# In[175]:


import matplotlib.pyplot as plt

# Plot stock prices for a specific company (e.g., 'AAPL')
plt.figure(figsize=(10,6))
plt.plot(df['Date'], df[("Close", "AAPL")], label="AAPL Close Price", color='blue')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.title('AAPL Stock Price Over Time')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Repeat the plot for other companies as needed
# For example, for 'GOOG'
plt.figure(figsize=(10,6))
plt.plot(df['Date'], df[("Close", "GOOG")], label="GOOG Close Price", color='green')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.title('GOOG Stock Price Over Time')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[176]:


# Example of accessing 'Close' column for 'AAPL'
print(df['('Close', 'AAPL')_AAPL']))


# In[177]:


# Print the column names to verify the structure
print(df.columns)


# In[178]:


# Accessing the 'Close' prices for 'AAPL'
aapl_close = df[('Close', 'AAPL')_AAPL]
print(aapl_close)


# In[179]:


# Accessing 'Close' price for 'AAPL'
aapl_close = df[('Close', 'AAPL') + '_AAPL']
print(aapl_close.head())


# In[180]:


# Accessing 'Close' prices for 'AAPL'
aapl_close = df[('Close', 'AAPL') + '_AAPL']


# In[181]:


# Access 'Close' for 'AAPL' correctly
aapl_close = df[('Close', 'AAPL')]

# Display first few rows of the 'AAPL' Close column
print(aapl_close.head())


# In[182]:


# Access 'Close' prices for 'AAPL' using the exact column structure
aapl_close = df[('Close', 'AAPL')]
print(aapl_close.head())


# In[183]:


# If 'Close' for 'AAPL' is a tuple key, use this approach:
aapl_close = df[('Close', 'AAPL')]  # Make sure the tuple key is correct

# Print out the first few rows
print(aapl_close.head())


# In[184]:


# If 'Close' for 'AAPL' is a tuple key, use this approach:
aapl_close = df[('Close', 'AAPL')]  # Make sure the tuple key is correct

# Print out the first few rows
print(aapl_close.head())


# In[185]:


# Print the actual column names to check their structure
for col in df.columns:
    print(col)


# In[186]:


# Accessing 'Adj Close' for AAPL
df[('Adj Close', 'AAPL')]


# In[187]:


# List of tickers from the columns
tickers = ['AAPL', 'GOOG', 'MSFT', 'AMZN', 'TSLA', 'NFLX', 'META', 'NVDA', 'SPY', 'DIS', 'BA', 'V', 'GS', 'IBM', 'INTC', 'AMD', 'WMT', 'JNJ', 'PG']

# Loop through each ticker to clean and process the data
for ticker in tickers:
    # Accessing and cleaning 'Adj Close' for each ticker
    df[('Adj Close', ticker)] = df[('Adj Close', ticker)].fillna(method='ffill')  # Example: filling missing values with forward fill
    
    # Repeat the process for other columns as necessary (e.g., Close, Volume, etc.)
    df[('Close', ticker)] = df[('Close', ticker)].fillna(method='ffill')
    df[('High', ticker)] = df[('High', ticker)].fillna(method='ffill')
    df[('Low', ticker)] = df[('Low', ticker)].fillna(method='ffill')
    df[('Open', ticker)] = df[('Open', ticker)].fillna(method='ffill')
    df[('Volume', ticker)] = df[('Volume', ticker)].fillna(method='ffill')

# Verify the data
print(df.head())


# In[188]:


# Rename columns by flattening the multi-level structure
df.columns = [f'{col[0]}_{col[1]}' for col in df.columns]

# Verify the column names
print(df.columns)


# In[189]:


# Flatten the multi-level column names
df.columns = [f'{col[0]}_{col[1]}' if isinstance(col, tuple) else col for col in df.columns]

# Verify the new column names
print(df.columns)


# In[190]:


# Print the first few column names and check their structure
print(df.columns[:10])  # Just print the first 10 to get an idea of the structure


# In[191]:


# Check if the columns are tuples or strings
columns = df.columns
print(columns[:10])  # Print a sample of column names to understand their format

# Now flatten the columns properly
flattened_columns = []
for col in columns:
    if isinstance(col, tuple):  # Check if the column name is a tuple
        flattened_columns.append(f'{col[0]}_{col[1]}')  # Join tuple values with an underscore
    else:
        flattened_columns.append(col)  # If it's not a tuple, leave it as is

# Assign the new flattened column names
df.columns = flattened_columns

# Verify the result
print(df.columns[:10])


# In[192]:


# Print the first few rows to get a better understanding of the data
print(df.head())


# In[193]:


# Get info about the dataframe to understand the column data types
df.info()


# In[194]:


# List of original column names as per your previous input
original_columns = [
    'Date', 'Adj Close AAPL', 'Close AAPL', 'High AAPL', 'Low AAPL', 'Open AAPL', 'Volume AAPL',
    'Adj Close GOOG', 'Close GOOG', 'High GOOG', 'Low GOOG', 'Open GOOG', 'Volume GOOG',
    'Adj Close MSFT', 'Close MSFT', 'High MSFT', 'Low MSFT', 'Open MSFT', 'Volume MSFT',
    'Adj Close AMZN', 'Close AMZN', 'High AMZN', 'Low AMZN', 'Open AMZN', 'Volume AMZN',
    'Adj Close TSLA', 'Close TSLA', 'High TSLA', 'Low TSLA', 'Open TSLA', 'Volume TSLA',
    'Adj Close NFLX', 'Close NFLX', 'High NFLX', 'Low NFLX', 'Open NFLX', 'Volume NFLX',
    'Adj Close META', 'Close META', 'High META', 'Low META', 'Open META', 'Volume META',
    'Adj Close NVDA', 'Close NVDA', 'High NVDA', 'Low NVDA', 'Open NVDA', 'Volume NVDA',
    'Adj Close SPY', 'Close SPY', 'High SPY', 'Low SPY', 'Open SPY', 'Volume SPY',
    'Adj Close DIS', 'Close DIS', 'High DIS', 'Low DIS', 'Open DIS', 'Volume DIS',
    'Adj Close BA', 'Close BA', 'High BA', 'Low BA', 'Open BA', 'Volume BA',
    'Adj Close V', 'Close V', 'High V', 'Low V', 'Open V', 'Volume V',
    'Adj Close GS', 'Close GS', 'High GS', 'Low GS', 'Open GS', 'Volume GS',
    'Adj Close IBM', 'Close IBM', 'High IBM', 'Low IBM', 'Open IBM', 'Volume IBM',
    'Adj Close INTC', 'Close INTC', 'High INTC', 'Low INTC', 'Open INTC', 'Volume INTC',
    'Adj Close AMD', 'Close AMD', 'High AMD', 'Low AMD', 'Open AMD', 'Volume AMD',
    'Adj Close WMT', 'Close WMT', 'High WMT', 'Low WMT', 'Open WMT', 'Volume WMT',
    'Adj Close JNJ', 'Close JNJ', 'High JNJ', 'Low JNJ', 'Open JNJ', 'Volume JNJ',
    'Adj Close PG', 'Close PG', 'High PG', 'Low PG', 'Open PG', 'Volume PG'
]

# Check the current column names
current_columns = df.columns.tolist()

# Ensure that the number of columns matches before proceeding
if len(current_columns) == len(original_columns):
    # Rename columns
    df.columns = original_columns
else:
    print("Mismatch in the number of columns.")

# Verify the new column names
print(df.columns)


# In[195]:


# Check for missing values in the dataset
missing_data = df.isnull().sum()

# Display columns with missing values
missing_data[missing_data > 0]


# In[196]:


# Fill missing values with forward fill
df = df.fillna(method='ffill')


# In[197]:


# Fill missing values with forward fill
df = df.ffill()


# In[198]:


# Drop rows with any missing values
df = df.dropna()


# In[199]:


# Check for duplicate rows
duplicates = df.duplicated().sum()

# If duplicates exist, drop them
if duplicates > 0:
    df = df.drop_duplicates()


# In[200]:


# Convert 'Date' column to datetime format
df['Date'] = pd.to_datetime(df['Date'])

# Convert columns that should be numeric (if not already)
numeric_columns = df.columns.difference(['Date'])
df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')


# In[201]:


# Save the cleaned dataframe to a new CSV file
df.to_csv("cleaned_stock_data.csv", index=False)


# In[202]:


# Get basic statistics of the data
print(df.describe())


# In[203]:


import matplotlib.pyplot as plt

# Example of plotting the 'Close' prices of AAPL and GOOG
plt.figure(figsize=(10, 5))
plt.plot(df['Date'], df['Close AAPL'], label='AAPL Close')
plt.plot(df['Date'], df['Close GOOG'], label='GOOG Close')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Stock Price Trends of AAPL and GOOG')
plt.legend()
plt.show()


# In[204]:


# Calculate 7-day moving average for AAPL Close price
df['MA_7_AAPL'] = df['Close AAPL'].rolling(window=7).mean()


# In[205]:


# Calculate 7-day moving average and other features
df['MA_7_AAPL'] = df['Close AAPL'].rolling(window=7).mean()

# You can add all new features together using concat
df_new_features = pd.DataFrame({
    'MA_7_AAPL': df['Close AAPL'].rolling(window=7).mean(),
    # Add other features as needed
})

# Concatenate all new features at once
df = pd.concat([df, df_new_features], axis=1)


# In[206]:


df['MA_7_AAPL'] = df['Close AAPL'].rolling(window=7).mean()
df = df.copy()  # This will create a de-fragmented copy of the DataFrame


# In[207]:


# Perform all necessary operations like moving averages and other feature calculations
df['MA_7_AAPL'] = df['Close AAPL'].rolling(window=7).mean()

# Rebuild the DataFrame after modifications
df = df.copy()


# In[208]:


from sklearn.model_selection import train_test_split

# Assuming 'target' is the column you're predicting (e.g., 'Close AAPL' or price change)
X = df.drop(columns=['Date', 'target_column_name'])  # Drop the target column and any non-predictive columns
y = df['target_column_name']  # Target column

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[209]:


from sklearn.model_selection import train_test_split

# Assuming the target column is 'Close AAPL'
target_column = 'Close AAPL'

# Drop the Date and target columns from the features
X = df.drop(columns=['Date', target_column])  # Drop the target column and any non-predictive columns
y = df[target_column]  # Define the target column

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[210]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Initialize the Linear Regression model
model = LinearRegression()

# Train the model using the training data
model.fit(X_train, y_train)

# Predict on the testing data
y_pred = model.predict(X_test)

# Evaluate the model's performance
mse = mean_squared_error(y_test, y_pred)  # Mean Squared Error
r2 = r2_score(y_test, y_pred)  # R-squared (Goodness of fit)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")


# In[211]:


# Check for missing values in the features and target variable
print(X_train.isnull().sum())  # For features
print(y_train.isnull().sum())  # For target variable


# In[212]:


from sklearn.impute import SimpleImputer

# For features (X)
imputer = SimpleImputer(strategy='mean')  # or 'median'
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

# For target variable (y)
y_train = imputer.fit_transform(y_train.values.reshape(-1, 1)).ravel()  # Use .ravel() to flatten
y_test = imputer.transform(y_test.values.reshape(-1, 1)).ravel()  # Use .ravel() to flatten


# In[213]:


# Drop rows with missing values in features and target
train_data = pd.concat([X_train, y_train], axis=1)
train_data = train_data.dropna()

# Separate the features and target again after dropping
X_train = train_data.drop(columns='target_column_name')
y_train = train_data['target_column_name']


# In[214]:


from sklearn.impute import SimpleImputer

# Impute missing values for features (X)
imputer = SimpleImputer(strategy='mean')  # Or use 'median'
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

# Convert the numpy array back to a pandas DataFrame
X_train = pd.DataFrame(X_train, columns=X_train.columns)  # Reassign columns if needed
X_test = pd.DataFrame(X_test, columns=X_test.columns)  # Reassign columns if needed


# In[215]:


from sklearn.impute import SimpleImputer

# Assuming 'X_train' is a pandas DataFrame with columns
imputer = SimpleImputer(strategy='mean')  # Or use 'median' as needed

# Impute missing values for features (X)
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

# Convert the numpy array back to a pandas DataFrame with the original column names
X_train_imputed = pd.DataFrame(X_train_imputed, columns=X_train.columns)
X_test_imputed = pd.DataFrame(X_test_imputed, columns=X_test.columns)


# In[216]:


from sklearn.impute import SimpleImputer
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Impute missing values in the feature set (X)
imputer = SimpleImputer(strategy='mean')  # You can use 'median' if needed

# Fit and transform for training data
X_train_imputed = imputer.fit_transform(X_train)

# Transform the test data
X_test_imputed = imputer.transform(X_test)

# Step 2: Convert the imputed numpy arrays back to DataFrames
X_train_imputed = pd.DataFrame(X_train_imputed, columns=X_train.columns)
X_test_imputed = pd.DataFrame(X_test_imputed, columns=X_test.columns)

# Step 3: Impute missing values in the target variable (y)
y_train_imputed = imputer.fit_transform(y_train.values.reshape(-1, 1)).ravel()  # Convert to 1D array
y_test_imputed = imputer.transform(y_test.values.reshape(-1, 1)).ravel()  # Convert to 1D array

# Step 4: Train the model using the imputed data
model = LinearRegression()
model.fit(X_train_imputed, y_train_imputed)

# Step 5: Make predictions
y_pred = model.predict(X_test_imputed)

# Step 6: Evaluate the model
mse = mean_squared_error(y_test_imputed, y_pred)
r2 = r2_score(y_test_imputed, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")


# In[217]:


from sklearn.impute import SimpleImputer
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Ensure that the training data is in DataFrame form
X_train = pd.DataFrame(X_train)  # Convert to DataFrame if not already
X_test = pd.DataFrame(X_test)  # Convert to DataFrame if not already

# Step 2: Impute missing values in the feature set (X)
imputer = SimpleImputer(strategy='mean')

# Fit and transform the training data
X_train_imputed = imputer.fit_transform(X_train)

# Transform the test data
X_test_imputed = imputer.transform(X_test)

# Convert the numpy array back to DataFrame and retain original column names
X_train_imputed = pd.DataFrame(X_train_imputed, columns=X_train.columns)
X_test_imputed = pd.DataFrame(X_test_imputed, columns=X_test.columns)

# Step 3: Impute missing values in the target variable (y)
y_train = y_train.values.reshape(-1, 1)  # Convert y_train to 2D array
y_test = y_test.values.reshape(-1, 1)  # Convert y_test to 2D array

# Impute the target variable using the same imputer
y_train_imputed = imputer.fit_transform(y_train).ravel()  # Flatten to 1D
y_test_imputed = imputer.transform(y_test).ravel()  # Flatten to 1D

# Step 4: Train the model using the imputed data
model = LinearRegression()
model.fit(X_train_imputed, y_train_imputed)

# Step 5: Make predictions
y_pred = model.predict(X_test_imputed)

# Step 6: Evaluate the model
mse = mean_squared_error(y_test_imputed, y_pred)
r2 = r2_score(y_test_imputed, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")


# In[218]:


from sklearn.impute import SimpleImputer
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Ensure that the training and test data are in DataFrame form
if isinstance(X_train, np.ndarray):
    X_train = pd.DataFrame(X_train)  # Convert to DataFrame if X_train is a numpy array
if isinstance(X_test, np.ndarray):
    X_test = pd.DataFrame(X_test)  # Convert to DataFrame if X_test is a numpy array

# Step 2: Impute missing values in the feature set (X)
imputer = SimpleImputer(strategy='mean')

# Fit and transform the training data
X_train_imputed = imputer.fit_transform(X_train)

# Transform the test data
X_test_imputed = imputer.transform(X_test)

# Convert the numpy array back to DataFrame and retain original column names
X_train_imputed = pd.DataFrame(X_train_imputed, columns=X_train.columns)
X_test_imputed = pd.DataFrame(X_test_imputed, columns=X_test.columns)

# Step 3: Impute missing values in the target variable (y)
if isinstance(y_train, np.ndarray):
    y_train = pd.Series(y_train)  # Convert y_train to Series if it's a numpy array
if isinstance(y_test, np.ndarray):
    y_test = pd.Series(y_test)  # Convert y_test to Series if it's a numpy array

# Impute the target variable using the same imputer
y_train_imputed = imputer.fit_transform(y_train.values.reshape(-1, 1)).ravel()  # Flatten to 1D
y_test_imputed = imputer.transform(y_test.values.reshape(-1, 1)).ravel()  # Flatten to 1D

# Step 4: Train the model using the imputed data
model = LinearRegression()
model.fit(X_train_imputed, y_train_imputed)

# Step 5: Make predictions
y_pred = model.predict(X_test_imputed)

# Step 6: Evaluate the model
mse = mean_squared_error(y_test_imputed, y_pred)
r2 = r2_score(y_test_imputed, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")


# In[219]:


import numpy as np
from sklearn.impute import SimpleImputer
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Ensure that the training and test data are in DataFrame form
if isinstance(X_train, np.ndarray):
    X_train = pd.DataFrame(X_train)  # Convert to DataFrame if X_train is a numpy array
if isinstance(X_test, np.ndarray):
    X_test = pd.DataFrame(X_test)  # Convert to DataFrame if X_test is a numpy array

# Step 2: Impute missing values in the feature set (X)
imputer = SimpleImputer(strategy='mean')

# Fit and transform the training data
X_train_imputed = imputer.fit_transform(X_train)

# Transform the test data
X_test_imputed = imputer.transform(X_test)

# Convert the numpy array back to DataFrame and retain original column names
X_train_imputed = pd.DataFrame(X_train_imputed, columns=X_train.columns)
X_test_imputed = pd.DataFrame(X_test_imputed, columns=X_test.columns)

# Step 3: Impute missing values in the target variable (y)
if isinstance(y_train, np.ndarray):
    y_train = pd.Series(y_train)  # Convert y_train to Series if it's a numpy array
if isinstance(y_test, np.ndarray):
    y_test = pd.Series(y_test)  # Convert y_test to Series if it's a numpy array

# Impute the target variable using the same imputer
y_train_imputed = imputer.fit_transform(y_train.values.reshape(-1, 1)).ravel()  # Flatten to 1D
y_test_imputed = imputer.transform(y_test.values.reshape(-1, 1)).ravel()  # Flatten to 1D

# Step 4: Train the model using the imputed data
model = LinearRegression()
model.fit(X_train_imputed, y_train_imputed)

# Step 5: Make predictions
y_pred = model.predict(X_test_imputed)

# Step 6: Evaluate the model
mse = mean_squared_error(y_test_imputed, y_pred)
r2 = r2_score(y_test_imputed, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")


# In[221]:


from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

# Step 1: Final Preprocessing
# Make sure there are no missing values
df.fillna(method='ffill', inplace=True)

# Normalize or scale the data
scaler = StandardScaler()
X = df.drop(columns=['target_column_name', 'Date'])  # remove target column
y = df['target_column_name']

X_scaled = scaler.fit_transform(X)

# Step 2: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, shuffle=False)

# Step 3: Model Training (Using XGBoost as an example)
model = xgb.XGBRegressor(objective='reg:squarederror')

# Optionally, tune hyperparameters (example)
# param_grid = {'max_depth': [3, 5, 7], 'n_estimators': [100, 200, 300]}
# grid_search = GridSearchCV(model, param_grid, cv=5)
# grid_search.fit(X_train, y_train)
# model = grid_search.best_estimator_

model.fit(X_train, y_train)

# Step 4: Model Evaluation
y_pred = model.predict(X_test)

# Evaluate with metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")


# In[222]:


print(df.columns)


# In[223]:


X = df.drop(columns=['Close AAPL', 'Date'])  # Drop the target column and the 'Date' column
y = df['Close AAPL']  # Set your target column (replace 'Close AAPL' with your actual target)


# In[224]:


print(df.columns)


# In[225]:


# Assuming the target column is 'Close AAPL'
X = df.drop(columns=['Close AAPL', 'Date'])  # Drop the target column and the 'Date' column
y = df['Close AAPL']  # Set the target column for prediction


# In[226]:


X = df.drop(columns=['Close AAPL', 'Date'])  # Drop 'Close AAPL' and 'Date' columns
y = df['Close AAPL']  # 'Close AAPL' is the target variable


# In[228]:


# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Load the dataset (replace 'your_cleaned_data.csv' with the actual filename if needed)
df = pd.read_csv('cleaned_stock_data.csv')

# Display the columns to identify the target column and ensure proper data
print(df.columns)

# Step 1: Define the target column and feature columns
target_column = 'Close AAPL'  # This is the column you want to predict
X = df.drop(columns=[target_column, 'Date'])  # Drop target column and 'Date' column
y = df[target_column]  # Target column for prediction

# Step 2: Handle missing values by forward filling
df = df.fillna(method='ffill')

# Step 3: Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Scale the features using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Fit and transform on training data
X_test_scaled = scaler.transform(X_test)  # Only transform on test data

# Step 5: Initialize and train the model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Step 6: Make predictions using the trained model
y_pred = model.predict(X_test_scaled)

# Step 7: Evaluate the model's performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Output the results
print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

# Optionally, save the trained model for future use
import joblib
joblib.dump(model, 'stock_price_prediction_model.pkl')


# In[229]:


# Load the trained model
model = joblib.load('stock_price_prediction_model.pkl')

# Prepare new data for prediction (ensure it is processed the same way as training data)
new_data = pd.DataFrame({
    'Adj Close AAPL': [your_value],  # Replace with actual values
    'Close AAPL': [your_value],
    # Add other feature columns as per your data
})

# Don't forget to scale the new data
new_data_scaled = scaler.transform(new_data)

# Make a prediction
prediction = model.predict(new_data_scaled)
print(f'Predicted stock price: {prediction}')


# In[231]:


# Import the necessary libraries
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the trained model
model = joblib.load('stock_price_prediction_model.pkl')

# Example values for the input features (replace these with actual values you want to test)
new_data = pd.DataFrame({
    'Adj Close AAPL': [175.25],  # Example: Adjusted close for AAPL
    'Close AAPL': [174.50],      # Example: Close price for AAPL
    'High AAPL': [176.00],       # Example: High price for AAPL
    'Low AAPL': [173.50],        # Example: Low price for AAPL
    'Open AAPL': [175.00],       # Example: Opening price for AAPL
    'Volume AAPL': [55000000],   # Example: Volume for AAPL
    'Adj Close GOOG': [1350.30], # Example: Adjusted close for GOOG
    'Close GOOG': [1345.80],     # Example: Close price for GOOG
    'High GOOG': [1360.50],      # Example: High price for GOOG
    'Low GOOG': [1340.20],       # Example: Low price for GOOG
    'Open GOOG': [1349.00],      # Example: Opening price for GOOG
    'Volume GOOG': [1500000],    # Example: Volume for GOOG
    # Add the rest of the stock data columns in the same way
})

# Load the scaler (use the same one used during training)
scaler = joblib.load('stock_price_prediction_model.pkl')

# Scale the new data using the same scaler
new_data_scaled = scaler.transform(new_data)

# Make a prediction with the model
prediction = model.predict(new_data_scaled)

# Print the predicted stock price
print(f'Predicted stock price: {prediction[0]}')


# In[233]:


# Import necessary libraries
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the trained model and scaler
model = joblib.load('stock_price_prediction_model.pkl')
scaler = joblib.load('scaler.pkl')

# Example values for the input features (replace these with actual values you want to test)
new_data = pd.DataFrame({
    'Adj Close AAPL': [175.25],  # Example: Adjusted close for AAPL
    'Close AAPL': [174.50],      # Example: Close price for AAPL
    'High AAPL': [176.00],       # Example: High price for AAPL
    'Low AAPL': [173.50],        # Example: Low price for AAPL
    'Open AAPL': [175.00],       # Example: Opening price for AAPL
    'Volume AAPL': [55000000],   # Example: Volume for AAPL
    'Adj Close GOOG': [1350.30], # Example: Adjusted close for GOOG
    'Close GOOG': [1345.80],     # Example: Close price for GOOG
    'High GOOG': [1360.50],      # Example: High price for GOOG
    'Low GOOG': [1340.20],       # Example: Low price for GOOG
    'Open GOOG': [1349.00],      # Example: Opening price for GOOG
    'Volume GOOG': [1500000],    # Example: Volume for GOOG
    # Add the rest of the stock data columns in the same way
})

# Scale the new data using the same scaler used during training
new_data_scaled = scaler.transform(new_data)

# Make a prediction with the trained model
prediction = model.predict(new_data_scaled)

# Print the predicted stock price
print(f'Predicted stock price: {prediction[0]}')


# In[234]:


from sklearn.preprocessing import StandardScaler
scaler = joblib.load('scaler.pkl')  # Loading the scaler from file


# In[235]:


# Re-train your model (use your training data)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib

# Assuming X and y are your features and target variables
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and fit the scaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Save the scaler and model
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(model, 'stock_price_prediction_model.pkl')


# In[237]:


# Loading the model and scaler
import numpy as np
model = joblib.load('stock_price_prediction_model.pkl')
scaler = joblib.load('scaler.pkl')

# Example X_input: You should replace these values with real stock data for the features used in training
X_input = np.array([[150.2, 149.5, 151.0, 148.8, 150.0, 5000000]])  # Example for AAPL

# Use the scaler to transform your input data for prediction
X_input_scaled = scaler.transform(X_input)  # Assuming X_input is your input data for prediction

# Make predictions
predictions = model.predict(X_input_scaled)



# In[238]:


import numpy as np

# Example X_input: Replace these values with the actual stock data for your prediction
# Ensure that the input has the same number of features as the training data (113 in this case)

# Example of stock data for multiple companies (replace with actual stock data)
# Assuming you're using stock prices for AAPL, GOOG, MSFT, etc. with 6 features each

# Here, we're just using dummy values for demonstration purposes (replace with real stock data)
X_input = np.array([[150.2, 149.5, 151.0, 148.8, 150.0, 5000000,  # AAPL features
                     2800.5, 2795.3, 2810.0, 2785.0, 2802.3, 1000000,  # GOOG features
                     300.5, 298.3, 302.0, 297.5, 299.7, 1200000,  # MSFT features
                     # Add features for other companies here...
                     ]])

# Ensure that the number of columns matches the number of features used in the training (113)
# Your X_input should have 113 features (the same as during training)
print(X_input.shape)  # Should print (1, 113) if you have 113 features

# Now, you can proceed to load the scaler and model as shown before:


# In[239]:


import joblib

# Load the trained model and scaler
model = joblib.load('stock_price_prediction_model.pkl')
scaler = joblib.load('scaler.pkl')

# Transform the input data using the scaler
X_input_scaled = scaler.transform(X_input)

# Make a prediction
predictions = model.predict(X_input_scaled)

# Print the predicted stock price
print("Predicted Stock Price:", predictions)



# In[240]:


import numpy as np

# Example input for 113 features: Replace these values with the actual stock data for your prediction
# Example: 19 stocks, 6 features per stock (Open, Close, High, Low, Volume, Adj Close)

# This is just a sample with dummy values (replace with real stock data)
X_input = np.array([[
    150.2, 149.5, 151.0, 148.8, 150.0, 5000000,   # AAPL features (6 columns)
    2800.5, 2795.3, 2810.0, 2785.0, 2802.3, 1000000, # GOOG features (6 columns)
    300.5, 298.3, 302.0, 297.5, 299.7, 1200000,     # MSFT features (6 columns)
    134.5, 133.0, 136.0, 132.5, 134.2, 900000,      # TSLA features (6 columns)
    210.3, 208.5, 212.0, 207.0, 210.0, 800000,      # AMZN features (6 columns)
    # Add features for other stocks (up to 19 stocks in total)
]])

# Check if the shape is (1, 113)
print(X_input.shape)  # It should print (1, 113) to match the model's expected input

# Ensure that the scaler and model are loaded correctly
import joblib
scaler = joblib.load('scaler.pkl')
model = joblib.load('stock_price_prediction_model.pkl')

# Transform the input data using the scaler
X_input_scaled = scaler.transform(X_input)

# Make the prediction
predictions = model.predict(X_input_scaled)

# Print the predicted stock price
print("Predicted Stock Price:", predictions)


# In[241]:


import numpy as np

# Example input for 19 stocks, each with 6 features
X_input = np.array([[
    150.2, 149.5, 151.0, 148.8, 150.0, 5000000,   # AAPL: Open, Close, High, Low, Volume, Adj Close
    2800.5, 2795.3, 2810.0, 2785.0, 2802.3, 1000000, # GOOG: Open, Close, High, Low, Volume, Adj Close
    300.5, 298.3, 302.0, 297.5, 299.7, 1200000,     # MSFT: Open, Close, High, Low, Volume, Adj Close
    134.5, 133.0, 136.0, 132.5, 134.2, 900000,      # TSLA: Open, Close, High, Low, Volume, Adj Close
    210.3, 208.5, 212.0, 207.0, 210.0, 800000,      # AMZN: Open, Close, High, Low, Volume, Adj Close
    140.5, 139.2, 141.0, 138.0, 139.5, 700000,      # FB: Open, Close, High, Low, Volume, Adj Close
    180.2, 179.8, 182.0, 178.0, 180.3, 600000,      # NVDA: Open, Close, High, Low, Volume, Adj Close
    250.3, 248.5, 255.0, 247.0, 249.7, 500000,      # INTC: Open, Close, High, Low, Volume, Adj Close
    50.5, 49.9, 52.0, 48.5, 51.0, 400000,           # DIS: Open, Close, High, Low, Volume, Adj Close
    300.8, 298.2, 305.0, 295.0, 300.0, 300000,      # PYPL: Open, Close, High, Low, Volume, Adj Close
    220.4, 218.0, 222.0, 217.0, 220.1, 350000,      # SNAP: Open, Close, High, Low, Volume, Adj Close
    180.2, 179.6, 181.5, 178.5, 179.8, 600000,      # AMD: Open, Close, High, Low, Volume, Adj Close
    130.7, 128.5, 132.0, 127.0, 130.1, 450000,      # VZ: Open, Close, High, Low, Volume, Adj Close
    50.2, 48.5, 51.5, 47.0, 49.2, 500000,           # BA: Open, Close, High, Low, Volume, Adj Close
    120.3, 119.0, 122.0, 118.0, 119.5, 550000,      # IBM: Open, Close, High, Low, Volume, Adj Close
    210.1, 208.0, 212.0, 206.5, 209.0, 700000,      # GS: Open, Close, High, Low, Volume, Adj Close
    240.0, 239.5, 243.0, 238.0, 240.2, 800000,      # JNJ: Open, Close, High, Low, Volume, Adj Close
    400.0, 399.5, 405.0, 398.0, 400.1, 900000,      # PFE: Open, Close, High, Low, Volume, Adj Close
    300.4, 298.9, 303.0, 297.0, 299.5, 850000       # PG: Open, Close, High, Low, Volume, Adj Close
]])

# Verify the shape
print(X_input.shape)  # Should print (1, 113)

# If needed, use your trained scaler to transform this input
import joblib
scaler = joblib.load('scaler.pkl')  # Load the previously saved scaler
X_input_scaled = scaler.transform(X_input)

# Load your trained model
model = joblib.load('stock_price_prediction_model.pkl')

# Make the prediction
predictions = model.predict(X_input_scaled)

# Output the prediction result
print("Predicted Stock Price:", predictions)


# In[242]:


print(X_input.shape)  # Should print (1, 113)


# In[243]:


import numpy as np

# Example input for 19 stocks, each with 6 features (Open, Close, High, Low, Volume, Adj Close)
X_input = np.array([[
    150.2, 149.5, 151.0, 148.8, 150.0, 5000000,   # AAPL: Open, Close, High, Low, Volume, Adj Close
    2800.5, 2795.3, 2810.0, 2785.0, 2802.3, 1000000, # GOOG: Open, Close, High, Low, Volume, Adj Close
    300.5, 298.3, 302.0, 297.5, 299.7, 1200000,     # MSFT: Open, Close, High, Low, Volume, Adj Close
    134.5, 133.0, 136.0, 132.5, 134.2, 900000,      # TSLA: Open, Close, High, Low, Volume, Adj Close
    210.3, 208.5, 212.0, 207.0, 210.0, 800000,      # AMZN: Open, Close, High, Low, Volume, Adj Close
    140.5, 139.2, 141.0, 138.0, 139.5, 700000,      # FB: Open, Close, High, Low, Volume, Adj Close
    180.2, 179.8, 182.0, 178.0, 180.3, 600000,      # NVDA: Open, Close, High, Low, Volume, Adj Close
    250.3, 248.5, 255.0, 247.0, 249.7, 500000,      # INTC: Open, Close, High, Low, Volume, Adj Close
    50.5, 49.9, 52.0, 48.5, 51.0, 400000,           # DIS: Open, Close, High, Low, Volume, Adj Close
    300.8, 298.2, 305.0, 295.0, 300.0, 300000,      # PYPL: Open, Close, High, Low, Volume, Adj Close
    220.4, 218.0, 222.0, 217.0, 220.1, 350000,      # SNAP: Open, Close, High, Low, Volume, Adj Close
    180.2, 179.6, 181.5, 178.5, 179.8, 600000,      # AMD: Open, Close, High, Low, Volume, Adj Close
    130.7, 128.5, 132.0, 127.0, 130.1, 450000,      # VZ: Open, Close, High, Low, Volume, Adj Close
    50.2, 48.5, 51.5, 47.0, 49.2, 500000,           # BA: Open, Close, High, Low, Volume, Adj Close
    120.3, 119.0, 122.0, 118.0, 119.5, 550000,      # IBM: Open, Close, High, Low, Volume, Adj Close
    210.1, 208.0, 212.0, 206.5, 209.0, 700000,      # GS: Open, Close, High, Low, Volume, Adj Close
    240.0, 239.5, 243.0, 238.0, 240.2, 800000,      # JNJ: Open, Close, High, Low, Volume, Adj Close
    400.0, 399.5, 405.0, 398.0, 400.1, 900000,      # PFE: Open, Close, High, Low, Volume, Adj Close
    300.4, 298.9, 303.0, 297.0, 299.5, 850000       # PG: Open, Close, High, Low, Volume, Adj Close
]])

# Check the shape of the input
print(X_input.shape)  # Should print (1, 113)

# If needed, use your trained scaler to transform this input
import joblib
scaler = joblib.load('scaler.pkl')  # Load the previously saved scaler
X_input_scaled = scaler.transform(X_input)

# Load your trained model
model = joblib.load('stock_price_prediction_model.pkl')

# Make the prediction
predictions = model.predict(X_input_scaled)

# Output the prediction result
print("Predicted Stock Price:", predictions)


# In[244]:


import numpy as np

# Example input for 19 stocks, each with 6 features (Open, Close, High, Low, Volume, Adj Close)
X_input = np.array([[
    150.2, 149.5, 151.0, 148.8, 150.0, 5000000,   # AAPL: Open, Close, High, Low, Volume, Adj Close
    2800.5, 2795.3, 2810.0, 2785.0, 2802.3, 1000000, # GOOG: Open, Close, High, Low, Volume, Adj Close
    300.5, 298.3, 302.0, 297.5, 299.7, 1200000,     # MSFT: Open, Close, High, Low, Volume, Adj Close
    134.5, 133.0, 136.0, 132.5, 134.2, 900000,      # TSLA: Open, Close, High, Low, Volume, Adj Close
    210.3, 208.5, 212.0, 207.0, 210.0, 800000,      # AMZN: Open, Close, High, Low, Volume, Adj Close
    140.5, 139.2, 141.0, 138.0, 139.5, 700000,      # FB: Open, Close, High, Low, Volume, Adj Close
    180.2, 179.8, 182.0, 178.0, 180.3, 600000,      # NVDA: Open, Close, High, Low, Volume, Adj Close
    250.3, 248.5, 255.0, 247.0, 249.7, 500000,      # INTC: Open, Close, High, Low, Volume, Adj Close
    50.5, 49.9, 52.0, 48.5, 51.0, 400000,           # DIS: Open, Close, High, Low, Volume, Adj Close
    300.8, 298.2, 305.0, 295.0, 300.0, 300000,      # PYPL: Open, Close, High, Low, Volume, Adj Close
    220.4, 218.0, 222.0, 217.0, 220.1, 350000,      # SNAP: Open, Close, High, Low, Volume, Adj Close
    180.2, 179.6, 181.5, 178.5, 179.8, 600000,      # AMD: Open, Close, High, Low, Volume, Adj Close
    130.7, 128.5, 132.0, 127.0, 130.1, 450000,      # VZ: Open, Close, High, Low, Volume, Adj Close
    50.2, 48.5, 51.5, 47.0, 49.2, 500000,           # BA: Open, Close, High, Low, Volume, Adj Close
    120.3, 119.0, 122.0, 118.0, 119.5, 550000,      # IBM: Open, Close, High, Low, Volume, Adj Close
    210.1, 208.0, 212.0, 206.5, 209.0, 700000,      # GS: Open, Close, High, Low, Volume, Adj Close
    240.0, 239.5, 243.0, 238.0, 240.2, 800000,      # JNJ: Open, Close, High, Low, Volume, Adj Close
    400.0, 399.5, 405.0, 398.0, 400.1, 900000,      # PFE: Open, Close, High, Low, Volume, Adj Close
    300.4, 298.9, 303.0, 297.0, 299.5, 850000       # PG: Open, Close, High, Low, Volume, Adj Close
]])

# Check the shape of the input
print(X_input.shape)  # Should print (1, 113)

# If needed, use your trained scaler to transform this input
import joblib
scaler = joblib.load('scaler.pkl')  # Load the previously saved scaler
X_input_scaled = scaler.transform(X_input)

# Load your trained model
model = joblib.load('stock_price_prediction_model.pkl')

# Make the prediction
predictions = model.predict(X_input_scaled)

# Output the prediction result
print("Predicted Stock Price:", predictions)


# In[245]:


print(X_input.shape)  # This should print (1, 113) (1 row, 113 features)


# In[246]:


import numpy as np

# Example input for 19 stocks, each with 6 features (Open, Close, High, Low, Volume, Adj Close)
X_input = np.array([[
    150.2, 149.5, 151.0, 148.8, 150.0, 5000000,   # AAPL: Open, Close, High, Low, Volume, Adj Close
    2800.5, 2795.3, 2810.0, 2785.0, 2802.3, 1000000, # GOOG: Open, Close, High, Low, Volume, Adj Close
    300.5, 298.3, 302.0, 297.5, 299.7, 1200000,     # MSFT: Open, Close, High, Low, Volume, Adj Close
    134.5, 133.0, 136.0, 132.5, 134.2, 900000,      # TSLA: Open, Close, High, Low, Volume, Adj Close
    210.3, 208.5, 212.0, 207.0, 210.0, 800000,      # AMZN: Open, Close, High, Low, Volume, Adj Close
    140.5, 139.2, 141.0, 138.0, 139.5, 700000,      # FB: Open, Close, High, Low, Volume, Adj Close
    180.2, 179.8, 182.0, 178.0, 180.3, 600000,      # NVDA: Open, Close, High, Low, Volume, Adj Close
    250.3, 248.5, 255.0, 247.0, 249.7, 500000,      # INTC: Open, Close, High, Low, Volume, Adj Close
    50.5, 49.9, 52.0, 48.5, 51.0, 400000,           # DIS: Open, Close, High, Low, Volume, Adj Close
    300.8, 298.2, 305.0, 295.0, 300.0, 300000,      # PYPL: Open, Close, High, Low, Volume, Adj Close
    220.4, 218.0, 222.0, 217.0, 220.1, 350000,      # SNAP: Open, Close, High, Low, Volume, Adj Close
    180.2, 179.6, 181.5, 178.5, 179.8, 600000,      # AMD: Open, Close, High, Low, Volume, Adj Close
    130.7, 128.5, 132.0, 127.0, 130.1, 450000,      # VZ: Open, Close, High, Low, Volume, Adj Close
    50.2, 48.5, 51.5, 47.0, 49.2, 500000,           # BA: Open, Close, High, Low, Volume, Adj Close
    120.3, 119.0, 122.0, 118.0, 119.5, 550000,      # IBM: Open, Close, High, Low, Volume, Adj Close
    210.1, 208.0, 212.0, 206.5, 209.0, 700000,      # GS: Open, Close, High, Low, Volume, Adj Close
    240.0, 239.5, 243.0, 238.0, 240.2, 800000,      # JNJ: Open, Close, High, Low, Volume, Adj Close
    400.0, 399.5, 405.0, 398.0, 400.1, 900000,      # PFE: Open, Close, High, Low, Volume, Adj Close
    300.4, 298.9, 303.0, 297.0, 299.5, 850000       # PG: Open, Close, High, Low, Volume, Adj Close
]])

# Check the shape of the input
print(X_input.shape)  # Should print (1, 113)

# If needed, use your trained scaler to transform this input
import joblib
scaler = joblib.load('scaler.pkl')  # Load the previously saved scaler
X_input_scaled = scaler.transform(X_input)

# Load your trained model
model = joblib.load('stock_price_prediction_model.pkl')

# Make the prediction
predictions = model.predict(X_input_scaled)

# Output the prediction result
print("Predicted Stock Price:", predictions)


# In[ ]:




