
# Stock-Prediction-Project.
Predicting stock price trends using user-generated content from Reddit and stock data.

# Stock Prediction Project
This project uses sentiment analysis of user-generated content to predict stock market trends.
# Feature
- Scrapes data from Reddit and Yahoo Finance.
- Preprocesses and cleans data for analysis.
- Trains a machine learning model to predict stock price trends.
# Setup Instructions
 1. Clone the Repository
Clone this repository to your local machine:

git clone https://github.com/YourUsername/Stock-Prediction-Project1.git
cd Stock-Prediction-Project1

 2. Install Dependencies
Ensure that you have Python 3.8 or later installed on your system. Then, install the required libraries using pip:

                 pip install -r requirements.txt
This will install all the necessary dependencies for running the project.
 3. Run the Project
Once you have cloned the repository and installed the dependencies, you can run the project script:
python stock_prediction_project.py
The script will:
1.Scrape data from Reddit and Yahoo Finance.
2.Preprocess and clean the data.
3.Train the machine learning model to predict stock trends.

Dependencies:
To run this project, you'll need the following Python libraries:

1.Python 3.8+
2.Pandas: For data manipulation and cleaning.
3.Scikit-learn: For machine learning and model training.
4.Matplotlib: For data visualization.
5.BeautifulSoup4: For web scraping (used for scraping Reddit posts).
6.PRAW: Python Reddit API Wrapper for accessing Reddit posts and comments.
7.Yahoo_fin: For scraping financial data from Yahoo Finance.


How the Script Works?
The stock_prediction_project.py script is designed to:
Scrape Data:

The script uses PRAW (Python Reddit API Wrapper) to scrape Reddit posts related to stock predictions or discussions. It gathers the post title, score, comments, and sentiment.
It also uses Yahoo_fin to fetch historical stock data such as open, high, low, close, adjusted close, and volume for the stock symbols (tickers) being discussed.

Data Preprocessing:

It processes and cleans the scraped data by removing any irrelevant or missing information. The data is then merged based on the stock ticker and date for analysis.
Training the Machine Learning Model:
The preprocessed data is used to train a machine learning model (e.g., Random Forest, Linear Regression) to predict stock price trends.
The model uses both historical stock prices and sentiment data from Reddit to make predictions.

Model Evaluation:

The model's performance is evaluated using accuracy metrics such as Mean Squared Error (MSE), R-squared, or other relevant metrics.

Example Output:
The script outputs the following:

1.The predicted stock prices based on the sentiment of Reddit posts and historical stock data.
2.Visualizations such as graphs of stock prices and sentiment scores over time.

License:
This project is licensed under the MIT License. See the LICENSE file for details.

Authors:
Madhav Paluru: Project creator and developer.

Notes:
Ensure you have access to Reddit and Yahoo Finance API credentials if required for scraping.
Modify the stock_prediction_project.py script to include your own stock tickers or subreddits if you want to scrape data for different stocks or discussions.
The accuracy of predictions may vary depending on the quality and quantity of the data scraped.



---

 3d19c7a0d9efe79c13f08ab87ebd7f4a269c11ee
