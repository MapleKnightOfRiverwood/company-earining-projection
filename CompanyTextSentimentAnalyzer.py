import glob
import os
import shutil
from string import punctuation
import nltk
import pandas as pd
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from sec_edgar_downloader import Downloader
import warnings
warnings.filterwarnings('ignore')

# Text Analytics

class TextSentimentAnalyser:
    def __init__(self, lm_word_list_df):
        self.lm_word_list_df = lm_word_list_df
        self.sentiment_dict = self.create_sentiment_dict(lm_word_list_df)

    # Scrapper function
    def fetch_10k(self, company, year):
        dl = Downloader()
        n = dl.get("10-K", company, after=f"{year}-01-01", before=f"{year}-12-31")
        if n == 0:
            print(f"No 10k found for {company} in {year}.")
            return 0
        directory_path = f".\\sec-edgar-filings\\{company}\\10-K"
        html_file_path = glob.glob(os.path.join(directory_path, '*', 'filing-details.html'))[0]
        # Read the contents of the HTML file into memory as a string
        with open(html_file_path, 'r', encoding='utf-8') as file:
            html_content = file.read()
        # Use beautiful soup to convert html to plain text
        soup = BeautifulSoup(html_content, 'html.parser')
        plain_text = soup.get_text()
        # Delete the html file
        shutil.rmtree(os.path.join('.\\sec-edgar-filings', company))
        return plain_text


    # Preprocess scrapped 10k document
    def preprocess_10k(self, annual_report):
        englishStopWords = set(stopwords.words('english'))
        transTable = str.maketrans('', '', punctuation)
        tokens = nltk.word_tokenize(annual_report)  # Tokenization
        tokens = [word.translate(transTable) for word in tokens]  # Delete punctuations
        tokens = [word for word in tokens if word.isalpha()]  # Delete numbers
        tokens = [word.lower() for word in tokens]  # Lower casing
        tokens = [word for word in tokens if word not in englishStopWords]  # Delete stop words
        tokens = [word for word in tokens if len(word) > 1]  # Delete single letters
        return ' '.join(tokens)  # Merge all elements in a list into one string separated by space


    # Calculate sentiment
    def sentiment_counts(self, preprocessed_text, sentiment_dict):
        sentiment_categories = {
            'Positive': 0,
            'Negative': 0,
            'Uncertainty': 0,
            'Litigious': 0,
            'Strong_Modal': 0,
            'Weak_Modal': 0,
            'Constraining': 0
        }
        word_list = nltk.word_tokenize(preprocessed_text)
        for word in word_list:
            word_data = sentiment_dict.get(word.upper())
            if word_data:
                for category in sentiment_categories.keys():
                    if word_data[category] > 0:
                        sentiment_categories[category] += 1
        # Return the sentiment with the highest score
        max_sentiment = max(sentiment_categories, key=sentiment_categories.get)
        return max_sentiment


    # We use dictionary to reduce the time complexity of sentiment_count to O(n) since lookup is O(1)
    def create_sentiment_dict(self, df):
        sentiment_dict = {}
        for _, row in df.iterrows():
            word = row['Word']
            sentiment_dict[word] = {
                'Positive': row['Positive'],
                'Negative': row['Negative'],
                'Uncertainty': row['Uncertainty'],
                'Litigious': row['Litigious'],
                'Strong_Modal': row['Strong_Modal'],
                'Weak_Modal': row['Weak_Modal'],
                'Constraining': row['Constraining']
            }
        return sentiment_dict


    # Wrapper function
    def get_sentiment(self, company, year):
        filling_unprocessed = self.fetch_10k(company, year)
        # Check if 10-K filing was found
        if filling_unprocessed == 0:
            return None
        filling_preprocessed = self.preprocess_10k(filling_unprocessed)
        sentiment = self.sentiment_counts(filling_preprocessed, self.sentiment_dict)
        return sentiment


# How to use the object
if __name__ == '__main__':
    # 1. Load the dictionary as dataframe
    lm_word_list_df = pd.read_csv('Loughran-McDonald_MasterDictionary_1993-2021.csv')
    # 2. Create a TextSentimentAnalyser object
    text_analyzer = TextSentimentAnalyser(lm_word_list_df)
    # 3. Pass in company ticker and year to calculate sentiment. If company does not have 10k
    # that year it will return null
    sentiment = text_analyzer.get_sentiment('AAPL', 2012)
    print(sentiment)










