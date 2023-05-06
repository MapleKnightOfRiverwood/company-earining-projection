from CompanyTextSentimentAnalyzer import TextSentimentAnalyser
import pandas as pd

# 1. Load the dictionary as dataframe
lm_word_list_df = pd.read_csv('Loughran-McDonald_MasterDictionary_1993-2021.csv')
# 2. Create a TextSentimentAnalyser object
text_analyzer = TextSentimentAnalyser(lm_word_list_df)
# 3. Pass in company ticker and year to calculate sentiment. If company does not have 10k
# that year it will return null
sentiment = text_analyzer.get_sentiment('KBAL', 2010)
print(sentiment)









