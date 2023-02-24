import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Import the dataset
df = pd.read_csv('qa_dataset.csv')

# Preprocess the data
df['question'] = df['question'].apply(lambda x: ' '.join([word.lower() for word in x.split() if word.isalpha()]))
df['answer'] = df['answer'].apply(lambda x: ' '.join([word.lower() for word in x.split() if word.isalpha()]))

# Tokenize the data
nltk.download('punkt')
df['question_tokens'] = df['question'].apply(nltk.word_tokenize)
df['answer_tokens'] = df['answer'].apply(nltk.word_tokenize)

# Perform sentiment analysis
sia = SentimentIntensityAnalyzer()
df['question_sentiment'] = df['question'].apply(lambda x: sia.polarity_scores(x)['compound'])
df['answer_sentiment'] = df['answer'].apply(lambda x: sia.polarity_scores(x)['compound'])

# Store the analysis in another file
df.to_csv('sentiment_data.csv', index=False)
