import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Load data
data = pd.read_csv("sentiment_data.csv")

# Initialize sentiment analyzer
sid = SentimentIntensityAnalyzer()

# Calculate sentiment scores for each message
data['sentiment_score'] = data['message'].apply(lambda x: sid.polarity_scores(x)['compound'])

# Group messages by user
grouped_data = data.groupby('user_id')

# Calculate average sentiment score for each user
user_sentiment = grouped_data['sentiment_score'].mean()

# Identify users with consistently negative sentiment scores
negative_users = user_sentiment[user_sentiment < -0.5]

# Provide personalized insights to negative users
for user in negative_users.index:
    negative_messages = data.loc[(data['user_id'] == user) & (data['sentiment_score'] < -0.5), 'message']
    if len(negative_messages) > 0:
        print(f"User {user} has consistently negative sentiment in the following messages:")
        print(negative_messages)
        # Provide recommendations or resources for managing negative emotions
        # For example, you could suggest mindfulness exercises or provide links to mental health resources

# Identify users with a high percentage of positive messages
positive_users = user_sentiment[user_sentiment['positive'] > 0.5]

# Provide personalized insights to positive users
for user in positive_users.index:
    positive_messages = data.loc[(data['user_id'] == user) & (data['sentiment'] == 'positive'), 'message']
    if len(positive_messages) > 0:
        print(f"User {user} has a high percentage of positive messages:")
        print(positive_messages)
        # Provide recommendations or resources for maintaining positive emotions
        # For example, you could suggest gratitude journaling or provide links to mental health resources for maintaining positive mental health

