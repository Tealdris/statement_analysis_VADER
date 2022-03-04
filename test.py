import nltk
import sklearn
import pandas as pd

nltk.download("vader_lexicon")
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.metrics import accuracy_score, classification_report


# Change data format; in this way there only two columns
def format_data(data):
    last_col = str(data.columns[-1])
    first_col = str(data.columns[0])

    # columns set in here
    data.rename(columns={last_col: 'tweet_text', first_col: 'polarity'}, inplace=True)

    # Change 0, 2, 4 to negative, neutral and positive
    labels = {0: 'negative', 2: 'neutral', 4: 'positive'}
    data['polarity'] = data['polarity'].map(labels)

    # Get only the two columns
    return data[['tweet_text', 'polarity']]


# change the evaluation about the message
def format_output(output_dict):
    polarity = "neutral"

    if output_dict['compound'] >= 0.05:
        polarity = "positive"

    elif output_dict['compound'] <= -0.05:
        polarity = "negative"

    return polarity


def predict_sentiment(text):
    output_dict = sent_analyzer.polarity_scores(text)
    return format_output(output_dict)


sent_analyzer = SentimentIntensityAnalyzer()

data_url = "https://raw.githubusercontent.com/keitazoumana/VADER_sentiment-Analysis/main/data/testdata.manual.2009.06.14.csv"
sentiment_data = pd.read_csv(data_url)
print(sentiment_data)

# Apply the transformation
data = format_data(sentiment_data)
print(data.head(3))

# Run the predictions
data["vader_prediction"] = data["tweet_text"].apply(predict_sentiment)
# Show 5 random rows of the data
print(data.sample(5))

accuracy = accuracy_score(data['polarity'], data['vader_prediction'])
print("Accuracy: {}\n".format(accuracy))
# Show the classification report
print(classification_report(data['polarity'], data['vader_prediction']))
