import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import string

# Ensure that the necessary NLTK data packages are downloaded
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('vader_lexicon')

# Text normalization function
def normalize_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Tokenize the text
    tokens = word_tokenize(text)
    # Lemmatize the tokens
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    # Join tokens back into a single string
    normalized_text = ' '.join(lemmatized_tokens)
    return normalized_text

# Sentiment analysis function
def analyze_sentiment(text):
    # Normalize the text
    normalized_text = normalize_text(text)
    # Initialize VADER sentiment intensity analyzer
    sid = SentimentIntensityAnalyzer()
    # Get the sentiment scores
    sentiment_scores = sid.polarity_scores(normalized_text)
    # Determine the sentiment
    compound_score = sentiment_scores['compound']
    print("compound score: ", compound_score)  # Lewis
    if compound_score >= 0.05:
        sentiment = 'positive'
    elif compound_score <= -0.05:
        sentiment = 'negative'
    else:
        sentiment = 'neutral'


    # Print the sentiment scores dictionary  - Lewis
    print("Sentiment Scores:")
    for key, value in sentiment_scores.items():
        print(f"{key}: {value:.2f}")
    return sentiment, sentiment_scores

# Example usage
text = input("Enter review: ")
# text = "Microsoft was founded by Bill Gates and Paul Allen on April 4, 1975."
sentiment, scores = analyze_sentiment(text)
print(f"Text: {text}")
print(f"Sentiment: {sentiment}")
print(f"Sentiment Scores: {scores}")
