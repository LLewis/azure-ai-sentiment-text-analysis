import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string

# Ensure that the necessary NLTK data packages are downloaded
nltk.download('punkt')
nltk.download('wordnet')

# Sample lexicon with sentiment scores
# Positive words have positive scores, negative words have negative scores
lexicon = {
    'good': 2.0,
    'bad': -2.0,
    'happy': 3.0,
    'sad': -3.0,
    'great': 3.0,
    'terrible': -3.0,
    'excellent': 4.0,
    'poor': -2.0,
    'love': 3.0,
    'hate': -3.0,
    'dope': 4.0
}

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
    return lemmatized_tokens

# Custom sentiment scoring function
def custom_sentiment_score(text):
    # Normalize and tokenize the text
    tokens = normalize_text(text)
    # Initialize score variables
    positive_score = 0
    negative_score = 0

    # Calculate the sentiment scores
    for token in tokens:
        if token in lexicon:
            score = lexicon[token]
            if score > 0:
                positive_score += score
            elif score < 0:
                negative_score += score

    # Aggregate scores
    total_score = positive_score + negative_score
    sentiment = 'neutral'
    if total_score > 0:
        sentiment = 'positive'
    elif total_score < 0:
        sentiment = 'negative'

    # Return detailed scores and overall sentiment
    return {
        'positive': positive_score,
        'negative': negative_score,
        'total': total_score,
        'sentiment': sentiment
    }

# Example usage
# text = "Microsoft was founded by Bill Gates and Paul Allen. It is a great company with excellent products."
text = input('Enter review')
scores = custom_sentiment_score(text)
print(f"Text: {text}")
print(f"Sentiment: {scores['sentiment']}")
print(f"Scores: Positive = {scores['positive']}, Negative = {scores['negative']}, Total = {scores['total']}")
