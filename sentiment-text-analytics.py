# Import necessary modules
import keys
from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential


#  LaTonya Lewis
# How to Use:
# API Integration:
# Developers can integrate Text Analytics into their applications by making API calls to the service endpoints.
#
# Authentication:
# Use API keys to authenticate and access the service endpoint.
#
# Send Text Data:
# Submit text data for analysis and receive structured results.


# Function to authenticate and create a TextAnalyticsClient
def authenticate_client():
    # Initialize AzureKeyCredential with the API key
    text_analytics_credential = AzureKeyCredential(keys.LANG_KEY)

    # Create a TextAnalyticsClient using the endpoint and credential
    text_analytics_client = TextAnalyticsClient(
        endpoint=keys.LANG_ENDPOINT,
        credential=text_analytics_credential
    )
    return text_analytics_client


# Function to perform sentiment analysis on a list of documents
def sentiment_analysis(client):
    # List of text documents to analyze
    documents = [
        "I am ecstatic about the promotion I received at work; it's a dream come true!",
        "The concert was absolutely fantastic, and I had an amazing time.",
        "I enjoyed the meal at the new restaurant; it was quite good overall.",
        "The new software update is decent and has some useful features.",
        "I was completely dissatisfied with the customer service I received.",
        "The flight was delayed for hours, and the experience was terrible.",
        "The book was not as interesting as I had hoped; it was a bit dull.",
        "The hotel room was okay, but the cleanliness could have been better.",
        "I'm indifferent about the new policy changes at work; they don't affect me much."
        # "The weather today is average, with nothing remarkable about it."
        # "The football game was dope."
        # "The Team USA Gymnastics is totally off the chain."
        # "The Team USA Gymnastics nailed it."
        # "The Team USA Gymnastics was slamming."
        # "The Team USA - Gymnastics is dope ."
        # "Team USA Gymnastics was bad"

    ]

    # Analyze sentiment of each document
    results = client.analyze_sentiment(documents)

    # Print the results for each document
    for idx, result in enumerate(results):
        print(f"{idx + 1:>2}. {''.join([sentence.text for sentence in result.sentences])}")
        print(result.sentiment)
        print(f"- positive: {result.confidence_scores.positive:.2f}")
        print(f"- negative: {result.confidence_scores.negative:.2f}")
        print(f"- neutral: {result.confidence_scores.neutral:.2f}")


# Main function to authenticate the client and perform sentiment analysis
def main():
    client = authenticate_client()
    sentiment_analysis(client)


# Entry point of the script
if __name__ == '__main__':
    main()
