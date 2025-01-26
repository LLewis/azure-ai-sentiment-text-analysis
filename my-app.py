# my-app.py
import keys
from flask import Flask, render_template, request
from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential

app = Flask(__name__)

def authenticate_client():
    text_analytics_credential = AzureKeyCredential(keys.LANG_KEY)
    text_analytics_client = TextAnalyticsClient(
        endpoint=keys.LANG_ENDPOINT,
        credential=text_analytics_credential
    )
    return text_analytics_client


client = authenticate_client()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    text = request.form['text']
    response = client.analyze_sentiment(documents=[text])[0]

    if not response.is_error:
        sentiment = {
            "positive": response.confidence_scores.positive,
            "negative": response.confidence_scores.negative,
            "neutral": response.confidence_scores.neutral
        }
    else:
        sentiment = {"error": response.error}

    return render_template('result.html', sentiment=sentiment, text=text)


if __name__ == '__main__':
    app.run(debug=True)

