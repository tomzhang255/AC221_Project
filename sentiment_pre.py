# Purpose: just predict sentiment using a pre-trained model
# Labels: 0 -> Negative; 1 -> Neutral; 2 -> Positive

import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor


def predict_sentiment(text):
    max_sequence_length = 512  # The maximum sequence length for RoBERTa-based models
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_sequence_length)
    outputs = model(**inputs)
    logits = outputs.logits
    sentiment = torch.argmax(logits, dim=1).item()
    return sentiment


if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
    model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")

    all_tweets_both = pd.read_csv('all_tweets_both.csv')
    all_tweets_both.dropna(subset=['text'], inplace=True)

    t1 = datetime.now()
    print('processing...')

    with ThreadPoolExecutor() as executor:
        all_tweets_both['sentiment'] = list(executor.map(predict_sentiment, all_tweets_both['text']))

    print('done')
    t2 = datetime.now()
    print(f'time elapsed: {t2 - t1}')

    all_tweets_both.to_csv('all_tweets_sentiment.csv', index=False)
