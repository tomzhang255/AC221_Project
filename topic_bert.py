import pandas as pd
import numpy as np
import re
from datetime import datetime
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from joblib import Parallel, delayed

# some logging helper functions
def log(msg):
    print(f'{datetime.now()} | {msg}')

def time_elapsed(t1, t2, msg):
    print(f'Time elapsed: {t2 - t1} ({msg})\n')

def now():
    return datetime.now()

# a decorator
def log_execution_time(func):
    def wrapper(*args, **kwargs):
        t1 = now()
        log(f'Executing {func.__name__}...')
        result = func(*args, **kwargs)
        t2 = now()
        log('Done')
        time_elapsed(t1, t2, f'Finished {func.__name__}')
        return result
    return wrapper

def preprocess_text(text):
    # remove URLs, mentions, and special characters
    text = re.sub(r"(?:\@|http\S+|www\S+|[^a-z0-9\s])", "", text.lower())
    return text

def embed_text(text, model):
    return model.encode(text)


if __name__ == '__main__':
    tt0 = now()

    data = pd.read_csv("data/all_tweets_sentiment.csv", nrows=300)

    # apply preprocessing to the text column
    data["processed_text"] = data["text"].apply(preprocess_text)

    # create and fit BERTopic model
    log("Creating BERTopic model...")
    model = BERTopic()

    # get SentenceTransformer model for parallel processing
    sentence_model = SentenceTransformer('distilbert-base-nli-mean-tokens')

    log("Generating embeddings in parallel...")
    num_cores = -1  # use all available CPU cores
    embeddings = Parallel(n_jobs=num_cores)(
        delayed(embed_text)(text, sentence_model) for text in data["processed_text"]
    )

    log("Fitting BERTopic model...")
    embeddings_array = np.asarray(embeddings)
    topics, _ = model.fit_transform(data["processed_text"], embeddings_array)
    log("Done fitting BERTopic model")

    # get actual topic strings
    topic_strings = []
    for topic_idx in topics:
        topic_words, probs = zip(*model.get_topic(topic_idx))
        if len(probs) > 0:
            topic_string = " ".join(topic_words[:5]) # concatenate the first five words of the topic
        else:
            topic_string = " ".join(topic_words)
        topic_strings.append(topic_string)

    # save the topic strings back into the data frame
    data["topic"] = topic_strings

    # save final data frame
    data.drop('processed_text', axis=1, inplace=True)
    data.to_csv('data/all_tweets_topic_bert.csv', index=False)

    tt1 = now()
    time_elapsed(tt0, tt1, 'overall')
