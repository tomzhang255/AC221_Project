# Purpose: apply topic modeling to generate a set number of topic for each piece of tweet
# Model: LDA

import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from gensim import corpora
from gensim.models import LdaModel, LdaMulticore
from multiprocessing import Pool, cpu_count
from datetime import datetime
import os


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

# a helper used within apply_preprocessing()
def preprocess_text(text):
    text = text.lower()
    # remove URLs, mentions, and special characters
    text = re.sub(r"(?:\@|http\S+|www\S+|[^a-z0-9\s])", "", text)
    # tokenize text
    words = nltk.word_tokenize(text)
    # remove stopwords
    stop_words = set(stopwords.words("english"))
    words = [word for word in words if word not in stop_words]
    # lemmatize words
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    return words

# a helper to be used by parallelize_dataframe_processing()
def apply_preprocessing(df):
    df["processed_text"] = df["text"].apply(preprocess_text)
    return df

@log_execution_time
def parallelize_dataframe_processing(df, func, num_partitions, num_workers):
    df_split = np.array_split(df, num_partitions)
    pool = Pool(num_workers)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df

# helper for assign_top_n_topics_to_documents_parallel()
def assign_top_n_topics_to_document(args):
    document, lda_model, top_n = args
    topic_probs = lda_model.get_document_topics(document)
    top_n_topics = sorted(topic_probs, key=lambda x: x[1], reverse=True)[:top_n]
    top_n_topics = [lda_model.show_topic(topic, topn=top_n) for topic, _ in top_n_topics]
    
    topic_words = []
    for topic in top_n_topics:
        words = " ".join([word for word, _ in topic])
        topic_words.append(words)
    
    topic_words += ['N/A'] * (top_n - len(topic_words))
    return topic_words

@log_execution_time
def assign_top_n_topics_to_documents_parallel(lda_model, corpus, top_n):
    args = [(document, lda_model, top_n) for document in corpus]
    with Pool(cpu_count()) as pool:
        top_n_topic_assignments = pool.map(assign_top_n_topics_to_document, args)
    return [assignment for assignment in top_n_topic_assignments]


if __name__ == '__main__':
    tt0 = now()

    data = pd.read_csv("all_tweets_sentiment.csv")

    # apply preprocessing to the text column in parallel
    num_partitions = 10
    num_workers = cpu_count()
    data = parallelize_dataframe_processing(data, apply_preprocessing, num_partitions, num_workers)

    # create a dictionary and corpus for LDA
    dictionary = corpora.Dictionary(data["processed_text"])
    corpus = [dictionary.doc2bow(text) for text in data["processed_text"]]

    # train the LDA model (parallel)
    num_topics = 50
    num_workers = cpu_count()
    t1 = now()
    log('training model...')
    lda_model = LdaMulticore(corpus, num_topics=num_topics, id2word=dictionary, workers=num_workers, random_state=42)
    t2 = now()
    log('done')
    time_elapsed(t1, t2, 'Finished training model')

    # save and load model
    if not os.path.exists('lda'):
        os.makedirs('lda')
    lda_model.save("lda/lda_model.gensim")
    loaded_lda_model = LdaModel.load("lda/lda_model.gensim")

    # get top n most probable topics to the documents (AKA tweets)
    top_n = 5
    top_n_topic_assignments = assign_top_n_topics_to_documents_parallel(lda_model, corpus, top_n)

    # save the topics back into the data frame
    top_n_topic_cols = [f"topic_{i + 1}" for i in range(top_n)]
    top_n_topic_probs_df = pd.DataFrame(top_n_topic_assignments, columns=top_n_topic_cols)
    data = pd.concat([data, top_n_topic_probs_df], axis=1)

    # save final data frame
    data.drop('processed_text', axis=1, inplace=True)
    data.to_csv('data/all_tweets_topic_lda.csv', index=False)

    tt1 = now()
    time_elapsed(tt0, tt1, 'overall')
