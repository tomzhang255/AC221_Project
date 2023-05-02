# putpose: filter to keep tweets that contain any of the mental health related hashtags

import pandas as pd


if __name__ == '__main__':
    df = pd.read_csv('data/all_tweets_topic_lda.csv')
    hashtags = ['#mentalhealth', '#mentalhealthawareness', '#mentalhealthmatters', '#selfcare',
                '#selflove', '#anxiety', '#depression', '#ptsd', '#bipolar', '#ocd', '#eatingdisorders']
    df = df[df['text'].str.contains('|'.join(hashtags))]
    df.to_csv('data/hashtag_filtered_tweets.csv', index=False)
