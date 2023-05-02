# Purpose: clean up aggregated csv; make sure it can be read.
# And add a month indicator too.


import csv
import pandas as pd


def clean_csv(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8', errors='ignore') as infile, \
            open(output_file, 'w', encoding='utf-8', newline='') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)

        for row in reader:
            try:
                writer.writerow(row)
            except csv.Error:
                print(f"Skipped row: {row}")


if __name__ == '__main__':
    # clean csv
    input_files = ['data/all_tweets.csv', 'data/all_geo.csv']
    output_files = ['data/cleaned_all_tweets.csv', 'data/cleaned_all_geo.csv']

    for i in range(len(input_files)):
        clean_csv(input_files[i], output_files[i])

    all_tweets = pd.read_csv('data/cleaned_all_tweets.csv')
    all_geo = pd.read_csv('data/cleaned_all_geo.csv')

    print('all loaded')

    # add month to tweets
    all_tweets['month'] = all_tweets['created_at'].str.slice(6, 7)
    # add month to geo
    merged = pd.merge(all_tweets, all_geo, left_on='geo',
                      right_on='id', suffixes=('_tweets', ''))
    all_geo = merged.loc[:, all_geo.columns.tolist() + ['month']]

    print('all mutated')

    # save final
    all_tweets.to_csv('data/all_tweets_both.csv', index=False)
    all_geo.to_csv('data/all_geo_both.csv', index=False)
