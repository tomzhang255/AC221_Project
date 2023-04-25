# some of the tweets are geo-tagged - get some summary statistics on those
# group by country, city, month-year - then get counts

import pandas as pd


all = pd.read_csv('all_tweets.csv', dtype=str)
geo = pd.read_csv('all_geo.csv', dtype=str)
print('read both csv done')

# join both
df = pd.merge(all, geo, how='inner', left_on='geo', right_on='id')
print('join both done')

# clean up
df = df.loc[:, ['country', 'name', 'created_at', 'id_x']]
df.rename({'name': 'region', 'id_x': 'id'}, axis=1, inplace=True)
print('clean up done')

# get year and month
df['year'] = pd.DatetimeIndex(df['created_at']).year
df['month'] = pd.DatetimeIndex(df['created_at']).month
print('get time done')

# groupby - count
res = df.groupby(['country', 'region', 'year', 'month'])[['id']].count().reset_index()
res.rename({'id': 'nazar_tweet_count'}, axis=1, inplace=True)
print('groupby done')

res.to_csv('stat_geo.csv', index=False)
print('csv saved')
