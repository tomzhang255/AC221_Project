"""
Purpose: aggregate all collected tweets into one single csv file.
We will handle regular tweets and geo-tagged tweets separately.

Here's an example of a geo-tagged tweet:

"data": [
    {
        "created_at": "2022-08-09T23:25:56.000Z",
        "geo": {
            "place_id": "97bcdfca1a2dca59"
        },
        "id": "1557146187450441730",
        "text": "T\u00e1 repreendido \ud83e\uddff\ud83c\udf36\ufe0f"
    },
    ...
]

However, its actual location fields are contained elsewhere (right after the data field):

"includes": {
    "places": [
        {
            "country": "Brazil",
            "country_code": "BR",
            "full_name": "Rio de Janeiro, Brazil",
            "geo": {
                "bbox": [
                    -43.795449,
                    -23.08302,
                    -43.0877068,
                    -22.7398234
                ],
                "properties": {},
                "type": "Feature"
            },
            "id": "97bcdfca1a2dca59",
            "name": "Rio de Janeiro",
            "place_type": "city"
        },
        ...
    ]
}

We will extract tweet data and geo data, then store them in separate dataframes;
each subsequent file will simply append to the relevant dataframes.
"""


import os
import json
import pandas as pd
import numpy as np
from datetime import datetime


if __name__ == '__main__':
    t1 = datetime.now()

    all_tweets = pd.DataFrame()
    all_geo = pd.DataFrame()

    file_names = os.listdir('res')
    i = 0

    for name in file_names:
        print(i)

        with open(f'res/{name}') as f:
            content = json.load(f)

            # get tweets data
            tweets = pd.DataFrame(content.get('data'))

            # clean up geo column in tweets df
            if 'geo' in tweets.columns:
                tweets['geo'] = tweets['geo'].map(lambda x: x.get('place_id')
                                                  if (type(x) != float) else np.nan)

            # get geo data
            if content.get('includes') is not None and content.get('includes').get('places') is not None:
                geo = pd.DataFrame(content.get('includes').get('places'))

            # append curr df to all df
            all_tweets = pd.concat([all_tweets, tweets])
            all_geo = pd.concat([all_geo, geo])

        i += 1

    # examine results
    print(all_tweets.head())
    print('==========')
    print(all_tweets.info())
    print('==========')
    print(all_tweets.shape)
    print('==========')
    print(all_geo.head())
    print('==========')
    print(all_geo.info())
    print('==========')
    print(all_geo.shape)
    print('==========')

    # make data/ folder if not exists
    if not os.path.exists('data'):
        os.makedirs('data')

    # save results
    all_tweets.to_csv('data/all_tweets.csv', index=False)
    all_geo.to_csv('data/all_geo.csv', index=False)

    t2 = datetime.now()
    print('Time')
    print(t1, t2)
    print(t2 - t1)
