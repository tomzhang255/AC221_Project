# the goal of this script is to retrieve all tweets within a specific time period;
# using the academic research account yields a response of up to 500 tweets per request;
# of course, the complete result set far exceeds 500 tweets;
# twitter api allows us to retrieve the complete result set with pagination;
# essentially, each response contains a 'next_token' field which points to
# the next partition of the complete result set;
# we will be able to retrieve the complete result set by repeatedly sending
# api requests with an extra 'next_token' parameter;
# stop when a response no longer has a 'next_token' field;


import requests
import json
from datetime import datetime
from time import sleep
from math import inf
import os


with open('credentials/api_credentials.json') as f:
    credentials = json.load(f)
    bearer_token = credentials['bearer_token']

search_url = 'https://api.twitter.com/2/tweets/search/all'

query_params = {
    # only retrieve original tweets in English
    'query': 'a lang:en -is:retweet -is:reply -is:quote',
    # 500 result limit per page
    'max_results': 500,
    'tweet.fields': 'created_at',
    'expansions': 'geo.place_id',
    'place.fields': 'full_name,id,contained_within,country,country_code,geo,name,place_type',
    # May (mental health awareness month), pre-COVID
    'end_time': '2019-05-31T00:00:00Z',
    'start_time': '2019-05-01T00:00:00Z',
    'next_token': None
}


def bearer_oauth(r):
    """
    Method required by bearer token authentication.
    """

    r.headers['Authorization'] = f'Bearer {bearer_token}'
    r.headers['User-Agent'] = 'v2FullArchiveSearchPython'
    return r


# sends an api request
def connect_to_endpoint(search_url, params):
    response = requests.request(
        'GET', search_url, auth=bearer_oauth, params=params)
    if response.status_code != 200:
        if response.status_code in [429, 503]:  # too many requests
            print('too many requests, now sleeping 30 min...')
            sleep(1800)  # 30 min pause
            print('sleep done, retrying request...')
            response = requests.request(
                'GET', search_url, auth=bearer_oauth, params=params)
            print('got response...')
            if response.status_code != 200:
                print('bad response even after sleeping...')
                raise Exception(response.status_code, response.text)
        else:
            raise Exception(response.status_code, response.text)
    return response.json()


def write_one_page_of_response(next_token=None):
    # send request
    query_params['next_token'] = next_token
    json_response = connect_to_endpoint(search_url, query_params)
    json_string = json.dumps(json_response, indent=4, sort_keys=True)

    # write json
    now_str = datetime.now().strftime("%m-%d-%Y_%H:%M:%S")
    if not os.path.exists('res'):
        os.makedirs('res')
    with open(f'res/{now_str}.json', 'w') as f:
        f.write(json_string)

    # log
    next_token = json_response.get('meta').get('next_token')
    print(now_str, next_token)

    return next_token


# if we want to start collecting tweets from a specific next_token, simply modify next_token = None
# if we know how many iterations we want to repeat, specify that too
def collect_all(next_token=None, max_iter=inf):
    next_token = write_one_page_of_response(next_token)
    i = 1
    while next_token is not None and i < max_iter:
        sleep(6)  # rate limit (300 requests / 15 mins)
        next_token = write_one_page_of_response(next_token)
        i += 1


if __name__ == '__main__':
    # I can only pull 10 mil tweets per month
    # ~ 20,000 total requests can be made if each request returns 500 tweets
    # I can make 19000 requests a month to be conservative
    # It takes max 4 seconds per request
    # 76,000 sec ~ 21 hours

    print('May, 2019...')
    collect_all(next_token=None, max_iter=10_000)

    print('April, 2019...')
    query_params['start_time'] = '2019-04-01T00:00:00Z'
    query_params['end_time'] = '2019-04-30T00:00:00Z'
    collect_all(next_token=None, max_iter=10_000)
