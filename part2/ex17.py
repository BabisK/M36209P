import tweepy
import pandas

consumer_key = 'uBfFf7mL6pbkyHgxdtQB7WjQ2'
consumer_secret = '1zDX3bFFXV1UyENP4RGzbn3d6dTmDJuOj23Fuuhzo5pW3AGWyR'

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
#auth.set_access_token(OAUTH_TOKEN, OAUTH_TOKEN_SECRET)
api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)

dataframe = pandas.read_csv('./2download/gold/train/100_topics_100_tweets.sentence-three-point.subtask-A.train.gold.txt', delim_whitespace=True, header=None, names=['id', 'sentiment'])

s = []
for i in dataframe.id:
    try:
        s.append(api.get_status(i).text)
    except tweepy.error.TweepError:
        s.append(None)
        print(i)

dataframe['tweet'] = s
dataframe.to_csv('./test.csv')
print('Saved test.csv')

dataframe = pandas.read_csv('./2download/gold/dev/100_topics_100_tweets.sentence-three-point.subtask-A.dev.gold.txt', delim_whitespace=True, header=None, names=['id', 'sentiment'])

s = []
for i in dataframe.id:
    try:
        s.append(api.get_status(i).text)
    except tweepy.error.TweepError:
        s.append(None)
        print(i)

dataframe['tweet'] = s
dataframe.to_csv('./dev.csv')
print('Saved dev.csv')

dataframe = pandas.read_csv('./2download/gold/devtest/100_topics_100_tweets.sentence-three-point.subtask-A.devtest.gold.txt', delim_whitespace=True, header=None, names=['id', 'sentiment'])

s = []
for i in dataframe.id:
    try:
        s.append(api.get_status(i).text)
    except tweepy.error.TweepError:
        s.append(None)
        print(i)

dataframe['tweet'] = s
dataframe.to_csv('./devtest.csv')
print('Saved devtest.csv')