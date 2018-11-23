try:
    import json
except ImportError:
    import simplejson as json
import twitter
import time

ACCESS_KEY = '910479255993274368-s7ugR4zq1DlECaqfbhyWyBqUVQVuDUm'
ACCESS_SECRET = 'htN53B6O8iiTW8kJ3Lb22J7yTPgdX3mpOqalKcvI3Za3T'
CONSUMER_KEY = 'dfcQU9kdwc2gJBKribuVs6Fx2'
CONSUMER_SECRET = 'z8iBmWI08BdLcA8FB7HtoRpN6PbN7FaFNUM8hxHz3oln7XauCr'

def main():
    api = twitter.Api(consumer_key=CONSUMER_KEY,
                      consumer_secret=CONSUMER_SECRET,
                      access_token_key=ACCESS_KEY,
                      access_token_secret=ACCESS_SECRET,
                      sleep_on_rate_limit=True)

    tweets= []
    i = 1

    with open('data/twitter_gender_data_pruned.json') as data:
        data = json.load(data)
        for key in data:
            try:
                tmp = api.GetUserTimeline(user_id=key['id'], count=200)
                print(tmp.Text)
                tweets.append(tmp)
                i = i + 1
                if i > 10:
                    break
            except Exception as e:
                with open('logs/log.txt', 'a') as log:
                        log.write(str(e))
                        log.write(str(key['id']))
                        log.write("\n")
        
        with open('data/twitter_tweets_pruned.json', 'w') as tweetsFile:
            tmp = []
            for user in tweets:
                tmp.append([tweet.__dict__ for tweet in user])
            json.dump(tmp, tweetsFile)

if __name__ == '__main__':
    main()

