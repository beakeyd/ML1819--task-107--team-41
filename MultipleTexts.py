try:
    import json
except ImportError:
    import simplejson as json
import twitter
import time

ACCESS_KEY = '1052159436108779520-umLnDy6pA9mfDiU2Sr52GDdFUFgR6l'
ACCESS_SECRET = 'QgiAitaCOqRrI7cq8b1AYiNiPC1Ol6TeLnLGBwxu6swn3'
CONSUMER_KEY = 'Vn2vtBzY8uZirkuqIMJQHLpbD'
CONSUMER_SECRET = 'BcluieZLGLqgwT8Xn2J6nbuYmjrOwNHeWdfsn9HDc8f2OKlQgK'

def main():
    api = twitter.Api(consumer_key=CONSUMER_KEY,
                      consumer_secret=CONSUMER_SECRET,
                      access_token_key=ACCESS_KEY,
                      access_token_secret=ACCESS_SECRET,
                      sleep_on_rate_limit=True)
    i = 1

    with open('data/twitter_gender_data_pruned.json') as data:
        data = json.load(data)
        usrDict = {}
        for key in data:
            usrDict[key['id']]=[]
            try:
                tmp = api.GetUserTimeline(user_id=key['id'], count=200)
              
                for tweet in tmp:
                    
                    text=tweet.text
                  
                    usrDict[key['id']].append(text)
            except Exception as e:
                with open('logs/log.txt', 'a') as log:
                        log.write(str(e))
                        log.write(str(key['id']))
                        log.write("\n")
        
        with open('data/twitter_tweets_pruned.json', 'w') as tweetsFile:
            json.dump(usrDict, tweetsFile)

if __name__ == '__main__':
    main()

