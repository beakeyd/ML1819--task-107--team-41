try:
    import json
except ImportError:
    import simplejson as json
import twitter
import time
from langid.langid import LanguageIdentifier, model
ACCESS_KEY = '1052159436108779520-umLnDy6pA9mfDiU2Sr52GDdFUFgR6l'
ACCESS_SECRET = 'QgiAitaCOqRrI7cq8b1AYiNiPC1Ol6TeLnLGBwxu6swn3'
CONSUMER_KEY = 'Vn2vtBzY8uZirkuqIMJQHLpbD'
CONSUMER_SECRET = 'BcluieZLGLqgwT8Xn2J6nbuYmjrOwNHeWdfsn9HDc8f2OKlQgK'
api = twitter.Api(consumer_key=CONSUMER_KEY,
                      consumer_secret=CONSUMER_SECRET,
                      access_token_key=ACCESS_KEY,
                      access_token_secret=ACCESS_SECRET,
                      sleep_on_rate_limit=True)


def gatherTweets(data):
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

def gatherHashtag(data):
    data = json.load(data)
    identifier = LanguageIdentifier.from_modelstring(model, norm_probs=True)
    usrDict = {}
    numbHashtag={}
    numHashtag=0
    i=0
    j=0
    for key in data:
        

       
        try:
            usrDict[key['id']]=[]
            tmp = api.GetUserTimeline(user_id=key['id'], count=200)
            for t in tmp:
                hashtags=t.hashtags
                
                numHashtag+=len(hashtags)
                
                for h in hashtags:
                    
                    hashtag=h.text
                    hashtag=hashtag.encode('ascii', 'ignore').decode("utf-8")
                    lang=(identifier.classify(hashtag))[0]
                    if lang=="en":
                        usrDict[key['id']].append(hashtag)
                  
            meanHashTag=numHashtag/200
            numHashtag=0
            
            numbHashtag[key['id']]=meanHashTag
        except Exception as e:
            print(e)
            with open('logs/log.txt', 'a') as log:
                    log.write(str(e))
                    log.write(str(key['id']))
                    log.write("\n")
    
    with open('data/twitter_Hashtag.json', 'w+') as tweetsFile:
        json.dump(usrDict, tweetsFile)
    with open('data/numb_Hashtag.json', 'w+') as tweetsFile:
            json.dump(numbHashtag, tweetsFile)
    

def main():
    
    i = 1

   # with open('data/twitter_gender_data_pruned.json') as data:
        #gatherTweets(data)
    with open('data/twitter_gender_data_pruned.json') as data:
        gatherHashtag(data)

if __name__ == '__main__':
    main()

