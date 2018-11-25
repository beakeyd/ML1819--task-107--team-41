import nltk
from nltk.corpus import stopwords 
from nltk.stem import PorterStemmer
from nltk.tokenize import RegexpTokenizer
from langid.langid import LanguageIdentifier, model
import re

try:
    import json
except ImportError:
    import simplejson as json
jsonFile="data/DavidsPruned.json"

 
    

def removeNonEnglish(data):
    identifier = LanguageIdentifier.from_modelstring(model, norm_probs=True)

    i=0
    listD=[]
    i=0
    j=0
    for d in data:
        
        tweets=[]
        for tweet in data[d]:  
                   
            lang=(identifier.classify(tweet))[0]
            
            if lang=="en":
                tweets.append(tweet)
        data[d]=tweets
            
           
            #print(len(listD))
    with open('data/tweets_lang_pruned.json', 'w+') as output:
        json.dump(data, output)
    print(i)
    print(j)

def test(data):
    identifier = LanguageIdentifier.from_modelstring(model, norm_probs=True)

    i=0
    listD=[]
    i=0
    j=0
    for d in data:
        
        
        for tweet in data[d]:  
                   
            lang=(identifier.classify(tweet))[0]
            j+=1
            if not lang=="en":
                i+=1
        
            
           
            #print(len(listD))
    
    print(i)
    print(j)            
    



def main():
    fileToStrip='data/twitter_tweets_pruned.json'
   
    
    #with open('data/twitter_tweets_pruned.json') as data:
     #   data=json.load(data)
      #  removeNonEnglish(data)
    with open('data/tweets_lang_pruned.json') as data:
        data=json.load(data)
        test(data)










if __name__ == '__main__':
    main()
