#!/usr/bin/python
from langid.langid import LanguageIdentifier, model
from py_translator import Translator
'''
This function contains code relevant to any transformations applied to text
'''
import json


def pruneGenderImbalance(data, pruneFileDest):
                 
    femaleRCount = 0
    i = 0
    for i in range(0, len(data) - 1):
        
        if femaleRCount > 1685:
            break
        if data[i]['gender'] == 'F':
            del data[i]
            femaleRCount = femaleRCount + 1
        i = i + 1

    with open(pruneFileDest, 'w+') as output2:
            json.dump(data, output2)
def test(data):
    identifier = LanguageIdentifier.from_modelstring(model, norm_probs=True)
    translator = Translator()
    i=0
    listD=[]
    
    for d in data:
       
        tweet=d["tweet"]
        tweetEncode=tweet.encode('utf-8')
        
        lang=(identifier.classify(tweet))[0]
        if not lang =="en":
            i+=1
        #print(len(listD))

    print(i)

def main():
    fileToPrune='data/twitter_gender_data_original.json'
    #with open(fileToPrune) as data:
     #   data = json.load(data)
      #  pruneGenderImbalance(data, 'data/DavidsPruned.json')
    
    with open('data/DavidsPruned.json') as data:
        data=json.load(data)
        test(data)
    
        

if __name__ == '__main__':
    main()
