#!/usr/bin/python
import json
from langid.langid import LanguageIdentifier, model


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
def genderPreProcessing(fileName):
    i=0
    with open(fileName) as data:
        data =json.load(data)
        for d in data:
            if d["gender"] == "M":
                d["gender"]=1
            else:
                d["gender"]=0
            
           
        with open(fileName, 'w+') as output2:
            json.dump(data, output2)
    
def test(data):
    identifier = LanguageIdentifier.from_modelstring(model, norm_probs=True)

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
    
    #with open('data/DavidsPruned.json') as data:
     #   data=json.load(data)
      #  test(data)
    genderPreProcessing("data/Davids.json")
    
        

if __name__ == '__main__':
    main()
