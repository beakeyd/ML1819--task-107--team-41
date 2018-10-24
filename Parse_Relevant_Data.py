try:
    import json
except ImportError:
    import simplejson as json
import twitter
import pprint

ACCESS_KEY = '910479255993274368-s7ugR4zq1DlECaqfbhyWyBqUVQVuDUm'
ACCESS_SECRET = 'htN53B6O8iiTW8kJ3Lb22J7yTPgdX3mpOqalKcvI3Za3T'
CONSUMER_KEY = 'dfcQU9kdwc2gJBKribuVs6Fx2'
CONSUMER_SECRET = 'z8iBmWI08BdLcA8FB7HtoRpN6PbN7FaFNUM8hxHz3oln7XauCr'

def main():
    api = twitter.Api(consumer_key=CONSUMER_KEY,
                      consumer_secret=CONSUMER_SECRET,
                      access_token_key=ACCESS_KEY,
                      access_token_secret=ACCESS_SECRET)
    
    allUsers = []
    newDict= {}    
    i = 0
    with open('data/gender.json') as f:
        data = json.load(f)
        copy = dict(data)        
        for key in data:
            try:
                
            except:
                del copy[key]

        with open('data/gender2.json', 'w') as f:
            json.dump(copy, f)

    with open('data/twitter_gender_data.json', 'w') as f:
        json.dump(allUsers, f)

if __name__ == '__main__':
    main()