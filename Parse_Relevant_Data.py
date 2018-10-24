try:
    import json
except ImportError:
    import simplejson as json
import twitter
import time

ACCESS_KEY = 'xxx'
ACCESS_SECRET = 'xxx'
CONSUMER_KEY = 'xxx'
CONSUMER_SECRET = 'xxx'

def main():
    api = twitter.Api(consumer_key=CONSUMER_KEY,
                      consumer_secret=CONSUMER_SECRET,
                      access_token_key=ACCESS_KEY,
                      access_token_secret=ACCESS_SECRET)

    allUsers = []
    newDict= {}
    i = 0
    with open('data/gender.json') as f:
        if i % 1000 == 0:
            print('waiting for 15 mins.')
            time.slep(900)
            print('watining finished.')
        data = json.load(f)
        copy = dict(data)        
        for key in data:
            try:
                i = i + 1      
                tmp = api.GetUser(key)
                newDict['created_at'] = tmp.created_at
                newDict['description'] = tmp.description
                newDict['favourites_count'] = tmp.favourites_count
                newDict['id'] = tmp.id
                newDict['lang'] = tmp.lang
                newDict['listed_count'] = tmp.listed_count
                newDict['location'] = tmp.location
                newDict['name'] = tmp.name
                newDict['profile_background_color'] = tmp.profile_background_color
                newDict['profile_link_color'] = tmp.profile_link_color
                newDict['profile_sidebar_border_color'] = tmp.profile_sidebar_border_color
                newDict['profile_sidebar_fill_color'] = tmp.profile_sidebar_fill_color
                newDict['profile_text_color'] = tmp.profile_text_color
                newDict['screen_name'] = tmp.screen_name
                newDict['tweet'] = tmp.status.text
                newDict['gender'] = data[key]
                allUsers.append(newDict)
                print(i)
            except Exception as e:
                del copy[key]
                print(e)

        with open('data/gender2.json', 'w') as f:
            json.dump(copy, f)

    with open('data/twitter_gender_data.json', 'w') as f:
        json.dump(allUsers, f)

if __name__ == '__main__':
    main()
