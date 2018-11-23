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
                      access_token_secret=ACCESS_SECRET)

    allUsers = []
    newDict= {}
    i = 1

    with open('data/gender.json') as gender:
            data = json.load(gender)
            copy = dict(data)
            for key in data:
                try:
                    if i % 800 == 0:
                        with open('logs/log.txt', 'a') as log:
                            log.write('waiting for 15 mins.')
                            log.write("\n")
                        time.sleep(900)
                        with open('logs/log.txt', 'a') as log:
                            log.write('waiting finished.')
                            log.write("\n")
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
                    newDict = {}
                    if(i % 10 == 0):
                        with open('logs/log.txt', 'a') as log:
                            log.write(str(i))
                            log.write("\n")
                except Exception as e:
                    del copy[key]
                    with open('logs/log.txt', 'a') as log:
                        log.write(str(e))
                        log.write("\n")

            with open('logs/log.txt', 'a') as log:
                log.write("Finishing scraping, writing output")
                log.write("\n")

            with open('data/gender2.json', 'w') as output1:
                json.dump(copy, output1)

            with open('data/newer_twitter_gender_data.json', 'w') as output2:
                json.dump(allUsers, output2)
            
            with open('logs/log.txt', 'a') as log:
                log.write("Finished writing output, script complete.")
                log.write("\n")

if __name__ == '__main__':
    main()

