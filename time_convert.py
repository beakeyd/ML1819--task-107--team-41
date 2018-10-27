try:
    import json
except ImportError:
    import simplejson as json

from datetime import datetime as dt
import time

def main():
    print("Running")
    with open('data/twitter_gender_data.json', 'r') as data:
        data = json.load(data)
        print("Starting conversion")
        for item in data:
            date = dt.strptime(item['created_at'], "%a %b %d %H:%M:%S +0000 %Y")
            item['created_at'] = time.mktime(date.timetuple())

    print("Writing output")
    with open('data/twitter_gender_data2.json', 'w') as output:
        json.dump(data, output)

if __name__ == '__main__':
    main()