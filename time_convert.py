try:
    import json
except ImportError:
    import simplejson as json


def main():
    with open('data/twitter_gender_data.json', 'r') as data:
        data = json.load(data)

        for item in data:
            date = dt.strptime(item['created_at'], "%a %b %d %H:%M:%S +0000 %Y")
            item['created_at'] = time.mktime(date.timetuple())

    with open('data/twitter_gender_data2.json', 'w') as output:
        json.dump(json_data, output)