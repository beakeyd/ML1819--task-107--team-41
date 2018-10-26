try:
    import json
except ImportError:
    import simplejson as json


def main():
    with open('data/twitter_gender_data.json', 'r') as data:
        data = json.load(data)

        for item in data:
<<<<<<< HEAD
            if item[]

    with open('data/twitter_gender_data2.json', 'w') as file:
        json.dump(json_data, file, indent=2)
=======
            date = dt.strptime(item['created_at'], "%a %b %d %H:%M:%S +0000 %Y")
            item['created_at'] = time.mktime(date.timetuple())

    with open('data/twitter_gender_data2.json', 'w') as output:
        json.dump(json_data, output)
>>>>>>> 2d21cc3b10706a7e2b42271b28236d9b022fd2b3
