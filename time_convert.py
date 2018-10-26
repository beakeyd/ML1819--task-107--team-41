try:
    import json
except ImportError:
    import simplejson as json


def main():
    with open('data/twitter_gender_data.json', 'r') as data:
        data = json.load(data)

        for item in data:
            if item[]

    with open('data/twitter_gender_data2.json', 'w') as file:
        json.dump(json_data, file, indent=2)