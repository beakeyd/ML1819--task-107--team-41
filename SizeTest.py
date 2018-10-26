try:
    import json
except ImportError:
    import simplejson as json

def main():
    with open('data/twitter_gender_data.json') as data:
        data = json.load(data)
        print(str(len(data)))

if __name__ == '__main__':
    main()