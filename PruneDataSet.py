try:
    import json
except ImportError:
    import simplejson as json

def main():
    with open('data/twitter_gender_data.json') as data:
        data = json.load(data)                     
        femaleRCount = 0
        i = 0
        for i in range(0, len(data) - 1):
            if femaleRCount > 1696:
                break
            if data[i]['gender'] == 'F':
                del data[i]
                femaleRCount = femaleRCount + 1
            i = i + 1

        with open('data/twitter_gender_data2.json', 'w+') as output2:
            json.dump(data, output2)

if __name__ == '__main__':
    main()
