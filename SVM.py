try:
    import json
except ImportError:
    import simplejson as json
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split
from functools import reduce

def main():
    with open('data/twitter_gender_data.json') as data:
        data = json.load(data)

        # slice data
        created_at = [d["created_at"] for d in data]
        favourites_count = [d["favourites_count"] for d in data]
        color = [d["profile_background_color"] for d in data]
        listed_count = [d["listed_count"] for d in data]
        description = [d["description"] for d in data]
        tweet = [d["tweet"] for d in data]
        name = [d["name"] for d in data]
        screen_name = [d["screen_name"] for d in data]
        gender = [d["gender"] for d in data]



        # create models, plot and then get accuracy of models
        #created_at_acc = created_at_model(created_at, gender)
        favourites_acc = favourites_count_model(favourites_count, gender)
        print('accuracy:')
        print(str(favourites_acc))
        #color_acc = color_model(color, gender)
        #listed_acc = listed_count_model(listed_count, gender)
        #description_acc = description(description, gender)
        #tweet_acc = tweet(tweet, gender)
        #name_acc = name(name, gender)
        #screen_name_acc = screen_name(screen_name, gender)

        #scikit_test()

        #plotAccuracy(created_at_acc, favourites_acc,
        #             color_acc, listed_acc, description_acc,
        #             tweet_acc, name_acc, screen_name_acc, 'Accuracy')

def scikit_test():
    X = [[0, 0], [1, 1]]
    y = [0, 1]
    clf = svm.SVC(gamma='scale')
    clf.fit(X, y)

def plotAccuracy(created_at_acc, favourites_acc,
                 color_acc, listed_acc, description_acc,
                 tweet_acc, name_acc, screen_name_acc, graph_name):
    

    X = ['created_at', 'favourites',
         'color', 'listed', 'description',
         'tweet', 'name', 'screen_name']
    y = [created_at_acc, favourites_acc,
         color_acc, listed_acc, description_acc,
         tweet_acc, name_acc, screen_name_acc]

    fig, ax = plt.bar(figsize=(12,2))
    ax.bar(X, y, label='Accuracy')

    ax.set_xlabel('Feature')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy of Features chart')
    graph = 'plots/' + graph_name + '.png'   
    fig.savefig(graph)


def plotFeatureData(X, actY, predY, graph_name):
    fig, ax = plt.subplots(figsize=(12,2))
    ax.scatter(X, actY, label='Data')
    ax.scatter(X, predY, label='Prediction')

    ax.set_xlabel('Test Feature')
    ax.set_ylabel('Gender')
    ax.set_title('Feature Plot')
    graph = 'plots/' + graph_name + '.png'    
    fig.savefig(graph)

def isPredictionCorrect(y, pred):
  if y == pred:
    return 1
  else:
    return 0

def getAccuracy(actY, predY):
    acc = reduce(lambda m, n: m+n, list(map(isPredictionCorrect, actY, predY)))
    return (acc / len(actY))

''' 
    Models that ARE NOT doing text classification
'''

def created_at_model(created_at, gender):
    # create Model
    X = np.array(created_at)
    y = np.array(gender)
    clf = svm.SVC(kernel='linear', C = 1.0)
    clf.fit(X, y)

    # make predicitions
    predY = [clf.predict(x) for x in X]
    #plot data, get and return accuracy of model
    plotFeatureData(X, y, predY, 'Created_At')
    return validate(X, y, predY)

def color_model(color, gender):
    # profile_background_color
    # profile_link_color
    # profile_sidebar_border_color
    # profile_sidebar_fill_color
    # profile_text_color

    # create Model
    X = np.array()
    y = np.array(gender)
    clf = svm.SVC(kernel='linear', C = 1.0)
    clf.fit(X, y)

    # make predicitions
    predY = [clf.predict(x) for x in X]
    #plot data, get and return accuracy of model
    plotFeatureData(X, y, predY, 'Favourites_Count')
    return validate(X, y, predY)

def favourites_count_model(favourites_count, gender):
    # create Model
    X = np.array(favourites_count).reshape(-1,1)
    y = np.where(np.array(gender) == 'M', 0, 1)
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.9)

    clf = svm.SVC(kernel='linear', C = 1.0)

    print(Xtrain)
    print(ytrain)

    clf.fit(Xtrain, ytrain)

    print('here!')

    # make predicitions
    predY = clf.predict(Xtest.reshape(-1,1))
    #plot data, get and return accuracy of model
    plotFeatureData(Xtest, ytest, predY, 'Favourites_Count')
    print("now")
    return getAccuracy(ytest, predY)

def listed_count_model(listed_count):
    # create Model
    X = np.array(listed_count)
    y = np.array(gender)
    clf = svm.SVC(kernel='linear', C = 1.0)
    clf.fit(X, y)

    # make predicitions
    predY = [clf.predict(x) for x in X]
    # plot data, get and return accuracy of model
    plotFeatureData(X, y, predY, 'Listed_Count')
    return validate(X, y, predY)

''' 
    Models that ARE doing text classification
'''

#def description():
#def tweet():
#def name():
#def screen_name():

#def combinedFeatures():

if __name__ == '__main__':
    main()