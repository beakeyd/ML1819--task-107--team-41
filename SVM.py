import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

def main():
    with open('data/twitter_gender_data.json') as data:
        data = json.load(data)

        # slice data
        created_at = data[:, 'created_at']
        favourites_count = data[:, 'favourites_count']
        color = data[:, 'color']
        listed_count = data[:, 'listed_count']
        description = data[:, 'description']
        tweet = data[:, 'tweet']
        name = data[:, 'name']
        screen_name = data[:, 'screen_name']
        gender = data[:, 'gender']

        # create models, plot and then get accuracy of models
        created_at_acc = created_at(created_at, gender)
        favourites_acc = favourites_count(favourites_count, gender)
        color_acc = color(color, gender)
        listed_acc = listed_count(listed_count, gender)
        description_acc = description(description, gender)
        tweet_acc = tweet(tweet, gender)
        name_acc = name(name, gender)
        screen_name_acc = screen_name(screen_name, gender)

        plotAccuracy(created_at_acc, favourites_acc,
                     color_acc, listed_acc, description_acc,
                     tweet_acc, name_acc, screen_name_acc, 'Accuracy')



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

def validate(actY, predY):
    acc = reduce(lambda m, n: m+n, list(isPredictionCorrect, actY, predY))
    return (acc / len(actY))
        

''' 
    Models that ARE NOT doing text classification
'''

def created_at(created_at):
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

def color(pf_background_color, pf_link_color, 
          pf_sidebar_border_color, pf_sidebar_fill_color, 
          pf_text_color):
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

def favourites_count(favourites_count, gender):
    # create Model
    X = np.array(favourites_count)
    y = np.array(gender)
    clf = svm.SVC(kernel='linear', C = 1.0)
    clf.fit(X, y)

    # make predicitions
    predY = [clf.predict(x) for x in X]
    #plot data, get and return accuracy of model
    plotFeatureData(X, y, predY, 'Favourites_Count')
    return validate(X, y, predY)

def listed_count(listed_count):
    # create Model
    X = np.array(listed_count)
    y = np.array(gender)
    clf = svm.SVC(kernel='linear', C = 1.0)
    clf.fit(X, y)

    # make predicitions
    predY = [clf.predict(x) for x in X]
    #plot data, get and return accuracy of model
    plotFeatureData(X, y, predY, 'Listed_Count')
    return validate(X, y, predY)

''' 
    Models that ARE doing text classification
'''

def description():
def tweet():
def name():
def screen_name():

#def combinedFeatures():

if __name__ == '__main__':
    main()