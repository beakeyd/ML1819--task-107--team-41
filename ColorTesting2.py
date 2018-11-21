try:
    import json
except ImportError:
    import simplejson as json
import numpy as np

def main():
    with open('data/new_twitter_gender_data.json') as data:
        
        data = json.load(data)
        profile_background_color = [d["profile_background_color"] for d in data]
        profile_sidebar_fill_color = [d["profile_sidebar_fill_color"] for d in data]
        gender = np.where(np.array([d["gender"] for d in data]) == 'M', 0, 1)
        #profile_link_color = [d["profile_link_color"] for d in data]
        #profile_sidebar_fill_color = [d["profile_sidebar_fill_color"] for d in data]
        profile_text_color = [d["profile_text_color"] for d in data]
        #profile_sidebar_border_color = [d["profile_sidebar_border_color"] for d in data]
        #Ctrain, Ctest, Gtrain, Gtest = train_test_split(description, gender, test_size=0.3, random_state=42)

        twitterColour=0
        userColour=0
        i=0
       
        for color in profile_text_color:
           
            
            if(color in Twitter_Color_Classes):
                twitterColour=twitterColour+1
            else:
                userColour=userColour+1
            i=i+1
        
        print(twitterColour)
        print("break")
        print(userColour)
        


if __name__ == '__main__':
    main()
