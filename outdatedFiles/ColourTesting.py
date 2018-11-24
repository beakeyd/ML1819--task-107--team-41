try:
    import json
except ImportError:
    import simplejson as json
import numpy as np

#def color

def main():
    Twitter_Color_Classes = ["FF691F", "FAB81E", "7FDBB6", "19CF86", "91D2FA", "1B95E0", "ABB8C2", "E81C4F", "F58EA8", "981CEB"]
    with open('data/twitter_gender_data.json') as data:
       
        data = json.load(data)
        profile_background_color = [d["profile_background_color"] for d in data]
        profile_sidebar_border_color = [d["profile_sidebar_border_color"] for d in data]
        gender = np.where(np.array([d["gender"] for d in data]) == 'M', 0, 1)
        profile_link_color = [d["profile_link_color"] for d in data]
        #profile_sidebar_fill_color = [d["profile_sidebar_fill_color"] for d in data]
        profile_text_color = [d["profile_text_color"] for d in data]
        #profile_sidebar_border_color = [d["profile_sidebar_border_color"] for d in data]
        #Ctrain, Ctest, Gtrain, Gtest = train_test_split(description, gender, test_size=0.3, random_state=42)
        

        allUsers = []
        newDict= {}
        i = 1
       
            
              
      
      

            
        


        twitterColour=0
        userColour=0
        i=0
        l={}
        for color in profile_text_color:
          
            r=color[0]+"0"
            g=color[2]+"0"
            b=color[4]+"0"
           # color=r+g+b
           
            
            if color not in l:
                l[color]=0
            else:
                l[color]+=1
            if(color in Twitter_Color_Classes):
                twitterColour=twitterColour+1
            else:
                userColour=userColour+1
            i=i+1
        unique=0
        amount=0
        for  k,v in l.items():
            if v >500:
                print(k)
                amount+=v
                #print(v)
                unique+=1

        print(twitterColour)
        print(userColour)
        print(unique)
        print(len(l))
        


if __name__ == '__main__':
    main()
