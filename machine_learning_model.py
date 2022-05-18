
from re import L
from matplotlib.pyplot import get
import pandas as pd
import json
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
import load_today_race
import statsmodels.api as sm
import joblib


df_win = pd.read_pickle("horses_win3.pkl")
df_no_wins = pd.read_pickle("horses_n_race3.pkl")
f = open('toto_starts_2009.json')
team = json.load(f)


def search_y_city(city):

    race_winner = []
    race_second = []
    race_horse = pd.DataFrame()
    days = []

    try:
        index = 0
        for i in range(len(team)):
            if team[i]['place'] == city:
                race_winner.append(int(team[i]['results'][0]))
                #race_second.append(int(team[i]['results'][1]))

                day = team[i]['day']
                days.append(day)
                race_typ = team[i]['race_type']
                race_distance = team[i]['race_distance']
            
                
                horses = team[i]['horses']
                for k in range(len(horses)):
                    horses[k]['day'] = day
                    horses[k]['race_type'] = race_typ
                    horses[k]['distance'] = race_distance
                    race_horse = race_horse.append(horses[k], ignore_index=True)
                        
    except:
        print("er")

    return {"horses": race_horse,"winners": race_winner, "days": list(dict.fromkeys(days)) }

def winner_2d_array(winner_array):
    winer_2d = []
    
    for i in range(len(winner_array)):
        outcome_array = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        outcome_array.insert(int(winner_array[i]), 1)

        winer_2d.append(outcome_array)

    return winer_2d


def make_horses_to_2d(data, data2, days):
   
    test_ar = []
    all_in_one = []
    print(len(days))
    starts_num = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]

    home_town = get_array(list(data2['home_town'])) 
    le_home = LabelEncoder()
    le_home.fit(home_town)
    data['home_l'] = le_home.fit_transform(data['home_town'])
    
    driver_names = get_array(list(data2['driver']))
    le_horse = LabelEncoder()
    le_horse.fit(driver_names)
    data['driver_l'] = le_horse.fit_transform(data['driver'])
    
    coach_names = get_array(list(data2['coach']))
    le_coach = LabelEncoder()
    le_coach.fit(coach_names)
    data['coach_l'] = le_coach.fit_transform(data['coach'])

    horse_names = get_array(list(data2['name']))
    le_horse = LabelEncoder()
    le_horse.fit(horse_names)
    data['horse_l'] = le_coach.fit_transform(data['name'])

    print(data)
    
    
    for d in days:
        
        for start in starts_num:
          
            df_res = data.query("day == @d and start_num == @start")
            test_ar = []
            
            for index, row in df_res.iterrows():     
                test_ar.extend([row['track'] ,row['age'],  row['probable'], row['amount'], 
                                            horse_gender(row['gender']),
                                            race_type(row['race_type']),
                                            hash_shoes(row['front_shoes']),
                                            row['win_money'], row['home_l'] , row['driver_l'],row['coach_l'],
                                         
                                            

                                           
                                            
                                            ])
            
            
            if len(test_ar) != 0:
                #print(len(test_ar))
                #same_len = 189 - len(test_ar)
                for i in range(len(test_ar), 178):
                    #if test_ar[i] != float:
                        test_ar.append(0.0)

                #print("what" , len(test_ar))
                all_in_one.append(test_ar)
          
    #print(all_in_one)

    col_len = []
    for i in range(178):
        col_len.append(i)
    
    df = pd.DataFrame(all_in_one, columns=col_len)
    
    return df

def get_array(data):
    unique = []
    for number in data:
        if number in unique:
            continue
        else:
            unique.append(number)
    return unique


def horse_gender(gender):
    if gender == "TAMMA":
        return 1
    elif gender == "RUUNA":
        return 2
    else:
        return 3

def race_type(race):
    if race == "CAR_START":
        return 5
    else:
        return 6

def hash_shoes(shoes):
    if shoes == "HAS_SHOES":
        return 1
    else:
        return 0


if __name__ == "__main__":

    """
    place = 'Forssa'
    
    all_in_city = search_y_city(place)
    w_2d = np.array(all_in_city['winners'])

    df = make_horses_to_2d(all_in_city['horses'], all_in_city['days']) #.reshape(-1,1)
    df['winner'] = w_2d

    print(df)
    col_len = []
    for i in range(178):
        col_len.append(i)

    X = df[col_len].to_numpy(dtype="float")
    y = df['winner'].to_numpy(dtype="int")

    X_train,X_test,y_train,y_test = train_test_split(X, np.array(y), test_size = 0.20 ,random_state = 20)

    clf = LogisticRegression(solver='saga',max_iter=1200).fit(X_train, y_train)
    print("LogasticRegression", clf.score(X_test, y_test))

    clf_boost = GradientBoostingClassifier(learning_rate=0.2, max_depth=1, random_state=0).fit(X_train, y_train)
    print("GradientBoost", clf_boost.score(X_test, y_test))
    """

    
    ### LOAD TODAY RACES AND PREDICT #### 
    place = 'Kuopio'
    clf_boost = joblib.load("gradientBoost_new.pkl")
    clf = joblib.load("logasticRegression_new.pkl")                   
    
    all_indf = pd.read_pickle("toto_all.pkl")

    today_race = load_today_race.make_horses(place)
    df_today = make_horses_to_2d(today_race['horses'], all_indf, today_race['days'])
    #print(df_today)
    print("boost", clf_boost.predict(df_today))
    print("logas", clf.predict(df_today))
    boost_res = clf_boost.predict_proba(df_today)
    clf_res = clf.predict_proba(df_today)

  
    for i in range(len(boost_res)):

        
        floats = boost_res[i].argpartition(-3)[-3:] 
        print( "start " +  str(i) + " boost: ", floats)
        floats2 = clf_res[i].argpartition(-3)[-3:] 
        print("start " + str(i) + " logas: ", floats2)
       
    
  


