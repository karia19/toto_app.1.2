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
import machine_all

f = open('toto_sort_2019.json')
team = json.load(f)


def search_y_city(city):

    race_winner = []
    race_second = []
    race_horse = pd.DataFrame()
    days = []
    print(city)


    try:
        index = 0
        for i in range(len(team)):
            # NOT PLACE IN USE
            if team[i]['place'] == city:
                race_winner.append(int(team[i]['results'][0]))   
                win = int(team[i]['results'][0])

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
                    horses[k]['race_city'] = city

                    if horses[k]['track'] == win:
                        horses[k]['winner'] = 1.0
                    else:
                        horses[k]['winner'] = 0.0

                    race_horse = race_horse.append(horses[k], ignore_index=True)
                        
    except:
        print("er")

    #print(race_horse.sort_values(['day']))

    return {"horses": race_horse,"winners": race_winner, "days": list(dict.fromkeys(days)) }


def make_horses_to_2d(data, df_history, days):
   
    test_ar = []
    all_in_one = []
    print(len(days))
    starts_num = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]

    home_town = get_array(list(df_history['home_town'])) 
    le_home = LabelEncoder()
    le_home.fit(home_town)
    data['home_l'] = le_home.fit_transform(data['home_town'])
    
    driver_names = get_array(list(df_history['driver']))
    le_horse = LabelEncoder()
    le_horse.fit(driver_names)
    data['driver_l'] = le_horse.fit_transform(data['driver'])
    
    coach_names = get_array(list(df_history['coach']))
    le_coach = LabelEncoder()
    le_coach.fit(coach_names)
    data['coach_l'] = le_coach.fit_transform(data['coach'])

    horse_names = get_array(list(df_history['name']))
    le_horse = LabelEncoder()
    le_horse.fit(horse_names)
    data['horse_l'] = le_horse.fit_transform(data['name'])

    
    """
    race_city = get_array(list(data['race_city']))
    le_city = LabelEncoder()
    le_city.fit(race_city)
    data['city_l'] = le_coach.fit_transform(data['race_city'])
    """

    """
    print(data)
    file = open('toto_all.pkl', 'wb')
    pickle.dump(data, file)
    file.close()
    """

    for d in days:
        
        for start in starts_num:
          
            df_res = data.query("day == @d and start_num == @start")
            test_ar = []
            
            for index, row in df_res.iterrows():     
                test_ar.extend([row['track'] ,row['age'],  row['prob_small'], row['amount'], 
                                            horse_gender(row['gender']),
                                            #race_type(row['race_type']),
                                            #hash_shoes(row['front_shoes']),
                                            row['win_money'],
                                            row['home_l'] , row['driver_l'],row['coach_l'],
                                            row['horse_starts'], row['driver_starts'],                           
                                            row['horse_win_prob']

                                           
                                            
                                            ])
            
            
            if len(test_ar) != 0:
                #print(len(test_ar))
                #same_len = 189 - len(test_ar)
                for i in range(len(test_ar), 192):
                    #if test_ar[i] != float:
                        test_ar.append(0.0)

                #print("what" , len(test_ar))
                all_in_one.append(test_ar)
          
    #print(all_in_one)

    col_len = []
    for i in range(192):
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


def get_array(data):
    unique = []
    for number in data:
        if number in unique:
            continue
        else:
            unique.append(number)
    return unique

def set_drivers_history(past_data, today_data, drivers):

    for d in drivers:
        try:
            drivers_race = past_data.query("driver == @d")
            drivers_last_starts = drivers_race.iloc[-1:]
            starts = float(drivers_last_starts['driver_starts'])
           

        except:
            starts = 0.0   

        for index, row in today_data.iterrows():
            if row['driver'] == d:
                today_data.at[index, "driver_starts"] = starts
    
    return today_data


if __name__ == "__main__":

    place = 'Kouvola'
    
    today_race = load_today_race.make_horses(place)
    
    df = today_race['horses']
    print(df)

    names = get_array(list(df['name']))
    drivers = get_array(list(df['driver']))

    for i , row in df.iterrows():
        df.at[i, 'horse_starts'] = row['starts']

        try:
            df.at[i, 'horse_win_prob'] = row['postion1'] / row['starts']
        
        except ZeroDivisionError as e:
            df.at[i, 'horse_win_prob'] = 0.5000

    df['prob_small'] = df['probable'] / 100
    print(df)

    df_history_data = machine_all.collect_all_data(place)
    print(df_history_data)

    df2 = set_drivers_history(df_history_data, df, drivers)
    
    for i in range(4,8):
        print(df2.query("start_num == @i"))

    today_pred_horses = make_horses_to_2d(df2, df_history_data, today_race['days'])
    print(today_pred_horses)

    clf_boost = joblib.load("gradientBoost_" + place + ".pkl")
    clf = joblib.load("logasticRegression_" + place + ".pkl")                   
    
   
    #print(df_today)
    print("boost", clf_boost.predict(today_pred_horses))
    print("logas", clf.predict(today_pred_horses))
    boost_res = clf_boost.predict_proba(today_pred_horses)
    clf_res = clf.predict_proba(today_pred_horses)

  
    for i in range(len(boost_res)):
        print("start " + str(i +1), clf_res[i].tolist())
        print("start " + str(i +1), np.sum(clf_res[i]))

        """
        biggest_number = max(boost_res[i])
        print(boost_res[i].index(biggest_number))
        """
        """
        floats = boost_res[i].argpartition(-3)[-3:] 
        print( "start " +  str(i) + " boost: ", floats)
        floats2 = clf_res[i].argpartition(-3)[-3:] 
        print("start " + str(i) + " logas: ", floats2)
        """