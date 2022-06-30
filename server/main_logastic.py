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
from xgboost import XGBClassifier

team = pd.read_pickle("horses.pkl")


def make_horses_to_2d(data, days):
   
    test_ar = []
    all_in_one = []
    print(len(days))
    starts_num = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]

   

    for d in days:
        
        for start in starts_num:
          
            df_res = data.query("day == @d and start_num == @start")
            test_ar = []
            
            for index, row in df_res.iterrows():     
                test_ar.extend([            #row['track'] ,
                                            row['horse_run_time'],  
                                            row['probable'], 
                                            row['amount'], 
                                            horse_gender(row['gender']),
                                            #race_type(row['race_type']),
                                            #row['age']
                                            #hash_shoes(row['front_shoes']),
                                            #row['rest_days'],
                                            row['win_money'],
                                            #start,
                                            #row['driver_l'],
                                            #row['distance'],
                                            #row['horse_starts'],
                                            #row['city_l'],
                                            #row['last_run'],
                                            row['probable_last'], 
                                            row['driver_starts'],                           
                                            row['horse_win_prob'],
                                            row['d_w_pr'],
                                            row['h_w_S'],
                                            row['c_w_pr']
                                            

                                            
                                            ])
            
            
            if len(test_ar) != 0:
                #print(len(test_ar))
                #same_len = 189 - len(test_ar)
                for i in range(len(test_ar), 176):
                    #if test_ar[i] != float:
                        test_ar.append(0.0)

                #print("what" , len(test_ar))
                all_in_one.append(test_ar)
          
    #print(all_in_one)

    col_len = []
    for i in range(176):
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
            win_prob = float(drivers_last_starts['d_w_pr'])
           

        except:
            starts = 0.0
            win_prob = 0.0   

        for index, row in today_data.iterrows():
            if row['driver'] == d:
                today_data.at[index, "driver_starts"] = starts
                today_data.at[index, 'd_w_pr'] = win_prob
    
    return today_data

def set_coach_history(past_data, today_data, drivers):

    for d in drivers:
        try:
            drivers_race = past_data.query("coach == @d")
            drivers_last_starts = drivers_race.iloc[-1:]
            #starts = float(drivers_last_starts['driver_starts'])
            d_win_prob = float(drivers_last_starts['c_w_pr'])
            print(d_win_prob)

        except:
            
            d_win_prob = 0.0   

        for index, row in today_data.iterrows():
            if row['driver'] == d:
                #today_data.at[index, "driver_starts"] = starts
                today_data.at[index, 'c_w_pr'] = d_win_prob
    
    today_data.fillna(0.0)

    return today_data


def set_horse_history(past_data, today_data, drivers):

    for d in drivers:
        try:
            horse_race = past_data.query("name == @d")
            horse_last_starts = horse_race.iloc[-1:]
            try:
                proba = float(horse_last_starts['probable'])
            except:
                proba = 0.0
            
            try:
                pos = float(horse_last_starts['position'])
            except:
                pos = 0.0

            try:
                last_win = float(horse_last_starts['horse_wins'])
            except:
                last_win = 0.0
           
            try:
                d_win_prob = float(horse_last_starts['c_w_pr'])
            except:
                d_win_prob = 0.0

        except:
            pos = 0.0
            proba = 0.0
            last_win = 0.0
            d_win_prob = 0.0   

        for index, row in today_data.iterrows():
            if row['name'] == d:
                today_data.at[index, "last_proba"] = proba
                today_data.at[index, "last_run"] = pos
                today_data.at[index, "h_w_S"] = last_win
                today_data.at[index, "c_w_pr"] = d_win_prob
    
    return today_data

def get_collection(city):
    place = city
    
    today_race = load_today_race.make_horses(place)
    
    df = today_race['horses']

    
    df['gender_new'] = list(map(horse_gender , list(df['gender'])))
    df['front_new'] = list(map(hash_shoes, list(df['front_shoes'])))
    df['race_new'] = list(map(race_type, list(df['race_type'])))
    
    drives = get_array(list(df['driver']))
    horses = get_array(list(df['name']))
    coaches = get_array(list(df['coach']))
    ### COLLETCT TO DAY HORSES####
    

    df2 = set_drivers_history(team, df, drives)
    df3 = set_horse_history(team, df2, horses)
    #df3 = set_coach_history(team, df2, coaches)

    df3['win_money'] = df3['win_money'] / 100

    print(df)
    print(df3)

    starts = 12
    for i in range(starts):
         print(df3.query("start_num == @i").sort_values(by=['horse_win_prob'], ascending=False))


    

    for i, row in df3.iterrows():
        try:        
            df3.at[i, 'probable_last'] = 1 / row['last_proba']
        except ZeroDivisionError:
            df3.at[i, 'probable_last'] = 0.0000

    today_pred_horses = make_horses_to_2d(df3, today_race['days'])
    print(today_pred_horses)

    start_nu = 6

    print(df3.query("start_num == @start_nu"))
  
    
    clf_boost = joblib.load("make_horse_stats/gradientBoost_" + place + ".pkl")
    clf = joblib.load("make_horse_stats/logasticRegression_" + place + ".pkl")      

    xgbc = XGBClassifier()

    xgbc.load_model("make_horse_stats/XGBC_" + place + ".txt" )                   
    xgbc._le = LabelEncoder().fit([1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0 ,11.0,12.0,13.0,14.0,15.0])
   
    
    print("boost", clf_boost.predict(today_pred_horses))
    print("logas", clf.predict(today_pred_horses))
    print("xgbc ", xgbc.predict(today_pred_horses))

    clf_arra =  clf.predict(today_pred_horses)
    boost_arr =  clf_boost.predict(today_pred_horses)
    xg_arr =  xgbc.predict(today_pred_horses)

    boost_res = clf_boost.predict_proba(today_pred_horses)
    clf_res = clf.predict_proba(today_pred_horses)
    xgbc_res = xgbc.predict_proba(today_pred_horses)
  
    for i in range(len(boost_res)):
        #print("start " + str(i +1), clf_res[i].tolist())
        #print("start " + str(i +1), np.sum(clf_res[i]))
        
        print("start " + str(i +1) + " xgbcC: " , xgbc_res[i].argsort()[::-1][:3] + 1)

        floats2 = clf_res[i].argsort()[::-1][:3] + 1
        print("start " + str(i +1) + " logas: ", floats2)
        
        floats3 = boost_res[i].argsort()[::-1][:3] + 1
        print("start " + str(i +1) + " boost: ", floats3)
    
    
    
    
    
    return json.dumps({ "boost": boost_arr.tolist(), "logas": clf_arra.tolist(), "xgbc": xg_arr.tolist() })



#get_collection("Teivo")
