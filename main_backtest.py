import json
import pandas as pd
import machine_all
from sklearn.preprocessing import LabelEncoder
import joblib


f = open('toto_sort_2019.json')
team = json.load(f)
print(team[len(team)-1])

f = open('toto_for_backtest.json')
backtest_data = json.load(f)



def search_y_city(city):

    race_winner = []
    race_second = []
    race_horse = pd.DataFrame()
    days = []
    print(city)


    try:
        index = 0
        for i in range(len(backtest_data)):
            # NOT PLACE IN USE
            if backtest_data[i]['place'] == city:
                race_winner.append(int(backtest_data[i]['results'][0]))   
                win = int(backtest_data[i]['results'][0])

                #race_second.append(int(team[i]['results'][1]))

                day = backtest_data[i]['day']
                days.append(day)
                race_typ = backtest_data[i]['race_type']
                race_distance = backtest_data[i]['race_distance']
            
                
                horses = backtest_data[i]['horses']
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
                test_ar.extend([row['track'] ,row['run_time'],  row['prob_small'], row['amount'], 
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

    place = "Jokimaa"
    res_back = search_y_city(place)

    df = res_back['horses']
    print(df)

    names = get_array(list(df['name']))
    drivers = get_array(list(df['driver']))



    for i , row in df.iterrows():
        df.at[i, 'horse_starts'] = row['starts']

        try:
            df.at[i, 'horse_win_prob'] = row['postion1'] / row['starts']
        
        except ZeroDivisionError as e:
            df.at[i, 'horse_win_prob'] = 0.0123

    df['prob_small'] = df['probable'] / 100
    print(df)

    df_history_data = machine_all.collect_all_data(place)
    print(df_history_data)

    df2 = set_drivers_history(df_history_data, df, drivers)
    print(df2)

    today_pred_horses = make_horses_to_2d(df2, df_history_data, res_back['days'])
    print(today_pred_horses)

    clf_boost = joblib.load("gradientBoost_" + place + ".pkl")
    clf = joblib.load("logasticRegression_" + place + ".pkl")                   
    
   
    #print(df_today)
    print("winne", res_back['winners'])

    print("boost", clf_boost.predict(today_pred_horses))
    print("logas", clf.predict(today_pred_horses))
    boost_res = clf_boost.predict_proba(today_pred_horses)
    clf_res = clf.predict_proba(today_pred_horses)

    

    for i in range(len(boost_res)):
        #print("start " + str(i +1), clf_res[i].tolist())
        #print("start " + str(i +1), np.sum(clf_res[i]))
        
        floats2 = clf_res[i].argsort()[::-1][:3] + 1
        print("start " + str(i +1) + " logas: ", floats2)
        
        floats3 = boost_res[i].argsort()[::-1][:3] + 1
        print("start " + str(i +1) + " boost: ", floats3)
