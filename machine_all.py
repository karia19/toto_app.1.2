
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
import pickle

#df_win = pd.read_pickle("horses_win3.pkl")
#df_no_wins = pd.read_pickle("horses_n_race3.pkl")
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
                win_money = int(team[i]['win_money'])

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
                        horses[k]['win_money'] = win_money
                    else:
                        horses[k]['winner'] = 0.0
                        horses[k]['win_money'] = 0.0


                    race_horse = race_horse.append(horses[k], ignore_index=True)
                        
    except:
        print("er")

    #print(race_horse.sort_values(['day']))

    return {"horses": race_horse,"winners": race_winner, "days": list(dict.fromkeys(days)) }



def make_horses_to_2d(data, days):
   
    test_ar = []
    all_in_one = []
    print(len(days))
    starts_num = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]

    home_town = get_array(list(data['home_town'])) 
    le_home = LabelEncoder()
    le_home.fit(home_town)
    data['home_l'] = le_home.fit_transform(data['home_town'])
    
    driver_names = get_array(list('driver'))
    le_horse = LabelEncoder()
    le_horse.fit(driver_names)
    data['driver_l'] = le_horse.fit_transform(data['driver'])
    
    coach_names = get_array(list(data['coach']))
    le_coach = LabelEncoder()
    le_coach.fit(coach_names)
    data['coach_l'] = le_coach.fit_transform(data['coach'])

    horse_names = get_array(list(data['name']))
    le_horse = LabelEncoder()
    le_horse.fit(horse_names)
    data['horse_l'] = le_horse.fit_transform(data['name'])

   
    race_city = get_array(list(data['race_city']))
    le_city = LabelEncoder()
    le_city.fit(race_city)
    data['city_l'] = le_coach.fit_transform(data['race_city'])
    
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
                                            row['horse_money'],
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


def collect_all_data(city):

    citys_arr = []
    winners = []
    days_arr = []

    
    all_city =  all_in_city = search_y_city(city)
    w_2d = np.array(all_in_city['winners'])
    d_2d = np.array(all_in_city['days'])
    #i = make_horses_to_2d(all_in_city['horses'], all_in_city['days']) #.reshape(-1,1)
    i = pd.DataFrame.from_dict(all_city['horses'])
    winners.extend(w_2d)
    days_arr.extend(d_2d)
    citys_arr.append(i)

    df = pd.concat(citys_arr,  axis = 0)
    df['wins'] = 0


    names = get_array(list(df['name']))
    drivers = get_array(list(df['driver']))

    
    for horse in names:
        horse_races = df.query("name == @horse")
        #print(horse)
        horse_starts = 1
        horse_wins = 0
        for index, row in horse_races.iterrows():
            df.at[index, 'horse_starts'] = horse_starts
            horse_starts += 1
            
            if row['winner'] == 1.0:
                df.at[index, "horse_wins"] = horse_wins
                df.at[index, "horse_win_prob"] = horse_wins / horse_starts
            else:
                df.at[index, "horse_wins"] =  0.0
                df.at[index, "horse_win_prob"] = horse_wins / horse_starts

 
    for d in drivers:
        driver_race = df.query("driver == @d")
        mem_starts = 1
        for index , row in driver_race.iterrows():
            df.at[index, "driver_starts"] = mem_starts
            mem_starts += 1
   
   

   
    
    horse_names = get_array(list(df['name']))
    le_horse = LabelEncoder()
    le_horse.fit(horse_names)
    df['horse_l'] = le_horse.fit_transform(df['name'])   

    driver_names = get_array(list(df['driver']))
    le_driver = LabelEncoder()
    le_horse.fit(driver_names)
    df['driver_l'] = le_driver.fit_transform(df['driver'])

    #df.pop(['wins',  'driver' ,'front_shoes', 'coach','day' ]) 
    #### MAKE OLS TEST ####
    df['prob_small'] = df['probable'] / 100
    center_function = lambda x: x - x.mean()
    df['amount_cen'] = center_function(np.array(df['amount']))


    return df



    

if __name__ == "__main__":

    
    tracks = ['Kuopio', 'Vermo', 'Pori', 'Jokimaa', 'Seinäjoki', 'Joensuu', 'Mikkeli', 'Lappeenranta', 'Oulu', 'Forssa', 'Turku', 'Jyväskylä']
    tracks2 = ['Kuopio', 'Vermo', 'Pori', 'Jokimaa', 'Seinäjoki', 'Forssa', 'Mikkeli', 'Lappeenranta']
    tracks3 = ['Kouvola']

    place = 'Pori'
    citys_arr = []
    winners = []
    days_arr = []
    for i in tracks3:
        all_city =  all_in_city = search_y_city(i)
        w_2d = np.array(all_in_city['winners'])
        d_2d = np.array(all_in_city['days'])
        #i = make_horses_to_2d(all_in_city['horses'], all_in_city['days']) #.reshape(-1,1)
        i = pd.DataFrame.from_dict(all_city['horses'])
        winners.extend(w_2d)
        days_arr.extend(d_2d)
        
        
        #i = pd.DataFrame.from_dict(res_city)
        #print(i)
        citys_arr.append(i)

    df = pd.concat(citys_arr,  axis = 0)
    df['wins'] = 0
    print(df)

    win_index = 0
    

    names = get_array(list(df['name']))
    drivers = get_array(list(df['driver']))

    #### MAKE HORSE WINMONEY AND WIN PROBA ####    
    for horse in names:
        horse_races = df.query("name == @horse")
        #print(horse)
        horse_starts = 1
        horse_wins = 0
        horse_win_money = 0.0
        for index, row in horse_races.iterrows():
            df.at[index, 'horse_starts'] = horse_starts
            horse_starts += 1
            
            if row['winner'] == 1.0:
                horse_wins += 1
                horse_win_money += float(row['win_money'])

                df.at[index, "horse_wins"] = horse_wins
                df.at[index, "horse_win_prob"] = horse_wins / horse_starts
                df.at[index, "horse_money"] = horse_win_money

            else:
                df.at[index, "horse_wins"] =  0.0123
                df.at[index, "horse_win_prob"] = horse_wins / horse_starts
                df.at[index, "horse_money"] = 0.0


 
    for d in drivers:
        driver_race = df.query("driver == @d")
        mem_starts = 1
        for index , row in driver_race.iterrows():
            df.at[index, "driver_starts"] = mem_starts
            mem_starts += 1
   
   

   
    
    horse_names = get_array(list(df['name']))
    le_horse = LabelEncoder()
    le_horse.fit(horse_names)
    df['horse_l'] = le_horse.fit_transform(df['name'])   

    driver_names = get_array(list(df['driver']))
    le_driver = LabelEncoder()
    le_horse.fit(driver_names)
    df['driver_l'] = le_driver.fit_transform(df['driver'])

    #df.pop(['wins',  'driver' ,'front_shoes', 'coach','day' ]) 
    #### MAKE OLS TEST ####
    df['prob_small'] = df['probable'] / 100
    center_function = lambda x: x - x.mean()
    df['amount_cen'] = center_function(np.array(df['amount']))

    wins_num = 2
    print(df.query("horse_wins >= @wins_num"))

    x = df[['age', 'track', 'prob_small', 'driver_starts', 'amount_cen', 'horse_starts', 'horse_win_prob', 'horse_money']]
    y = df['winner']
 
    # adding the constant term
    x = sm.add_constant(x)
 
    # performing the regression
    # and fitting the model
    result = sm.OLS(y, x).fit()
 
    # printing the summary table
    print(result.summary())
    print(df[1500:1530])

    df2 = make_horses_to_2d(df, days_arr) #.reshape(-1,1)
    df2['winner'] = winners
    print(df2)

    
    #file = open('toto_for_machine.pkl', 'wb')
    #pickle.dump(df2, file)
    #file.close()
    #print(df2)
    
    
    
    col_len = []
    for i in range(192):
        col_len.append(i)

    X = df2[col_len].to_numpy(dtype="float")
    y = df2['winner'].to_numpy(dtype="int")

    X_train,X_test,y_train,y_test = train_test_split(X, np.array(y), test_size = 0.20 ,random_state = 0)

    clf = LogisticRegression(solver='saga',max_iter=1200).fit(X_train, y_train)
    print("LogasticRegression", clf.score(X_test, y_test))

    clf_boost = GradientBoostingClassifier(learning_rate=0.2, max_depth=1, random_state=0).fit(X_train, y_train)
    print("GradientBoost", clf_boost.score(X_test, y_test))
    

    
    joblib.dump(clf_boost, "gradientBoost_"+ tracks3[0] +".pkl")
    joblib.dump(clf, "logasticRegression_" + tracks3[0]+ ".pkl")
    
    
    
    


