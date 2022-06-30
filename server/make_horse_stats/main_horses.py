
import pandas as pd
import json
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
import statsmodels.api as sm
import joblib
import pickle
from datetime import datetime
import collect_horse_stats
import collect_drivers_stats
import collect_coach_stats


f = open("toto_starts_2021.json")
#f = open("toto_starts_2021.json")
team = json.load(f)

for i in range(len(team)):
    if team[i]['day'] == '2021-07-17':
        print(team[i])

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

def days_between_races(days):
    
    days_arr = [0]
    try:
        for i in range(len(days)):
            d1 = datetime.strptime(days[i], "%Y-%m-%d")
            d2 = datetime.strptime(days[i +1], "%Y-%m-%d")
            days_arr.append(abs((d2 - d1).days))
        
    except:
        print("")
    return days_arr

def all_horses():
   

    race_winner = []
    race_second = []
    race_horse = pd.DataFrame()
    days = []
    

    print(len(team))
    try:
        index = 0
        for i in range(len(team)):
              
                # NOT PLACE IN USE
                try:
                    city = team[i]['place']
            
                    race_winner.append(int(team[i]['results'][0]))
                    race_second.append(int(team[i]['results'][1]))

                    win = int(team[i]['results'][0])
                    win_money = int(team[i]['win_money'])

                    #race_second.append(int(team[i]['results'][1]))

                    day = team[i]['day']
                    days.append(day)
                    race_typ = team[i]['race_type']
                    race_distance = team[i]['race_distance']

                    print(race_distance, race_typ, day, win_money, win, race_winner)
            
                
                    horses = team[i]['horses']
                  
                    for k in range(len(horses)):
                        try:
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

                            if horses[k]['track'] == race_second:
                                horses[k]['second'] = 2.0
                                
                            else:
                                horses[k]['second'] = 0.0
                                #horses[k]['win_money'] = 0.0
                        except:
                            print("erro in horses")


                        race_horse = race_horse.append(horses[k], ignore_index=True)
                except:
                    print("ee")
                        
    except:
        print("er")

    #print(race_horse.sort_values(['day']))

    return {"horses": race_horse,"winners": race_winner, "days": list(dict.fromkeys(days)) }



if __name__ == "__main__":
    
    citys_arr = []
    winners = []
    days_arr = []
    
    all_city = all_horses()
    w_2d = np.array(all_city['winners'])
    d_2d = np.array(all_city['days'])
    print(d_2d)
        #i = make_horses_to_2d(all_in_city['horses'], all_in_city['days']) #.reshape(-1,1)
    df = pd.DataFrame.from_dict(all_city['horses'])
    winners.extend(w_2d)
    days_arr.extend(d_2d)

    df['wins'] = 0
    print(df)


    
    win_index = 0
    

    names = get_array(list(df['name']))
    drivers = get_array(list(df['driver']))
    coach_names = get_array(list(df['coach']))

    df = collect_horse_stats.make_horses(df, names)
    df = collect_drivers_stats.make_drivers(df, drivers)
    df = collect_coach_stats.make_coach(df, coach_names)
    """
    #### MAKE HORSE WINMONEY AND WIN PROBA ####    
    for horse in names:
        horse_races = df.query("name == @horse")
        #print(horse)
        horse_starts = 1
        horse_wins = 0.0
        horse_win_money = 0.0

        days_bbetween_index = 0

        res_days =  days_between_races(list(horse_races['day']))
        horse_races['rest_days'] = res_days 
        

        for index, row in horse_races.iterrows():
            df.at[index, 'horse_starts'] = horse_starts
            horse_starts += 1
            
            horse_win_money += float(row['moneys'])
            
            if row['winner'] == 1.0:
                horse_wins += 1

                df.at[index, "horse_wins"] = horse_wins
                #df.at[index, "horse_win_prob"] = horse_wins / horse_starts
                #df.at[index, "horse_money"] = horse_win_money

                horse_races.at[index, "horse_wins"] = horse_wins
                horse_races.at[index, "horse_win_prob"] = horse_wins / horse_starts
                horse_races.at[index, "horse_money"] = horse_win_money
                
                
                #days_between =  days_between_races(horse_races['day'].iloc[days_bbetween_index], horse_races['day'].iloc[days_bbetween_index + 1])
                #days_bbetween_index += 1
                

                
            else:
                df.at[index, "horse_wins"] =  horse_wins
                #df.at[index, "horse_win_prob"] = horse_wins / horse_starts
                #df.at[index, "horse_money"] = 0.0

                #horse_races.at[index, "horse_wins"] =  0.0123
                horse_races.at[index, "horse_win_prob"] = horse_wins / horse_starts
                horse_races.at[index, "horse_money"] = horse_win_money


                #days_between =  days_between_races(horse_races['day'].iloc[days_bbetween_index], horse_races['day'].iloc[days_bbetween_index + 1])
                #days_bbetween_index += 1

        ### SHIFT DATA TO PAST ###
        horse_races['win_prob'] = horse_races['horse_win_prob'].shift(1, fill_value=0)
        horse_races['h_money'] = horse_races['horse_money'].shift(1, fill_value=0)
        horse_races['last_pr'] = horse_races['probable'].shift(1, fill_value=0)
        horse_races['time'] = horse_races['run_time'].shift(1, fill_value='0.0')
        horse_races['position_2'] = horse_races['position'].shift(1, fill_value='0.0')
        
        memory_index = 0
        pattern = '[a-z]+'
        
        for index, row in horse_races.iterrows():
            df.at[index, "horse_win_prob"] =  horse_races['win_prob'].iloc[memory_index]
            df.at[index, "horse_money"] = horse_races['h_money'].iloc[memory_index]
            df.at[index, "run_time_shift"] =  horse_races['time'].iloc[memory_index]
            df.at[index, "last_proba"] =  horse_races['last_pr'].iloc[memory_index]
            df.at[index, 'rest_days'] = horse_races['rest_days'].iloc[memory_index]
            df.at[index, 'last_position'] = horse_races['position_2'].iloc[memory_index]



            memory_index += 1
 
    for d in drivers:
        driver_race = df.query("driver == @d")
        mem_starts = 1
        for index , row in driver_race.iterrows():
            df.at[index, "driver_starts"] = mem_starts
            mem_starts += 1
   
    

   
    """
    horse_names = get_array(list(df['name']))
    le_horse = LabelEncoder()
    le_horse.fit(horse_names)
    df['horse_l'] = le_horse.fit_transform(df['name'])   

    driver_names = get_array(list(df['driver']))
    le_driver = LabelEncoder()
    le_horse.fit(driver_names)
    df['driver_l'] = le_driver.fit_transform(df['driver'])

    #df.drop([['starts', 'wins']])
    #df.drop(['starts', 'name', 'driver'], axis=1, inplace=True)
    print(df)
    
    
    #df.to_pickle("horses_for_demo.pkl")
    
