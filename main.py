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


def get_array(data):
    unique = []
    for number in data:
        if number in unique:
            continue
        else:
            unique.append(number)
    return unique



if __name__ == "__main__":
    place = 'Vermo'
    
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

    print(df)
    #df_place_horses = machine_all.collect_all_data(place)
    #print(df_place_horses)