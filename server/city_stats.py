from ast import Num
from dataclasses import dataclass
from tokenize import Number
import joblib
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import drivers_stats 
import coach_stats
import pickle




label_horse_gender = LabelEncoder()
label_race_type = LabelEncoder()


def get_names_from_array(data):
    unique = []
    for number in data:
        if number in unique:
            continue
        else:
            unique.append(number)
    return unique

def start_type(data, races_run):
    data_dict = {}

    for start in range(1,17):
        occures = 0

        for i in range(len(data)):
            try:
                
                if start == data[i]:
                    occures += 1
            
            except:
                print("no value")

        data_dict.__setitem__(start, round(occures / races_run, 3) * 100)

    
    #print(data_dict)
    return data_dict

def win_shoes(data):
    all_s = "HAS_SHOES"
    no_s = "NO_SHOES"
    
    res_shoes = data.query("front_shoes == @all_s and rear_shoes == @all_s")
    res_no_shoes = data.query("front_shoes == @no_s and rear_shoes == @no_s")
    print(len(res_shoes) / len(data), len(res_no_shoes) / len(data)) 


def search_y_city(city):
    
    team = pd.read_pickle("horses.pkl")
    #team = json.load(f)
    team = team.query("race_city == @city")
    race_winner = pd.DataFrame()
    race_horse = pd.DataFrame()

    try:
        
        for index, row in team.iterrows():
            if row['winner'] == 1.0:
                race_winner = race_winner.append(row, ignore_index=True)  
            else:
                race_horse = race_horse.append(row, ignore_index=True)

    except:
        print("er")

    car_start = "CAR_START"
    volt_start = "VOLT_START"
    start_num = 1
    races_in_data =  len(race_horse.query("track == @start_num")) + len(race_winner)
    #df = pd.concat([race_winner, race_horse],  axis = 1) 

    car = race_winner.query("race_type == @car_start")
    car_res = start_type(list(car['track']), races_in_data)

    volt = race_winner.query("race_type == @volt_start")
    volt_res = start_type(list(volt['track']), races_in_data)

    """
    horse_gender = get_names_from_array(race_winner['gender'])
    label_horse_gender.fit(horse_gender)
    race_winner['ge_num'] = label_horse_gender.fit_transform(race_winner['gender'])
    """
   
    drivers_on_track = drivers_stats.drivers(race_winner, race_horse)
    coach_on_track = coach_stats.coach(race_winner, race_horse)
    print("drivers:")
    print(drivers_on_track)
    print("coach;")
    print(coach_on_track)

    win_shoes(race_winner)

    #print(race_horse[40:57])
    #print(race_horse)
    pdN = pd.DataFrame.from_dict({ "car": car_res, "volt": volt_res})
    print(pdN)

    return { "car": pdN['car'].to_json(orient="records"), "volt": pdN['volt'].to_json(orient="records") , 
                "coach": coach_on_track.to_json(orient="records"),
                "drivers": drivers_on_track.to_json(orient="records") }

#def find_drivers(arr, arr2):
#    print(np.intersect1d(arr, arr2))


#tracks = ['Kuopio', 'Vermo', 'Pori', 'Jokimaa', 'Seinäjoki', 'Joensuu', 'Mikkeli', 'Lappeenranta', 'Oulu', 'Forssa', 'Turku', 'Jyväskylä']


"""
city = "Vermo"

res_city = search_y_city(city)

with open('track_stats/' + city + ".pkl", 'wb') as handle:
    pickle.dump(res_city, handle, protocol=pickle.HIGHEST_PROTOCOL)

"""


"""
with open('track_stats/' + city + '.pkl', 'rb') as handle:
    b = pickle.load(handle)
    print(b)
"""


