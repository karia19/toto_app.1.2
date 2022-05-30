from ast import Num
from dataclasses import dataclass
from tokenize import Number
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import drivers_stats 
import coach_stats




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
    
    f = open("toto_" + city+ ".json")
    team = json.load(f)

    race_winner = pd.DataFrame()
    race_horse = pd.DataFrame()

    try:
        index = 0
        for i in range(len(team)):
            if team[i]['place'] == city:
                winner = int(team[i]['results'][0])
                day = team[i]['day']
                race_typ = team[i]['race_type']
                race_distance = team[i]['race_distance']
            
                
                horses = team[i]['horses']
                for k in range(len(horses)):
                    horses[k]['day'] = day
                    horses[k]['race_type'] = race_typ
                    horses[k]['distance'] = race_distance
                    
                    if horses[k]['track'] == winner:
                        race_winner = race_winner.append(horses[k], ignore_index=True)  
                    else:
                    
                        race_horse = race_horse.append(horses[k], ignore_index=True)
                        
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



#res_city = search_y_city("Kaustinen")
#pdN = pd.DataFrame.from_dict(res_city)
#print(pdN)



"""
citys_arr = []
for i in tracks:
    res_city = search_y_city(i)
    i = pd.DataFrame.from_dict(res_city)
    print(i)
    citys_arr.append(i)

df = pd.concat(citys_arr,  axis = 1)
print(df)
file_name = 'starts_car_or_volt.xlsx'
df.to_excel(file_name)
"""

"""
df = pd.read_pickle("horse_track.pkl")

track = "Vermo"
driver = "Ari Moilanen"
df = df.query("place == @track")
results = df['results'].to_numpy()


for i in results:
   
    print(i)
    
"""


