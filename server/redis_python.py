import redis
import load_today_race
import json
from datetime import datetime

r = redis.StrictRedis(host='redis', port=6379, db=0)

def save_data_redis(city):

    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    current_day = now.strftime("%Y-%m-%d")
    
    res_t_day = load_today_race.make_horses(city)
    df = res_t_day['horses']

    for index, row in df.iterrows():
        df.at[index, "update_time"] = current_time
        df.at[index, "city"] = city
    
    data_in_json = df[['name', 'track', 'start_num', 'probable', 'update_time', 'day', "city"]].to_json(orient="records")
   
    testi = json.loads(data_in_json)

    
    added_data =  get_data_redis(testi, current_day)
    r.set(current_day, json.dumps(added_data))


def get_data_redis(data , todays):
    try:
        last_data_points = json.loads(r.get(todays))
        add_to_data = last_data_points    
        
        for i in range(len(data)):
            add_to_data.append(data[i])

        #print(add_to_data)
        #print(len(add_to_data))

        return add_to_data
    except:
        return data

def get_only_data(city):
    now = datetime.now()
    current_day = now.strftime("%Y-%m-%d")
    
    try:
        last_data_points = json.loads(r.get(current_day))
        #add_to_data = json.loads(last_data_points)
        #print(last_data_points)
        return json.dumps(last_data_points)
    
    except:
        print("error")
        return {"message": "error"}

def get_history_ods(day_num):
    
    try:
        history_ods = json.loads(r.get(day_num))

        return json.dumps(history_ods)

    except:
        return { "messgae" : "no data ..."}   





#save_data_redis("Kajaani")
#get_only_data('Kajaani')
