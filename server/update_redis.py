import redis_python
import load_today_race

def make_update_half_hour():
        try:
          today_city = load_today_race.check_races_for_city() 
          redis_python.save_data_redis(today_city)  
          #print(today_city)
        except:
          return {"message": "err in data update"}


make_update_half_hour()
