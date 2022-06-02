import pandas as pd
from datetime import datetime

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

def make_drivers(df, names):
     for horse in names:
        horse_races = df.query("driver == @horse")
        #print(horse)
        horse_starts = 1
        horse_wins = 0.0
        #horse_win_money = 0.0

        days_bbetween_index = 0

        rest_days =  days_between_races(list(horse_races['day']))
        horse_races['driver_rest_days'] = rest_days 
        

        for index, row in horse_races.iterrows():
            df.at[index, 'driver_starts'] = horse_starts
            horse_starts += 1
            
        
            
            if row['winner'] == 1.0:
                horse_wins += 1

                df.at[index, "driver_wins"] = horse_wins
               
                horse_races.at[index, "driver_wins"] = horse_wins
                horse_races.at[index, "driver_win_prob"] = horse_wins / horse_starts
              
            else:
                df.at[index, "driver_wins"] =  horse_wins
                horse_races.at[index, "driver_win_prob"] = horse_wins / horse_starts
               
        ### SHIFT DATA TO PAST ###
        horse_races['win_prob'] = horse_races['driver_win_prob'].shift(1, fill_value=0)
        #horse_races['h_money'] = horse_races['horse_money'].shift(1, fill_value=0)
        #horse_races['last_pr'] = horse_races['probable'].shift(1, fill_value=0)
        #horse_races['time'] = horse_races['run_time'].shift(1, fill_value='0.0')
        #horse_races['position_2'] = horse_races['position'].shift(1, fill_value='0.0')
        
        memory_index = 0
        #pattern = '[a-z]+'
        
        for index, row in horse_races.iterrows():
            df.at[index, "d_w_pr_s"] =  horse_races['win_prob'].iloc[memory_index]
            #df.at[index, "horse_money"] = horse_races['h_money'].iloc[memory_index]
            #df.at[index, "run_time_shift"] =  horse_races['time'].iloc[memory_index]
            #df.at[index, "last_proba"] =  horse_races['last_pr'].iloc[memory_index]
            df.at[index, 'd_r_days'] = horse_races['driver_rest_days'].iloc[memory_index]
            df.at[index, 'd_w_pr'] = horse_races['driver_win_prob'].iloc[memory_index]



            memory_index += 1


     return df
    