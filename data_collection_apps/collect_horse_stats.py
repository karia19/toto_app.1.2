

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

def make_horses(df, names):
    for horse in names:
        horse_races = df.query("name == @horse")
        #print(horse)
        horse_starts = 0
        horse_wins = 0.0
        horse_win_money = 0

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

                horse_races.at[index, "horse_wins"] =  horse_wins
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
        horse_races['winns'] = horse_races['horse_wins'].shift(1, fill_value='0.0')
        
       

        memory_index = 0
        pattern = '[a-z]+'
        
        for index, row in horse_races.iterrows():
            df.at[index, "horse_win_prob"] =  horse_races['win_prob'].iloc[memory_index]
            df.at[index, "horse_money"] = horse_races['h_money'].iloc[memory_index]
            df.at[index, "run_time_shift"] =  horse_races['time'].iloc[memory_index]
            df.at[index, "last_proba"] =  horse_races['last_pr'].iloc[memory_index]
            df.at[index, 'rest_days'] = horse_races['rest_days'].iloc[memory_index]
            df.at[index, 'last_position'] = horse_races['position_2'].iloc[memory_index]
            df.at[index, "h_w_s"] = horse_races['winns'].iloc[memory_index]
            



            memory_index += 1 

    return df