import pandas as pd
from datetime import datetime



def make_coach(df, names):
     for horse in names:
        horse_races = df.query("coach == @horse")
        #print(horse)
        horse_starts = 1
        horse_wins = 0.0
        #horse_win_money = 0.0

        days_bbetween_index = 0

    
        for index, row in horse_races.iterrows():
            df.at[index, 'driver_starts'] = horse_starts
            horse_starts += 1
            
        
            
            if row['winner'] == 1.0:
                horse_wins += 1

                df.at[index, "coach_wins"] = horse_wins
               
                horse_races.at[index, "coach_wins"] = horse_wins
                horse_races.at[index, "coach_win_prob"] = horse_wins / horse_starts
              
            else:
                df.at[index, "coach_wins"] =  horse_wins
                horse_races.at[index, "coach_win_prob"] = horse_wins / horse_starts
               
        ### SHIFT DATA TO PAST ###
        horse_races['win_prob'] = horse_races['coach_win_prob'].shift(1, fill_value=0)
        #horse_races['h_money'] = horse_races['horse_money'].shift(1, fill_value=0)
        #horse_races['last_pr'] = horse_races['probable'].shift(1, fill_value=0)
        #horse_races['time'] = horse_races['run_time'].shift(1, fill_value='0.0')
        #horse_races['position_2'] = horse_races['position'].shift(1, fill_value='0.0')
        
        memory_index = 0
        #pattern = '[a-z]+'
        
        for index, row in horse_races.iterrows():
            df.at[index, "c_w_pr_s"] =  horse_races['win_prob'].iloc[memory_index]
            #df.at[index, "horse_money"] = horse_races['h_money'].iloc[memory_index]
            #df.at[index, "run_time_shift"] =  horse_races['time'].iloc[memory_index]
            #df.at[index, "last_proba"] =  horse_races['last_pr'].iloc[memory_index]
            #df.at[index, 'd_r_days'] = horse_races['driver_rest_days'].iloc[memory_index]
            df.at[index, 'c_w_pr'] = horse_races['coach_win_prob'].iloc[memory_index]



            memory_index += 1


     return df
    