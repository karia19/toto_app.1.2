import load_today_race
import pandas as pd


#team = pd.read_pickle("horses.pkl")
#print(team)


def get_array(data):
    unique = []
    for number in data:
        if number in unique:
            continue
        else:
            unique.append(number)
    return unique

def set_drivers_history(past_data, today_data, drivers):

    for d in drivers:
        try:
            drivers_race = past_data.query("driver == @d")
            drivers_last_starts = drivers_race.iloc[-1:]
            starts = float(drivers_last_starts['driver_starts'])
            win_prob = float(drivers_last_starts['d_w_pr'])
           

        except:
            starts = 0.0
            win_prob = 0.0   

        for index, row in today_data.iterrows():
            if row['driver'] == d:
                today_data.at[index, "driver_starts"] = starts
                today_data.at[index, 'd_w_pr'] = win_prob
    
    return today_data


def set_coach_history(past_data, today_data, drivers):

    for d in drivers:
        try:
            drivers_race = past_data.query("coach == @d")
            drivers_last_starts = drivers_race.iloc[-1:]
            #starts = float(drivers_last_starts['driver_starts'])
            d_win_prob = float(drivers_last_starts['c_w_pr'])
            #print(d_win_prob)

        except:
            
            d_win_prob = 0.0   

        for index, row in today_data.iterrows():
            if row['driver'] == d:
                #today_data.at[index, "driver_starts"] = starts
                today_data.at[index, 'c_w_pr'] = d_win_prob
    
    today_data.fillna(0.0)

    return today_data



def set_horse_history(past_data, today_data, drivers):

    for d in drivers:
        try:
            horse_race = past_data.query("name == @d")
            horse_last_starts = horse_race.iloc[-1:]
            try:
                proba = float(horse_last_starts['probable'])
            except:
                proba = 0.0
            
            try:
                pos = float(horse_last_starts['position'])
            except:
                pos = 0.0

            try:
                last_win = float(horse_last_starts['horse_wins'])
            except:
                last_win = 0.0
           
            try:
                d_win_prob = float(horse_last_starts['c_w_pr'])
            except:
                d_win_prob = 0.0

        except:
            pos = 0.0
            proba = 0.0
            last_win = 0.0
            d_win_prob = 0.0   

        for index, row in today_data.iterrows():
            if row['name'] == d:
                today_data.at[index, "last_proba"] = proba
                today_data.at[index, "last_run"] = pos
                today_data.at[index, "h_w_S"] = last_win
                today_data.at[index, "c_w_pr"] = d_win_prob
    
    return today_data

def main_today(city):
    team = pd.read_pickle("horses.pkl")
    today_race = load_today_race.make_horses(city)
    #print(today_race)
    df3 = pd.DataFrame()

    df = today_race['horses']

    drives = get_array(list(df['driver']))
    horses = get_array(list(df['name']))
    coaches = get_array(list(df['coach']))
    ### COLLETCT TO DAY HORSES####
    

    df2 = set_drivers_history(team, df, drives)
    df3 = set_horse_history(team, df2, horses)
    #df3 = set_coach_history(team, df2, coaches)
    start_nums = get_array(list(df3['start_num']))
    #print(df3, start_nums)

    return { "horses": df3.to_json(orient="records"), "starts": start_nums }

#res = main_today("Kouvola")
#print(res)
