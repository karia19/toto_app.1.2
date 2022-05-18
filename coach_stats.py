from logging import logProcesses
import pandas as pd

coach_name = []
coach_prob = []
coach_starts = []
coach_loses = []

def coach(data_wins, data):
    drivers_array = find_coach_to_array(list(data_wins['driver']))
    #print(drivers_array)
   
    for name in drivers_array:
        res_from_drivers = find_coach_prob_to_win(data_wins, data, name)
      
   
    df = pd.DataFrame()
    df['name'] = coach_name
    df['wins'] = coach_starts
    df['starts'] = coach_loses
    df['proba'] = coach_prob
    df = df.sort_values(by=['starts'], ascending=False)
    print(df[:20])

    return df[:20]

def find_coach_to_array(data_wins):
    unique = []
    for number in data_wins:
        if number in unique:
            continue
        else:
            unique.append(number)
    return unique


def find_coach_prob_to_win(data_wins, data, name):
    start_num = 1
    win = data_wins.query("coach == @name")
    lose = data.query("coach == @name")
    df = pd.concat([data_wins, data],  axis = 1) 

    
    #no_win = df.query("start_num == @start_num")

    try:
        prob_win = round(len(win) / (len(lose) + len(win)), 3) * 100
        coach_prob.append(prob_win)
        coach_name.append(name)
        coach_starts.append(len(win))
        coach_loses.append(len(lose) + len(win))
        #print(driver_name + " win %: " + str(prob_win))


    except ZeroDivisionError:
        print("No data")

