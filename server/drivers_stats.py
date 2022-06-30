from logging import logProcesses
import pandas as pd

drivers_name = []
driver_prob = []
drivers_starts = []
drivers_loses = []

def drivers(data_wins, data):
    drivers_array = find_drivers_to_array(list(data_wins['driver']))
    #print(drivers_array)
   
    for name in drivers_array:
        res_from_drivers = find_drivers_prob_to_win(data_wins, data, name)
      
   
    df = pd.DataFrame()
    df['name'] = drivers_name
    df['wins'] = drivers_starts
    df['starts'] = drivers_loses
    df['proba'] = driver_prob
    df = df.sort_values(by=['wins'], ascending=False)
   #print(df[:16])

    return df[:30]


def find_drivers_to_array(data_wins):
    unique = []
    for number in data_wins:
        if number in unique:
            continue
        else:
            unique.append(number)
    return unique


def find_drivers_prob_to_win(data_wins, data, driver_name):
    start_num = 1
    win = data_wins.query("driver == @driver_name")
    lose = data.query("driver == @driver_name")
    df = pd.concat([data_wins, data],  axis = 1) 

    
    #no_win = df.query("start_num == @start_num")

    try:
        prob_win = round(len(win) / (len(lose) + len(win)), 3) * 100
        driver_prob.append(prob_win)
        drivers_name.append(driver_name)
        drivers_starts.append(len(win))
        drivers_loses.append(len(lose) + len(win))
        #print(driver_name + " win %: " + str(prob_win))


    except ZeroDivisionError:
        print("No data")

