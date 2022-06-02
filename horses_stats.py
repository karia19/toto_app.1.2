import pandas as pd
import json

f = open('toto_starts_2021.json')
team = json.load(f)

horse_name = []
horse_prob = []
horses_starts = []
horses_loses = []


def horses_to_array(data_wins):
    unique = []
    for number in data_wins:
        if number in unique:
            continue
        else:
            unique.append(number)
    return unique


def collect_all():
    race_winner = pd.DataFrame()
    race_horse = pd.DataFrame()

    index = 0
    for i in range(len(team)):
            try:
                winner = int(team[i]['results'][0])
                day = team[i]['day']
                race_typ = team[i]['race_type']
                race_distance = team[i]['race_distance']
                race_place = team[i]['place']
            
                
                horses = team[i]['horses']
                for k in range(len(horses)):
                    horses[k]['day'] = day
                    horses[k]['race_type'] = race_typ
                    horses[k]['distance'] = race_distance
                    horses[k]['place'] = race_place

                    if horses[k]['track'] == winner:
                        race_winner = race_winner.append(horses[k], ignore_index=True)  
                    else:
                    
                        race_horse = race_horse.append(horses[k], ignore_index=True)
            except:
                print("no")
    print(race_winner)
    race_winner.to_pickle("horses_win3.pkl")
    race_horse.to_pickle("horses_n_race3.pkl")
    return race_winner                    


def find_horses_prob_to_win(data_wins, data, name):
    start_num = 1
    win = data_wins.query("name == @name")
    lose = data.query("name == @name")
    df = pd.concat([data_wins, data],  axis = 1) 

    
    #no_win = df.query("start_num == @start_num")

    try:
        prob_win = round(len(win) / (len(lose) + len(win)), 3) * 100
        horse_prob.append(prob_win)
        horse_name.append(name)
        horses_starts.append(len(win))
        horses_loses.append(len(lose) + len(win))
        #print(name + " win %: " + str(prob_win))


    except ZeroDivisionError:
        print("No data")



if __name__ == "__main__":
    collect_all()
    