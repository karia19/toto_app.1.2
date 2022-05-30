
from ast import Num
from time import process_time_ns
from click import FloatRange
import requests
import pandas as pd
import datetime
import pickle
import json
import re

s = requests.Session()

#api_url = "https://www.veikkaus.fi/api/toto-info/v1/pool/5018558/odds"

#day = "2022-04-30"

all_in_one_json = []


def get_horses(day):
    print(day)
    try:   
        df = pd.DataFrame()

        api_url = "https://www.veikkaus.fi/api/toto-info/v1/cards/date/" + day
        response = s.get(api_url)
        what = response.json()

        for i in range(len(what['collection'])):
            if what['collection'][i]['country'] == "FI" and what['collection'][i]['trackName'] == 'Vieremä':
                print("cellction", i)
                collection_track_id = i 


        to_day_cardId = what['collection'][collection_track_id]['cardId']
        race_place = what['collection'][collection_track_id]['trackName']
        
        print(to_day_cardId)

        if race_place != "Tu":

            res = s.get("https://www.veikkaus.fi/api/toto-info/v1/card/"+ str(to_day_cardId) + "/races")
            res_j = res.json()
            res_le = res_j['collection']

            race_ID = []
            race_results = []
            race_type = []
            race_riders = []
            #reverse_order = []
            race_distance = []
            start_win_money = []

            for i in range(len(res_le)):
                race_ID.append(res_le[i]['raceId'])
                race_results.append(res_le[i]['toteResultString'].split("-"))
                race_type.append(res_le[i]['startType'])
                race_riders.append(res_le[i]['raceStatus'])
                #reverse_order.append(res_le[i]['reserveHorsesOrder'].split("-"))
                race_distance.append(res_le[i]['distance'])
                start_win_money.append(res_le[i]["firstPrize"])

            print(race_ID)

            pool_ids = []
        
            horse_names_2d = []
            horse_age_2d = []
            driver_name_2d = []
            win_money_2d = []
            gender_2d = []
            starts_2d = []
            first_postion_2d = []


            horse_front_shoes_2d = []
            horse_rear_shoes_2d = []
            horse_coach_2d = []
            horse_city_2d = []
        
            for i in race_ID:
                res = s.get("https://www.veikkaus.fi/api/toto-info/v1/race/"+ str(i) + "/pools")
                res_j = res.json()

                race_horses = s.get("https://www.veikkaus.fi/api/toto-info/v1/race/" + str(i) + "/runners")
                race_horses_json = race_horses.json()
                race_horses_json_all = race_horses_json['collection']


                race_times = s.get("https://www.veikkaus.fi/api/toto-info/v1/race/" + str(i) +"/competition-results")
                race_time_json = race_times.json()

                race_time_runners = race_time_json['runners']
                
                

                horse_race_time = []

                horse_name = []
                horse_age = [] 
                driver_name = []
            
                gender = []   
                this_yaar_start = []
                this_yaar_1 = []
                this_yaar_2 = []
                this_yaar_winMoney = []

                horse_front_shoes = []
                horse_rear_shoes = []
                horse_coach = []
                horse_city = []

                for i in range(len(race_horses_json_all)):
                    #horse_name.append(race_horses_json_all[i]['horseName'])
                    #horse_age.append(race_horses_json_all[i]['horseAge'])
                    #driver_name.append(race_horses_json_all[i]['driverName'])

                    
                    try:
                        horse_race_time.append(race_time_runners[i]['kmTime'])

                        horse_name.append(race_horses_json_all[i]['horseName'])
                        horse_age.append(race_horses_json_all[i]['horseAge'])
                        driver_name.append(race_horses_json_all[i]['driverName'])
                        

                        
                        horse_front_shoes.append(race_horses_json_all[i]['frontShoes']) 
                        horse_rear_shoes.append(race_horses_json_all[i]['rearShoes'])
                        horse_city.append(race_horses_json_all[i]['ownerHomeTown'])
                        horse_coach.append(race_horses_json_all[i]['coachName'])

                        gender.append(race_horses_json_all[i]['gender'])
                        this_yaar_start.append(race_horses_json_all[i]['stats']['currentYear']['starts'])
                        this_yaar_winMoney.append(race_horses_json_all[i]['stats']['currentYear']['winMoney'])
                        this_yaar_1.append(race_horses_json_all[i]['stats']['currentYear']['position1'])
                        #this_yaar_2.append(race_horses_json_all[i]['stats']['currentYear']['position2'])
                    



                    except:
                        print("err from horse json")
                        #gender.append(race_horses_json_all[i]['gender'])
                        this_yaar_start.append(0)
                        this_yaar_winMoney.append(0)
                        this_yaar_1.append(0)
                        this_yaar_2.append(0)

                    
                horse_names_2d.append([horse_name])
                horse_age_2d.append([horse_age])
                driver_name_2d.append([driver_name])
                gender_2d.append([gender])

                win_money_2d.append([this_yaar_winMoney])
                starts_2d.append([this_yaar_start])       
                first_postion_2d.append([this_yaar_1]) 


                horse_front_shoes_2d.append(horse_front_shoes)
                horse_rear_shoes_2d.append(horse_rear_shoes)
                horse_coach_2d.append(horse_coach)
                horse_city_2d.append(horse_city)

            
                
                #win_money_2d.append([win_money])

                pool_ids.append(res_j['collection'][0]['poolId'])


        

            start_index = 0
            start_index_2 = -1
            track_numebr = []
           

            index_for_time = 0

            pattern = '[a-z]+'

            for i in pool_ids:
                res = s.get("https://www.veikkaus.fi/api/toto-info/v1/pool/"+ str(i) + "/odds")
                res_j = res.json()

                #print(start_index)
                #print(res_j)
                start_index += 1
                start_index_2 += 1
                index_for_time = 0


                net_sale = res_j['netSales'] / 100
                #print(net_sale)
                odds = res_j['odds']
                #print(odds)

                horses_for_json = []
                for k in range(len(odds)):
                    
                    try:
                        if horse_race_time[index_for_time] == "-":
                            h_time = 0.0
                        else:
                            h_time = float(re.sub(pattern, "", horse_race_time[index_for_time].replace(",", ".").replace("-", "0.0")))
                      
                        horses_for_json.append({"track": odds[k]['runnerNumber'],
                                "start_num": start_index,
                                "name": horse_names_2d[start_index_2][0][k],
                                "age": horse_age_2d[start_index_2][0][k],
                                "starts": starts_2d[start_index_2][0][k],
                                "postion1": first_postion_2d[start_index_2][0][k],
                                "driver": driver_name_2d[start_index_2][0][k],
                                "win_money": win_money_2d[start_index_2][0][k],
                                "gender": gender_2d[start_index_2][0][k],
                                "probable": odds[k]['probable'] / 100,
                                "amount": odds[k]['amount'] / 100,

                                
                                "front_shoes": horse_front_shoes_2d[start_index_2][k],
                                "rear_shoes": horse_rear_shoes_2d[start_index_2][k],
                                "coach": horse_coach_2d[start_index_2][k],
                                "home_town": horse_city_2d[start_index_2][k],
                                
                                "run_time": h_time                               
                                
                                })
                        index_for_time += 1
                        
                    except:
                        print("err from array append")
                        index_for_time += 1
                    

                        try:
                            track_numebr.pop()
                        
                            #horse_names_2d[start_index_2][0].pop(k)

                        except:
                            print("pop not need")

                all_in_one_json.append({"day": day, 'place': race_place, "start_num": start_index, "results": race_results[start_index_2],
                                "race_type": race_type[start_index_2],
                                "race_distance": race_distance[start_index_2], "win_money": start_win_money[start_index_2],
                                "horses": horses_for_json })


        else:
            print("no")


    except:
        print("err")
    

   

   
    return all_in_one_json
        

if __name__ == "__main__":

    
    today = datetime.datetime.now()
    last_years = today - datetime.timedelta(days=328)

    days = []
    
    for n in range(1,328):
        arrive = last_years + datetime.timedelta(days=n)
        days.append(arrive.strftime('%Y-%m-%d'))

    for day in days:
       day = get_horses(day)
    
    #print(all_in_one_json)
       

    
    with open('toto_vieremä.json', 'w') as f:
        json.dump(all_in_one_json, f)
    
    
