from traceback import print_tb
import pandas as pd
import datetime
import requests
import re

tracks = ['Kuopio', 'Vermo', 'Härmä', 'Pori', 'Ylivieska', 'Jokimaa', 'Kaustinen', 
            'Seinäjoki', 'Joensuu', 'Kouvola', 'Mikkeli', 'Lappeenranta', 'Oulu', 
            'Forssa', 'Turku', 'Joensuu', 'Jyväskylä','Teivo', 'Vieremä', 'Kuopio', 'Kajaani']



def check_races_for_city():
    s = requests.Session()
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    try:

        api_url = "https://www.veikkaus.fi/api/toto-info/v1/cards/date/" + today
        response = s.get(api_url)
        what = response.json()
        #print(what)

        for i in range(len(what['collection'])):
            if what['collection'][i]['country'] == "FI" and what['collection'][i]['trackName'] in tracks:
                #print(what['collection'][i]['trackName'])
                collection_track_id = i 
                return what['collection'][i]['trackName']

    except:
        print("not find")
        return "not find"



def get_todays_race(city):
    all_in_one_json = []
    s = requests.Session()
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    print(today)
   
    try:
        df = pd.DataFrame()

        api_url = "https://www.veikkaus.fi/api/toto-info/v1/cards/date/" + today
        response = s.get(api_url)
        what = response.json()
        #print(what)

        for i in range(len(what['collection'])):
            if what['collection'][i]['country'] == "FI" and what['collection'][i]['trackName'] == city:
                print("cellction", i)
                collection_track_id = i 


        to_day_cardId = what['collection'][collection_track_id]['cardId']
        race_place = what['collection'][collection_track_id]['trackName']

        res = s.get("https://www.veikkaus.fi/api/toto-info/v1/card/"+ str(to_day_cardId) + "/races")
        res_j = res.json()
        res_le = res_j['collection']


        race_ID = []
        
        race_type = []
        race_riders = []
        reverse_order = []
        race_distance = []

        for i in range(len(res_le)):
                race_ID.append(res_le[i]['raceId'])
                #race_results.append(res_le[i]['toteResultString'].split("-"))
                race_type.append(res_le[i]['startType'])
                race_riders.append(res_le[i]['raceStatus'])
                reverse_order.append(res_le[i]['reserveHorsesOrder'].split("-"))
                race_distance.append(res_le[i]['distance'])
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
                
                

                horse_race_time = []
                race_results = []

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


                        horse_name.append(race_horses_json_all[i]['horseName'])
                        horse_age.append(race_horses_json_all[i]['horseAge'])
                        driver_name.append(race_horses_json_all[i]['driverName'])
                        

                        
                        horse_front_shoes.append(race_horses_json_all[i]['frontShoes']) 
                        horse_rear_shoes.append(race_horses_json_all[i]['rearShoes'])
                        horse_city.append(race_horses_json_all[i]['ownerHomeTown'])
                        horse_coach.append(race_horses_json_all[i]['coachName'])
                        
                        try:
                            horse_race_time.append(race_horses_json_all[i]['stats']['currentYear']['record1'])
                        except:
                            horse_race_time.append("0.0")
                            
                            
                        gender.append(race_horses_json_all[i]['gender'])
                        this_yaar_start.append(race_horses_json_all[i]['stats']['currentYear']['starts'])
                        this_yaar_winMoney.append(race_horses_json_all[i]['stats']['currentYear']['winMoney'])
                        this_yaar_1.append(race_horses_json_all[i]['stats']['currentYear']['position1'])
                        
                        #this_yaar_2.append(race_horses_json_all[i]['stats']['currentYear']['position2'])
                    
                        #print(race_horses_json_all[i]['stats']['currentYear']['record2'])



                    except:
                        print("err from horse json")

                        #gender.append(race_horses_json_all[i]['gender'])
                        this_yaar_start.append(0)
                        this_yaar_winMoney.append(0)
                        this_yaar_1.append(0)
                        #this_yaar_2.append(0)

                        

                    
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
        ods_for_horse = []
        money_fro_horse = []
        money_total = []
        played_day = []
        start_number = []
        race_tyoe = []
        race_riders_arr = []
        place = []
        race_res_2 = []
        track_order = []
        track_distance = []
        horses_name = []
        horses_age = []
        drivers = []
        #horses_money = []
        genders = []
        win_moneys = []

        starts = []
        first_places = []

        print("this is poolid", pool_ids)

        pattern = '[a-z]+'
        for i in pool_ids:
                res = s.get("https://www.veikkaus.fi/api/toto-info/v1/pool/"+ str(i) + "/odds")
                res_j = res.json()

                #print(start_index)
                #print(res_j)
                start_index += 1
                start_index_2 += 1
                index_for_time = 0


                #net_sale = res_j['netSales'] / 100
                #print(net_sale)
                odds = res_j['odds']
                #print(odds)

                horses_for_json = []
                for k in range(len(odds)):
                   
                    try:
                        track_numebr.append(odds[k]['runnerNumber'])
                        
                        
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
                                "horse_run_time": h_time                               
                               
                                
                                })
                        index_for_time += 1

                    except:
                        print("err from array append")
                        index_for_time += 1
                        
                        ods = 0.0
                        amo = 0.0
                        try:
                            ods = odds[k]['probable'] / 100
                            amo =  odds[k]['amount'] / 100
                        except:
                            ods = 0.0
                            amo = 0.0
                        

                        horses_for_json.append({"track": odds[k]['runnerNumber'],
                                "start_num": start_index,
                                "name": horse_names_2d[start_index_2][0][k],
                                "age": horse_age_2d[start_index_2][0][k],
                                "starts": starts_2d[start_index_2][0][k],
                                "postion1": first_postion_2d[start_index_2][0][k],
                                "driver": driver_name_2d[start_index_2][0][k],
                                "win_money": win_money_2d[start_index_2][0][k],
                                "gender": gender_2d[start_index_2][0][k],
                                "probable": ods, #odds[k]['probable'] / 100,
                                "amount": amo, #odds[k]['amount'] / 100,

                                
                                "front_shoes": horse_front_shoes_2d[start_index_2][k],
                                "rear_shoes": horse_rear_shoes_2d[start_index_2][k],
                                "coach": horse_coach_2d[start_index_2][k],
                                "home_town": horse_city_2d[start_index_2][k],
                                "horse_run_time": h_time                               
                               
                                
                        })

                        """
                        print( {"track": odds[k]['runnerNumber'],
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
                                "home_town": horse_city_2d[start_index_2][k]                               
                        })
                        """
                        
                        
                        
                    

                        try:
                            track_numebr.pop()

                        
                            #horse_names_2d[start_index_2][0].pop(k)

                        except:
                            print("pop not need")

                all_in_one_json.append({"day": today, 'place': race_place, "start_num": start_index,
                            "reverse_order": reverse_order[start_index_2], "race_type": race_type[start_index_2],
                            "race_distance": race_distance[start_index_2],
                            "horses": horses_for_json })

        #print(all_in_one_json)

    
    except:
        print("err in data") 

   

    #return df
    return all_in_one_json

def make_horses(city):
    

    data = get_todays_race(city)
    race_horse = pd.DataFrame()
    days = []

    try:
        index = 0
        for i in range(len(data)):
            if data[i]['place'] == city:
                #race_winner.append(int(data[i]['results'][0]))
                #race_second.append(int(team[i]['results'][1]))

                day = data[i]['day']
                days.append(day)
                race_typ = data[i]['race_type']
                race_distance = data[i]['race_distance']
            
                
                horses = data[i]['horses']
                for k in range(len(horses)):
                    horses[k]['day'] = day
                    horses[k]['race_type'] = race_typ
                    horses[k]['distance'] = race_distance
                    try:
                        horses[k]['horse_win_prob'] = horses[k]['postion1'] / horses[k]['starts']
                    except:
                        horses[k]['horse_win_prob'] = 0.0
                    
                    race_horse = race_horse.append(horses[k], ignore_index=True)
                        
    except:
        print("er")

    for i, row in race_horse.iterrows():
        try:        
            race_horse.at[i, 'probable'] = 1 / row['probable']
        except ZeroDivisionError:
            race_horse.at[i, 'probable'] = 0.0000

    return {"horses": race_horse, "days": list(dict.fromkeys(days)) }


#make_h = make_horses('Mikkeli')
#print(make_h['horses'])
