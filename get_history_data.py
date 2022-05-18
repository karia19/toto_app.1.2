"""
import pandas as pd


workbook = pd.ExcelFile('toto.xlsx')
sheets = workbook.sheet_names
print(sheets)

sheets = ['Ohjastajat']
df = pd.concat([pd.read_excel(workbook, sheet_name=s)
                .assign(sheet_name=s) for s in sheets])

    
print(df[20:33])
"""
import requests
import pandas as pd
import datetime
import pickle
import json


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


        to_day_cardId = what['collection'][0]['cardId']
        race_place = what['collection'][0]['trackName']

        res = s.get("https://www.veikkaus.fi/api/toto-info/v1/card/"+ str(to_day_cardId) + "/races")
        res_j = res.json()
        res_le = res_j['collection']


        race_ID = []
        race_results = []
        race_type = []
        race_riders = []
        reverse_order = []
        race_distance = []

        for i in range(len(res_le)):
            race_ID.append(res_le[i]['raceId'])
            race_results.append(res_le[i]['toteResultString'].split("-"))
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

                    gender.append(race_horses_json_all[i]['gender'])
                    this_yaar_start.append(0.0)
                    #this_yaar_winMoney.append(race_horses_json_all[i]['stats']['currentYear']['winMoney'])
                    #this_yaar_1.append(race_horses_json_all[i]['stats']['currentYear']['position1'])
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

            #win_money_2d.append([this_yaar_winMoney])
            starts_2d.append([this_yaar_start])       
            #first_postion_2d.append([this_yaar_1]) 


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


        for i in pool_ids:
            res = s.get("https://www.veikkaus.fi/api/toto-info/v1/pool/"+ str(i) + "/odds")
            res_j = res.json()

            #print(start_index)
            #print(res_j)
            start_index += 1
            start_index_2 += 1

            net_sale = res_j['netSales'] / 100
            #print(net_sale)
            odds = res_j['odds']
            #print(odds)

            horses_for_json = []
            for k in range(len(odds)):
                try:
                    """
                    track_numebr.append(odds[k]['runnerNumber'])
                    ods_for_horse.append(odds[k]['probable'] / 100)
                    money_fro_horse.append(odds[k]['amount'] / 100)
                    #money_total.append(net_sale)
                    played_day.append(day)
                    start_number.append(start_index)
                    race_tyoe.append(race_type[start_index_2])
                    race_riders_arr.append(race_riders[start_index_2])
                    place.append(race_place)
                    race_res_2.append(race_results[start_index_2])
                    track_order.append(reverse_order[start_index_2])
                    track_distance.append(race_distance[start_index_2])

                    win_moneys.append(win_money_2d[start_index_2][0][k])

                    horses_name.append(horse_names_2d[start_index_2][0][k])
                    horses_age.append(horse_age_2d[start_index_2][0][k])
                    drivers.append(driver_name_2d[start_index_2][0][k])
                    genders.append(gender_2d[start_index_2][0][k])
                    #horses_money.append(win_money_2d[start_index_2][0][k])
                    starts.append(starts_2d[start_index_2][0][k])
                    first_places.append(first_postion_2d[start_index_2][0][k])
                    """
                    horses_for_json.append({"track": odds[k]['runnerNumber'],
                               "start_num": start_index,
                               "name": horse_names_2d[start_index_2][0][k],
                               "age": horse_age_2d[start_index_2][0][k],
                               "starts": starts_2d[start_index_2][0][k],
                               #"postion1": first_postion_2d[start_index_2][0][k],
                               "driver": driver_name_2d[start_index_2][0][k],
                               #"win_money": win_money_2d[start_index_2][0][k],
                               "gender": gender_2d[start_index_2][0][k],
                               "probable": odds[k]['probable'] / 100,
                               "amount": odds[k]['amount'] / 100,

                              
                               "front_shoes": horse_front_shoes_2d[start_index_2][k],
                               "rear_shoes": horse_rear_shoes_2d[start_index_2][k],
                               "coach": horse_coach_2d[start_index_2][k],
                               "home_town": horse_city_2d[start_index_2][k]                               
                               
                                })

                except:
                    print("err from array append")
                    
                

                    try:
                        track_numebr.pop()
                    
                        #horse_names_2d[start_index_2][0].pop(k)

                    except:
                        print("pop not need")

            all_in_one_json.append({"day": day, 'place': race_place, "start_num": start_index, "results": race_results[start_index_2],
                            "reverse_order": reverse_order[start_index_2], "race_type": race_type[start_index_2],
                            "race_distance": race_distance[start_index_2],
                            "horses": horses_for_json })



        
        #print(all_in_one_json)

       
        """    
        df['day'] = played_day
        df['place'] = race_place
        df['name'] = horses_name
        df['age'] = horses_age
        df['starts'] = starts
        df['position1'] = first_places
        df['driver'] = drivers
    
        df['win_money'] = win_moneys
        df['gender'] = genders
        df['start_num'] = start_number
        df['race_distance'] = track_distance
        df['race_type'] = race_tyoe
        df['race_riders'] = race_riders_arr
        df['track_num'] = track_numebr 
        df['ods_horse'] = ods_for_horse
        df['played_money'] = money_fro_horse
        df['results'] = race_res_2
        df['reverse_order'] = track_order


        #print(df)
        """   
    except:
        print("err in data") 

   

    #return df
    return all_in_one_json
        

if __name__ == "__main__":

    
    today = datetime.datetime.now()
    last_years = today - datetime.timedelta(days=299)

    days = []
    
    for n in range(1, 299):
        arrive = last_years + datetime.timedelta(days=n)
        days.append(arrive.strftime('%Y-%m-%d'))

    for day in days:
       day = get_horses(day)
       

    
    with open('toto_sort_2019.json', 'w') as f:
        json.dump(all_in_one_json, f)
    
    


  
    """
    today = datetime.datetime.now()
    days = []
    for n in range(1, 200):
        arrive = today - datetime.timedelta(days=n)
        days.append(arrive.strftime('%Y-%m-%d'))

    get_horses("2022-04-29")
    print(days)

    all_data = []
    for day in days:
       day = get_horses(day)
       all_data.append(day)

    df = pd.concat(all_data, sort=False)
    print(df)

    df.to_pickle('horse_track.pkl')
    """