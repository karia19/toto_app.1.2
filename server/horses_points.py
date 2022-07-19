#from cgi import print_directory
#from matplotlib.pyplot import get
import pandas as pd
import numpy as np
#from statsmodels.tsa.seasonal import seasonal_decompose
#import matplotlib.pyplot as plt  



df = pd.read_pickle("horses.pkl")
print(df)

points_dict = {}
points_dict['Vermo'] = 20
points_dict['Jokimaa'] = 18
points_dict['Teivo'] = 18
points_dict['Forssa'] = 16
points_dict['Turku'] = 16
points_dict['Mikkeli'] = 12

place = ['Vermo', 'Jokimaa', 'Teivo', 'Forssa', 'Turku', 'Mikkeli']
fitst = [20, 18, 18, 16, 16, 12]
second = [18, 16, 16, 14, 14, 10]
tree = [16, 14, 14, 12, 12, 8]
four = [14, 12 , 12, 10, 10, 6]
five = [12, 10, 10, 8, 8, 4]
six = [10, 8,8, 4, 4, 2]
df_raiting = pd.DataFrame({"city": place, "1": fitst, "2": second, "3": tree, "4": four, "5": five, "6": six })
#print(df_raiting)

#print(points_dict)

def get_array(data):
    unique = []
    for number in data:
        if number in unique:
            continue
        else:
            unique.append(number)
    return unique

def find_horse(name):
    
    horse_raiting = []

    horse_df = df.query("name == @name")
    #print(horse_df['race_city'])
    #print(horse_df)
    horse_df.set_index('day', inplace=True)

    analysis = horse_df[['run_time']].copy()
    
    #print(horse_df[['race_city', 'position', 'track', 'probable' , 'race_type']])

    for index, row in horse_df.iterrows():
        if row['position'] < 6 and row['position'] == 1:
            city = row['race_city']
        
            try:
                points = df_raiting.query("city == @city")
                if len(points) == 0:
                    horse_raiting.append(8)
                else:
                    horse_raiting.append(int(points['1']))
            except:
                print("no in city")
           
        
        
        elif  row['position'] < 6 and row['position'] == 2:
            city = row['race_city']
            try:
                points = df_raiting.query("city == @city")
                if len(points) == 0:
                    horse_raiting.append(7)
                else:
                    horse_raiting.append(int(points['2']))
            except:
                print("no in city")

        
        elif  row['position'] < 6 and row['position'] == 3:
            city = row['race_city']
            try:
                points = df_raiting.query("city == @city")
                if len(points) == 0:
                    horse_raiting.append(6)
                else:
                    horse_raiting.append(int(points['3']))
            except:
                print("no in city")

        
        elif  row['position'] < 6 and row['position'] == 4:
            city = row['race_city']
            try:
                points = df_raiting.query("city == @city")
                if len(points) == 0:
                    horse_raiting.append(5)
                else:
                    horse_raiting.append(int(points['4']))
            except:
                print("no in city")

        
        elif  row['position'] < 6 and row['position'] == 5:
            city = row['race_city']
            try:
                points = df_raiting.query("city == @city")
                if len(points) == 0:
                    horse_raiting.append(4)
                else:
                    horse_raiting.append(int(points['5']))
            except:
                print("no in city")

        
        elif  row['position'] < 6 and row['position'] == 6:
            city = row['race_city']
            try:
                points = df_raiting.query("city == @city")
                if len(points) == 0:
                    horse_raiting.append(2)
                else:
                    horse_raiting.append(int(points['6']))
            except:
                print("no in city")
        
        else:
            horse_raiting.append(-2)
    
    #print("horse_raiting", horse_raiting)
    #print("last rate", np.sum(horse_raiting))

    return np.sum(horse_raiting)

    """
    decompose_result_mult = seasonal_decompose(analysis, period=2)

    trend = decompose_result_mult.trend
    seasonal = decompose_result_mult.seasonal
    residual = decompose_result_mult.resid

    decompose_result_mult.plot()
    plt.show();
    """

    """
    plt.plot(horse_raiting)
    plt.show()
    """



def make_rates(horses_df):
    #ace_city = helper_to_load.check_races_for_city()
    #race_data = helper_to_load.make_horses(race_city)
    #start_nums = get_array(list(race_data['horses']['start_num']))
    #horses_df = race_data['horses']

    for index, row in horses_df.iterrows():
        res_points = find_horse(row['name'])
        horses_df.at[index, 'points'] = res_points

    return horses_df    
   


#make_rates()
