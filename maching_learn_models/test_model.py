from pyexpat import model
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.layers import Dense, Input, Flatten#, Reshape,
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from sklearn import metrics
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout

import pandas as pd
from load_today_race import make_horses
array_len = 208


data = pd.read_pickle("/Users/kari/Desktop/toto_horse2/make_horse_stats/horses.pkl")

def serach_city(data, city):
    df_city = data.query("race_city == @city")
    wins = []
    days =  [] #list(df_city['day'])
    start_numer = []
    data_df = pd.DataFrame()

    for index, row in df_city.iterrows():
        if row['winner'] == 1:
           
            wins.append(int(row['track']))
            days.append(row['day'])
            start_numer.append(row['start_num'])
    
    data_df['day'] = days
    data_df['starts'] = start_numer
    data_df['win'] = wins

    return {"horses": df_city ,"winners": wins, "days": list(dict.fromkeys(days)), 'all': data_df }

def get_array(data):
    unique = []
    for number in data:
        if number in unique:
            continue
        else:
            unique.append(number)
    return unique


def horse_gender(gender):
    if gender == "TAMMA":
        return 1
    elif gender == "RUUNA":
        return 2
    else:
        return 3

def race_type(race):
    if race == "CAR_START":
        return 5
    else:
        return 6

def hash_shoes(shoes):
    if shoes == "HAS_SHOES":
        return 1
    else:
        return 0

def make_horses_to_2d(data, days):
   
    test_ar = []
    all_in_one = []
    results_arr = []
   
   

    for index, row in days.iterrows():
        dd = row['day']
        star = row['starts']
        df_res = data.query("day == @dd and start_num == @star")



        test_ar = []
        race_position = []

        
        for i in df_res['position']:
            if i == 1:
                race_position.append(1)
            elif i == 2:
                race_position.append(1)
            elif i == 3:
                race_position.append(0)
            else:
                race_position.append(0)

        array_len_index = len(race_position)
        
        
        for i in range(len(race_position), 16):
            race_position.append(0)

       
        
        
        for index, row in df_res.iterrows():  
               
                test_ar.append([           
                                            row['run_time'],  
                                            row['probable'], 
                                            row['amount'], 
                                            horse_gender(row['gender']),
                                            race_type(row['race_type']),
                                            row['age'],
                                            hash_shoes(row['front_shoes']),
                                            hash_shoes(row['rear_shoes']),

                                            #row['rest_days'],
                                            row['horse_money'],
                                            #row['horse_money'],
                                            #row['driver_l'],
                                            #row['distance'],
                                            #row['horse_starts'],
                                            #row['city_l'],
                                            row['start_num'],
                                            #row['last_run'],
                                            row['probable_last'], 
                                            row['driver_starts'],                           
                                            row['horse_win_prob'],
                                            row['d_w_pr_s'],
                                            row['h_w_S'],
                                            row['c_w_pr_s']
                                            
                                            ])

         
        for i in range(array_len_index, 16):
            test_ar.append([ 0.0 , 0.0 , 0.0 , 0.0, 0.0,  0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0, 0.0, 0.0, 0.0, 0.0 , 0.0 ])
        
        #print(test_ar)
        all_in_one.append(test_ar)
        results_arr.append(race_position)
    
    #print(results_arr)
    #print(all_in_one)

    return { "data": all_in_one, "results": results_arr}


def make_start_to_tensor(data):
    test_ar = []
    all_in_one = []
    results_arr = []

    drivers_last_starts = data.iloc[-1:]
    starters_len = int(drivers_last_starts['track'])
    for index, row in data.iterrows():  
               
                test_ar.append([           
                                            row['horse_run_time'],  
                                            row['probable'], 
                                            row['amount'], 
                                            horse_gender(row['gender']),
                                            race_type(row['race_type']),
                                            row['age'],
                                            hash_shoes(row['front_shoes']),
                                            hash_shoes(row['rear_shoes']),
                                            #row['rest_days'],
                                            row['win_money'],
                                            #row['horse_money'],
                                            #row['driver_l'],
                                            #row['distance'],
                                            #row['horse_starts'],
                                            #row['city_l'],
                                            row['start_num'],
                                            #row['last_run'],
                                            row['probable_last'], 
                                            row['driver_starts'],                           
                                            row['horse_win_prob'],
                                            row['d_w_pr'],
                                            row['h_w_S'],
                                            row['c_w_pr']
                                            
                                            ])

         
    for i in range(starters_len, 16 ):
        test_ar.append([ 0.0 , 0.0 , 0.0 , 0.0, 0.0,  0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0, 0.0, 0.0, 0.0, 0.0 ])
        
        #print(test_ar)
    all_in_one.append(test_ar)
        
    return all_in_one


def make_horses_to_2_tensor(data, days):
    test_ar = []
    all_in_one = []
   
   

    for index, row in days.iterrows():
        dd = row['day']
        star = row['starts']
        df_res = data.query("day == @dd and start_num == @star")
        test_ar = []

        for index, row in df_res.iterrows():     
                test_ar.extend([            row['track'] ,
                                            row['run_time'],  
                                            row['probable'], 
                                            row['amount'], 
                                            horse_gender(row['gender']),
                                            #race_type(row['race_type']),
                                            #row['age'],
                                            hash_shoes(row['front_shoes']),
                                            #row['rest_days'],
                                            row['horse_money'],
                                            #row['horse_money'],
                                            #row['driver_l'],
                                            #row['distance'],
                                            #row['horse_starts'],
                                            #row['city_l'],
                                            #row['start_num'],
                                            #row['last_run'],
                                            row['probable_last'], 
                                            row['driver_starts'],                           
                                            row['horse_win_prob'],
                                            row['d_w_pr_s'],
                                            row['h_w_S'],
                                            row['c_w_pr_s']

                                           
                                            
                                            ])
            
            
        if len(test_ar) != 0:
                #print(len(test_ar))
                #same_len = 189 - len(test_ar)
                for i in range(len(test_ar), array_len):
                    #if test_ar[i] != float:
                        test_ar.append(0.0)

                #print("what" , len(test_ar))
                all_in_one.append(test_ar)
          
    #print(all_in_one)

    col_len = []
    for i in range(array_len):
        col_len.append(i)
    
    df = pd.DataFrame(all_in_one, columns=col_len)
    
    return df
   
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
            print(d_win_prob)

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

def make_today_race_2d(data, days):
    test_ar = []
    all_in_one = []
    print(len(days))
    starts_num = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]

   

    for d in days:
        
        for start in starts_num:
          
            df_res = data.query("day == @d and start_num == @start")
            test_ar = []
            
            for index, row in df_res.iterrows():     
                test_ar.extend([            row['track'] ,
                                            row['horse_run_time'],  
                                            row['probable'], 
                                            row['amount'], 
                                            horse_gender(row['gender']),
                                            #race_type(row['race_type']),
                                            #row['age']
                                            hash_shoes(row['front_shoes']),
                                            #row['rest_days'],
                                            row['win_money'],
                                            #start,
                                            #row['driver_l'],
                                            #row['distance'],
                                            #row['horse_starts'],
                                            #row['city_l'],
                                            #row['last_run'],
                                            row['probable_last'], 
                                            row['driver_starts'],                           
                                            row['horse_win_prob'],
                                            row['d_w_pr'],
                                            row['h_w_S'],
                                            row['c_w_pr']
                                            

                                            
                                            ])
            
            
            if len(test_ar) != 0:
                #print(len(test_ar))
                #same_len = 189 - len(test_ar)
                for i in range(len(test_ar), array_len):
                    #if test_ar[i] != float:
                        test_ar.append(0.0)

                #print("what" , len(test_ar))
                all_in_one.append(test_ar)
          
    #print(all_in_one)

    col_len = []
    for i in range(array_len):
        col_len.append(i)
    
    df = pd.DataFrame(all_in_one, columns=col_len)
    
    return df


if __name__ == "__main__":


    city = "Jokimaa"


    res = serach_city(data, city)
    df_all = res['horses']
    for i, row in df_all.iterrows():
            try:        
                df_all.at[i, 'probable_last'] = 1 / row['last_proba']
            except ZeroDivisionError:
                df_all.at[i, 'probable_last'] = 0.0000

    for i, row in df_all.iterrows():
            try:        
                df_all.at[i, 'probable'] = 1 / row['probable']
            except ZeroDivisionError:
                df_all.at[i, 'probable'] = 0.0000

    #print(res['all'])
    df_all['h_w_S'] = df_all['h_w_s'].astype(float)
    df_all['gender_new'] = list(map(horse_gender , list(df_all['gender'])))
    df_all['front_new'] = list(map(hash_shoes, list(df_all['front_shoes'])))
    df_all['race_new'] = list(map(race_type, list(df_all['race_type'])))
    print(df_all)


    #df_winners.plot( x = "d_w_pr", y = 'probable', kind="scatter")
    #plt.show()

    """MAKE HORSES FOR START 2D"""

    #res_horse = make_horses_to_2d(df_all, res['all'])

    """MAKE HORSES INDIVIDUAL"""
    """
    X = df_all[['probable', 'gender_new', 'amount', 'track', 'h_w_S' , 'race_new', 'front_new', 'probable_last', 'driver_starts', 'horse_win_prob', 'd_w_pr', 'c_w_pr']].values
  
    
    y = []
    for index, row in df_all.iterrows():
        if row['position'] == 1:
            y.append(1)
        elif row['position'] == 2:
            y.append(2)
        elif row['position'] == 3:
            y.append(3)
        else:
            y.append(0)

    print(len(X))
    
    y = df_all['position']
    

    encoder = LabelEncoder()
    encoder.fit(y)
    encoded_Y = encoder.transform(y)
    # convert integers to dummy variables (i.e. one hot encoded)
    dummy_y = np_utils.to_categorical(encoded_Y)
    print(dummy_y)
    # define baseline model
    X_train,X_test,y_train,y_test = train_test_split(X, dummy_y, test_size = 0.20 ,random_state = 0)

    # create model
    model = Sequential()
    model.add(Dense(32, input_dim=12, activation='relu'))
    model.add(Dense(16, activation='softmax'))
        # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


    print(model.summary())	
    score, acc = model.evaluate(X_train, y_train, batch_size=128, verbose=2)
    print("accuracy", acc)
    history = model.fit(X_train, y_train, batch_size=128, epochs=30, verbose=2)
    
    pred_res = model.predict(X_test)
    
    matrix = metrics.confusion_matrix(y_test.argmax(axis=1), pred_res.argmax(axis=1))    
    matrix_p = metrics.ConfusionMatrixDisplay(matrix)
    matrix_p.plot()
    plt.show()
    """

    """MAKE HORSES FOR START 2D"""

    #res_horse = make_horses_to_2d(df_all, res['all'])

    """MAKE TENSORFLOWW MODEL AND DATA"""
    
    """
    horse_2d = np.array(res_horse['data']).astype(float)
    res_2d = np.array(res_horse['results']).astype(float)

    train_x = np.asarray(horse_2d)
    #train_y = np.asarray(train_y)
    validation_x = np.asarray(res_2d)
    #validation_y = np.asarray(validation_y)


    m = 16
    y_len = 8 #len(res_horse['data'][0])

    inputs = Input(shape=(m, 16)) # -1 as we dont use fist column (game number)
    inputs_flatten = Flatten()(inputs)
    x = Dense(1024, activation='relu')(inputs_flatten)
    x = Dense(512, activation='relu')(x)
    x = Dense(256, activation='relu')(x)
    outputs = Dense(m, activation=None)(x)
    model = Model(inputs = inputs, outputs = outputs)

    adam = Adam(learning_rate=0.001)
    model.compile(optimizer=adam, loss='mse', metrics=['accuracy'])
    hist = model.fit(train_x, validation_x, epochs=30, batch_size=20)
    """
    
    """ TENSORFLOW BEST IN SO FAR; FEED DATA HAS BEEN MODIFY"""
    
    df2 = make_horses_to_2_tensor(df_all, res['all'])
    print(df2)
    
    df2['winner'] = res['winners']
    
    win = 1
    two = 2
    tree = 3
    four = 4
    five  = 5
    df2 = df2.query("winner == @win or winner == @two or winner == @tree or winner == @four or winner == @five ")
  
    col_len = []
    for i in range(array_len):
        col_len.append(i)

    X = df2[col_len].to_numpy(dtype="float")
    y = df2['winner'].to_numpy(dtype="int")
    y_tf = to_categorical(y)

    X_train_tf,X_test_tf,y_train_tf,y_test_tf = train_test_split(X, y_tf, test_size = 0.05 ,random_state = 0)
    model_tf = Sequential([
        Dense(252, activation='relu', input_shape=(array_len,)),
        Dense(126, activation='tanh'),
        Dropout(rate=0.05),
        Dense(64, activation='relu'),
        Dense(6, activation='softmax')
    ])
    model_tf.compile(
        optimizer='Adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    
    model_tf.fit(X_train_tf, y_train_tf, epochs=20, batch_size=64, shuffle=False)
    pred_res = model_tf.predict(X_test_tf)
    
    #matrix = metrics.confusion_matrix(y_test_tf.argmax(axis=1), pred_res.argmax(axis=1))    
    #matrix_p = metrics.ConfusionMatrixDisplay(matrix)
    #matrix_p.plot()
    #plt.show()
    




    """LOAD TODAY RACE AND MAKE IT TO TENSORFLOW"""
    
    
    today_race = make_horses(city)
    
    df = today_race['horses']
    df['gender_new'] = list(map(horse_gender , list(df['gender'])))
    df['front_new'] = list(map(hash_shoes, list(df['front_shoes'])))
    df['race_new'] = list(map(race_type, list(df['race_type'])))
        
    drives = get_array(list(df['driver']))
    horses = get_array(list(df['name']))
    coaches = get_array(list(df['coach']))
        ### COLLETCT TO DAY HORSES####
        

    df2 = set_drivers_history(data, df, drives)
    df3 = set_horse_history(data, df2, horses)

    for i, row in df3.iterrows():
            try:        
                df3.at[i, 'probable_last'] = 1 / row['last_proba']
            except ZeroDivisionError:
                df3.at[i, 'probable_last'] = 0.0000
        #df3 = set_coach_history(team, df2, coaches)

    df3['win_money'] = df3['win_money'] / 100
    print(df3)
    
    df2 = make_today_race_2d(df3, today_race['days'])
    print(df2)

    res = model_tf.predict(df2)
    print(res.argmax(axis=1))
    
    


    """
    last_starts = df.iloc[-1:]

    start_num =  int(last_starts['start_num']) + 1
    for i in range(1,start_num):
        try:
            start_one = df3.query("start_num == @i")
            print(start_one)
            res_ten = make_start_to_tensor(start_one)
            horse_2d = np.array(res_ten)
            train_x = np.asarray(horse_2d)
           
            ####MAKE PREDICT####
            #predict third, second and best players for the first game
            # the print number, is the player number
            Y_pred = model.predict(train_x)
            
            print(np.argsort(Y_pred.reshape(-1))[-3:])
        except:
            print("no race")
    
    """
