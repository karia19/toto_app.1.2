
from matplotlib.pyplot import get
import pandas as pd
import json
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
import statsmodels.api as sm
import joblib
import pickle
from datetime import datetime
import plotly.express as px
from sklearn.model_selection import cross_val_score, KFold
from sklearn.model_selection import RepeatedKFold
from numpy import absolute
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import xgboost as xgb



array_len = 144

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



def make_horses_to_2d(data, days):
   
    test_ar = []
    all_in_one = []
   
   

    for index, row in days.iterrows():
        dd = row['day']
        star = row['starts']
        df_res = data.query("day == @dd and start_num == @star")
        test_ar = []

        for index, row in df_res.iterrows():     
                test_ar.extend([            #row['track'] ,
                                            row['run_time'],  
                                            row['probable'], 
                                            row['amount'], 
                                            horse_gender(row['gender']),
                                            #race_type(row['race_type']),
                                            #row['age'],
                                            #hash_shoes(row['front_shoes']),
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
                                            row['d_w_pr_s']

                                           
                                            
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


if __name__ == "__main__":

    city = "Seinäjoki"
    
    
    df_all = pd.read_pickle("horses.pkl")

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


    #df_all['last_proba'] = 1 / df_all['last_proba']

    
    coach_names = get_array(list(df_all['coach']))
    le_coach = LabelEncoder()
    le_coach.fit(coach_names)
    df_all['coach_l'] = le_coach.fit_transform(df_all['coach'])

    race_city = get_array(list(df_all['race_city']))
    le_city = LabelEncoder()
    le_city.fit(race_city)
    df_all['city_l'] = le_city.fit_transform(df_all['race_city'])
    
    
    df_all['gender_new'] = list(map(horse_gender , list(df_all['gender'])))
    df_all['front_new'] = list(map(hash_shoes, list(df_all['front_shoes'])))
    df_all['race_new'] = list(map(race_type, list(df_all['race_type'])))

    df_all['last_run'] =  list((df_all['last_position']).astype(float))


    res_search = serach_city(df_all, city)
    df = res_search['horses']
    
    
    print(df)
   

    cor_x = df[['d_w_pr_s', 'probable_last', 'probable', 'city_l', 'last_run', 'rest_days', 'coach_l', 'race_new', 'front_new', 'start_num', 'horse_starts', 'gender_new', 'run_time', 'track', 'driver_l', 'horse_l','amount', 'driver_starts', 'age', 'horse_win_prob', 'horse_money']]
    y_cor = df['winner']
    print(cor_x.corrwith(y_cor, method='pearson'))
    
    #fig = px.imshow(cor_x.corr())
    #fig.show()
   
    

    df['prob_small'] = df['probable'] / 100
    center_function = lambda x: x - x.mean()
    df['amount_cen'] = center_function(np.array(df['amount']))

    x = df[['rest_days', 'track', 'age', 'd_w_pr_s', 'd_w_pr', 'probable_last','probable', 'gender_new', 'amount', 'horse_win_prob', 'horse_money']]
    y = df['winner'].to_numpy(dtype="float")
    
  
   
    # adding the constant term
    x = sm.add_constant(x)
 
    # performing the regression
    # and fitting the model
    result = sm.OLS(y, x).fit()
 
    # printing the summary table
    print(result.summary())

    days_len = len(res_search['days'])
    df2 = make_horses_to_2d(df, res_search['all']) #.reshape(-1,1)
    df2['winner'] = res_search['winners']
    print(df2)

    
    col_len = []
    for i in range(array_len):
        col_len.append(i)

    X = df2[col_len].to_numpy(dtype="float")
    y = df2['winner'].to_numpy(dtype="int")

  

    #scaler = StandardScaler()
    #X = scaler.fit_transform(X)
    
    X_train,X_test,y_train,y_test = train_test_split(X, np.array(y), test_size = 0.12 ,random_state = 0)

    clf = LogisticRegression(solver='saga',max_iter=1200).fit(X_train, y_train)
    print("LogasticRegression", clf.score(X_test, y_test))

    clf_boost = GradientBoostingClassifier(learning_rate=0.2, max_depth=1, random_state=0).fit(X_train, y_train)
    print("GradientBoost", clf_boost.score(X_test, y_test))

    ### XBOOST CLASS ###
    model = xgb.XGBRFClassifier()
    model.fit(X_train, y_train)
    print("XGBC_class ", model.score(X_test, y_test))


    ### SAVE MODELS ###
    joblib.dump(clf_boost, "gradientBoost_"+ city +".pkl")
    joblib.dump(clf, "logasticRegression_" + city + ".pkl")
    model.save_model("XGBC_" + city + ".txt")
    
    #### MAKE PREDICTIONS ####
    xgbc_pred = model.predict(X_test)
    boost_pred =(clf_boost.predict(X_test))
    clf_pred = (clf.predict(X_test))

    xgbc = model.predict_proba(X_test)
    boost_res = clf_boost.predict_proba(X_test)
    clf_res = clf.predict_proba(X_test)


    collect_pred_res = []
    for i in range(len(boost_pred)):
        if xgbc_pred[i] == y_test[i] or  boost_pred[i] == y_test[i] or clf_pred[i] == y_test[i]:
            collect_pred_res.append(1)
        else:
            collect_pred_res.append(0)
    
    print("sum of preds",  np.sum(collect_pred_res) / len(collect_pred_res))

    
    for i in range(len(boost_res)):
        #print("start " + str(i +1), clf_res[i].tolist())
        #print("start " + str(i +1), np.sum(clf_res[i]))
        print(y_test[i])    

        print("XGBC class: " , xgbc[i].argsort()[::-1][:3] + 1)
        floats2 = clf_res[i].argsort()[::-1][:3] + 1
        print("start " + str(i +1) + " logas: ", floats2)
        
        floats3 = boost_res[i].argsort()[::-1][:3] + 1
        print("start " + str(i +1) + " boost: ", floats3)
    
    