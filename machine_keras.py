from tabnanny import check
from unicodedata import name
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
import pickle
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn import tree
import joblib
from sklearn.linear_model import LogisticRegression
import sys
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import SGD




def check_predict(data):
     col_len = []
     for i in range(224):
        col_len.append(i)

     X = data[col_len].to_numpy(dtype="float")

     #sc = StandardScaler()
     #X = sc.fit_transform(X)
     
     y =  data['winner'].to_numpy(dtype="int")
    
     y = to_categorical(y)
  
     
     X_train,X_test,y_train,y_test = train_test_split(X, np.array(y), test_size = 0.20 ,random_state = 0 )
     print(len(X_test))
    
     ### Make Keras ###
     model = Sequential()
     """
     model.add(Dense(units=128,activation="relu",input_shape=(192,)))
     model.add(Dense(units=128,activation="relu"))
     model.add(Dense(units=128,activation="relu"))
     model.add(Dense(units=17,activation="softmax"))

     model.compile(optimizer=SGD(0.001),loss="categorical_crossentropy",metrics=["accuracy"])
     model.fit(X_train,y_train,batch_size=32,epochs=10,verbose=1)
     accuracy = model.evaluate(x=X_test,y=y_test,batch_size=32)
     print("Accuracy: ",accuracy[1])
     """   
        
     
     model.add(Dense(256, input_dim=224, activation='relu'))
     model.add(Dense(128,activation="relu"))
     model.add(Dense(64,activation="relu"))
     model.add(Dense(17, activation='softmax'))
     model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
     model.fit(X_train, y_train, epochs=100, batch_size=128, shuffle=True)

     accuracy = model.evaluate(x=X_test,y=y_test,batch_size=32)
     print("Accuracy: ",accuracy[1])

   
     """    
     plt.plot(history.history['acc'])
     plt.plot(history.history['loss'])
     plt.title('model accuracy')
     plt.ylabel('accuracy')
     plt.xlabel('epoch')
     plt.legend(['train', 'val'], loc='upper left')
     plt.show()
     """
     y_pred = model.predict(X_test)
     #Converting predictions to label
     pred = list()
     for i in range(len(y_pred)):
        pred.append(np.argmax(y_pred[i]))
     
     test = list()
     for i in range(len(y_test)):
         test.append(np.argmax(y_test[i]))
     

     a = accuracy_score(pred,test)
     print('Accuracy is:', a*100)
     #model.save("model.h5")
     
def get_horses_names(data):
    unique = []
    for number in data:
        if number in unique:
            continue
        else:
            unique.append(number)
    return unique






df = pd.read_pickle('toto_for_machine.pkl')
df_2 = pd.read_pickle("toto_all.pkl").sort_values(['day', 'start_num'])

names = get_horses_names(list(df_2['name']))
drivers = get_horses_names(list(df_2['driver']))

for horse in names:
   horse_races = df_2.query("name == @horse")
   #print(horse)
   for index, row in horse_races.iterrows():
      df_2.at[index, 'horse_starts'] = row['starts'] + 1.0

for d in drivers:
   driver_race = df_2.query("driver == @d")
   mem_starts = 1
   for index , row in driver_race.iterrows():
      df_2.at[index, "driver_starts"] = mem_starts
      mem_starts += 1


print(df_2)
#check_predict(df)
