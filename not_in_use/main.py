import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import confusion_matrix
import pickle
import joblib
from scipy import stats
from sklearn.metrics import classification_report





#print(df)


def make_data_to_numbers(data):
   a0 = data['Day']
   a1 = data['arvo1'].str.replace(",","").astype(float).to_numpy()
   a2 = data['arvo2'].str.replace(",","").astype(float).to_numpy()
   a3 = data['arvo3'].str.replace(",","").astype(float).to_numpy()
   a4 = data['Rata_nro'].to_numpy()
   #a5 = data['Lähto'].to_numpy()
   a6 = data['Luku'].astype(float).to_numpy()
   y =  data['Voitto'].str.replace(",","").astype(float).to_numpy()

   allTo = {"Day": a0, "a1":a1, "a2": a2, "a3": a3, "rata": a4, "Voitto":y, "Luku": a6 }
   df = pd.DataFrame(allTo, columns=["Day", "a1","a2","a3", "rata", "Voitto", "Luku" ])

   return df



def make_reg(data):
  

   X = data[['rata', 'a1','a2', 'a3', 'Luku']]
   y = data['Voitto']

   print(X.corrwith(y, method='pearson'))
   print(X.corrwith(y, method='kendall'))
   print(X.corrwith(y, method='spearman'))





if __name__ == '__main__':

   df = pd.read_csv('./vermo.csv', index_col=False, usecols=['Day','arvo1', 'arvo2','arvo3', 'Rata_nro', 'Voitto', 'Luku'])
   df = df.dropna()
   #X = df[['arvo1', 'arvo2', 'arvo3']].str.replace(",""").astype(float)
   df['Voitto'] = df.Voitto.replace(" ", "0,00", regex=True)
   df['Day'] = pd.to_datetime(df.Day)

   clean_data = df[df['arvo3'] != "0,00"]
   clean_data_num = make_data_to_numbers(clean_data)
   print("what", clean_data_num)
   make_reg(clean_data_num)






