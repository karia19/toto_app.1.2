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





df = pd.read_csv('./vermo.csv', index_col=False, usecols=['Day','arvo1', 'arvo2','arvo3', 'Rata_nro', 'Voitto', 'Luku'])
df = df.dropna()
#X = df[['arvo1', 'arvo2', 'arvo3']].str.replace(",""").astype(float)
df['Voitto'] = df.Voitto.replace(" ", "0,00", regex=True)
df['Day'] = pd.to_datetime(df.Day)
print(df)


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
  

   X = data[['a1', 'a2','a3', 'rata', 'Luku', 'ols']]
   y = data['Voitto']

   new_arr = []
   for i, tt in data.iterrows():
       new_arr.append([tt['a1'], tt['a3'], tt['Luku'], tt['rata']])

   X_train,X_test,y_train,y_test = train_test_split(new_arr, np.array(y), test_size = 0.20 ,random_state = 100 )

   clf = svm.SVC()
   clf.fit(X_train, y_train)
   score = clf.score(X_train, y_train)
   print("SCM", score)
   ypred = clf.predict(X_test)
   cm = classification_report(y_test, ypred)
   print(ypred, y_test)
   print(cm)

   
   

   
   logit_model = sm.OLS (y , X)
   result = logit_model.fit() 
   print(result.summary())
   
   """
   X_train,X_test,y_train,y_test = train_test_split(X, np.array(y), test_size = 0.20 ,random_state = 50 )
   clf = LogisticRegression(random_state=0, solver='lbfgs').fit(X_train, y_train)
   print("LogasticRegression", clf.score(X_test, y_test))
   """   

model_variables = ['a3', 'Luku' ]

def make_ols_plot(data):
    """
    shotcount_dist=np.histogram(df['Angle'],bins=100)
    goalcount_dist=np.histogram(ddGoals['Angle'],bins=100)
    prob_goal=np.divide(goalcount_dist[0],shotcount_dist[0])
    print(prob_goal)

    angle=shotcount_dist[1]
    midangle= (angle[:-1] + angle[1:])/2
    """
    value_one = data['a2']

    model = ''
    for v in model_variables[: -1]:
       model = model + v + '+'
    model = model + model_variables[-1]     
    
    test_model = smf.glm(formula="Voitto ~" + model , data=data, 
                           family=sm.families.Binomial()).fit()
    print(test_model.summary())        
    b=test_model.params
    #b.to_pickle('./new.pkl')
  
    xG = data.apply(calculate_xG, axis=1) 
    data = data.assign(ols=xG)

   
    """
    fig,ax=plt.subplots(num=1)
    ax.plot(value_one / 100, value_one / 100, linestyle='none', marker= '.', markerSize= 12, color='black')
    ax.plot(xGprob, xGprob, linestyle='solid', color='red')
    ax.set_ylabel('Probability chance scored')
    ax.set_xlabel("Shot angle (degrees)")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.show()

    xGprob=1/(1+np.exp(b[0]+b[1]*midangle)) 
    fig,ax=plt.subplots(num=1)
    ax.plot(midangle, prob_goal, linestyle='none', marker= '.', markerSize= 12, color='black')
    ax.plot(midangle, xGprob, linestyle='solid', color='black')
    ax.set_ylabel('Probability chance scored')
    ax.set_xlabel("Shot angle (degrees)")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.show()
    """

    return data

#dd = pd.read_pickle('./new.pkl')
#b = dd
#print("Mutta tämä", b)

def calculate_xG(sh):   
   bsum=b[0]
   
   for i,v in enumerate(model_variables):
      bsum= bsum + float(b[i+1]) * float(sh[v])
      xG = 1/(1+np.exp(bsum)) 
   return xG   
   

if __name__ == '__main__':
   clean_data = df[df['arvo3'] != "0,00"]
   clean_data_num = make_data_to_numbers(clean_data)
   print(clean_data)


   """
   result_from_ols = make_ols_plot(clean_data_num)
   #only_wins = result_from_ols[result_from_ols['Voitto'] == 1 ]
   print(result_from_ols)
   #plt.plot(only_wins['ols'])
   #plt.show()

   
   only_wins = result_from_ols[clean_data_num['Voitto'] == 1]
   only_loses = result_from_ols[clean_data_num['Voitto'] != 1]
   print(len(only_loses), len(only_wins))

   frames = [ only_wins, only_loses[:500]]
   new_mixed_data = pd.concat(frames, sort=False)
   #print(len(new_mixed_data))
   make_reg(new_mixed_data)
   """   

"""
x_ax = range(len(make_to[:80]))
plt.scatter(x_ax, make_to['a2'][:80], s=5, color="blue", label="original")
plt.scatter(x_ax, make_to['a3'][:80], s=5, color="green", label="original")
plt.legend()
plt.show()
"""






