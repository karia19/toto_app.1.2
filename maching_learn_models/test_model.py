import numpy as np
from tensorflow.keras.layers import Dense, Input, Flatten#, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import pandas as pd


data = pd.read_pickle("horses_for_demo.pkl")
print(data)


"""
# generate data
games, m = 50, 40
X = np.zeros((games, m, 11))
Y = np.zeros((games, m))

for i in range(games):
    X[i, :, 0] = i
    X[i, :, 1:] = np.random.rand(m, 10)  

    y_indexes = np.arange(m)
    np.random.shuffle(y_indexes)
    # score players
    Y[i,y_indexes[0]] = 2 # best
    Y[i,y_indexes[1]] = 1 # second best
    Y[i,y_indexes[2]] = 0 # third best
    Y[i,y_indexes[3:]] = -1 # not best

# run model   
print(X)
print(Y)
inputs = Input(shape=(m,10)) # -1 as we dont use fist column (game number)
inputs_flatten = Flatten()(inputs)
x = Dense(1024, activation='relu')(inputs_flatten)
x = Dense(512, activation='relu')(x)
x = Dense(256, activation='relu')(x)
outputs = Dense(m, activation=None)(x)
model = Model(inputs = inputs, outputs = outputs)

adam = Adam(lr=0.001)
model.compile(optimizer=adam, loss='mse', metrics=['accuracy'])
hist = model.fit(X[:,:,1:], Y, epochs=20, batch_size=10)

# predict third, second and best players for the first game
# the print number, is the player number
Y_pred = model.predict(X[0:1,:,1:])
print(np.argsort(Y_pred.reshape(-1))[-3:])
#[7 29 19]
# True best players fist game
print(np.argsort(Y[0,:].reshape(-1))[-3:])
#[7 29 19]
"""