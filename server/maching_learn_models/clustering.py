import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_classification
import numpy as np

training_data, _ = make_classification(
    n_samples=1000,
    n_features=2,
    n_informative=2,
    n_redundant=0,
    n_clusters_per_class=1,
    random_state=4
)
print(training_data)


def plot_scatter(data):
    """
    wins = 1
    two = 2
    winners = data.query("winner == @wins")
    secons =  data.query("winner == @two")
    
    ax1 = winners.plot(
        x = "probable_last",
        y = "probable",
        kind = "scatter"
    )
    ax2 =  secons.plot(
        x = "probable_last",
        y = "probable",
        kind = "scatter",
        color = "r"
    )
    print(ax1 == ax2)
    """
    data = data[data.probable_last != 0.000]
    x = data.loc[:, ['probable_last', 'probable']].values
    print(x)
    dbscan = DBSCAN(eps = 0.081, min_samples = 25) # fitting the model
    pred = dbscan.fit_predict(x)
    anom_index = np.where(pred == -1)
    values = x[anom_index]

    plt.scatter(x[:,0], x[:,1])
    plt.scatter(values[:,0], values[:,1], color='r')
    plt.show()

    """
    print(data)
    fig = px.scatter(data, x="probable_last", y="probable", color="position",
                 symbol="position", marginal_x="histogram", marginal_y="rug",
                 size='probable'
                 )
    fig.show()
    plt.show()
    """