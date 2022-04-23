import datetime
import numpy as np
from sklearn import covariance, cluster
import yfinance as yf

names = ["NVIDIA Corporation", "Netflix", "Plug Power Inc", "Apple", "Pfizer Inc", "Johnson & Johnson"]
symbols = ["NVDA", "NFLX", "PLUG", "AAPL", "PFE", "JNJ"]
openList = []
closeList = []

for symbol in symbols:
    info = yf.Ticker(symbol)
    start_date = datetime.datetime(2003, 7, 3)
    end_date = datetime.datetime(2007, 5, 4)
    quotes = info.history(start=start_date, end=end_date)
    openList.append(quotes.Open)
    closeList.append(quotes.Close)

opening_quotes = np.array(openList).astype(np.float_)
closing_quotes = np.array(closeList).astype(np.float_)

quotes_diff = closing_quotes - opening_quotes

X = quotes_diff.copy().T
X /= X.std(axis=0)

edge_model = covariance.GraphicalLassoCV()

with np.errstate(invalid='ignore'):
    edge_model.fit(X)

_, labels = cluster.affinity_propagation(edge_model.covariance_)
num_labels = labels.max()

for i in range(num_labels + 1):
    print("Cluster", i + 1, "==>", ", ".join(names[i]))
