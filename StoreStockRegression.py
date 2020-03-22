import pandas as pd
import quandl, math, datetime
import numpy as np
from sklearn import preprocessing, svm, model_selection
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')

df = quandl.get('WIKI/GOOGL')  # get the data

df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]  # remove unnecessary columns
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Close'] * 100

df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]  # rearrange the data frame
forecast_col = 'Adj. Close'
df.fillna(-99999, inplace=True)  # replace missing values with the specified value

forecast_out = math.ceil(0.01*len(df))
df['label'] = df[forecast_col].shift(-forecast_out)  # shift the rows up

X = np.array(df.drop(['label'], 1))  # features - every column except of 'label'
X = preprocessing.scale(X)  # center to the mean
X_lately = X[-forecast_out:]   # last rows
X = X[:-forecast_out]  # delete rows that have NAN value in the 'label' column

df.dropna(inplace=True)  # remove missing values
y = np.array(df['label'])  # labels

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, train_size=0.2)

clf = LinearRegression()
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)

forecast_set = clf.predict(X_lately)
df['Forcast'] = np.nan
last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400  # seconds per day
next_unix = last_unix + one_day  # next day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]

df['Adj. Close'].plot()
df['Forcast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()