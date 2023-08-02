import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

dataset = pd.read_csv('Real estate.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=1)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

print(np.concatenate((y_test.reshape(len(y_test),1), y_pred.reshape(len(y_pred),1)),1))
from sklearn.metrics import r2_score
print(r2_score(y_test,y_pred))