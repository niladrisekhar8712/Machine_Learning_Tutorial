import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#  Logistic Regression
from sklearn.linear_model import LogisticRegression

logisticRegression = LogisticRegression()  # 98.54
logisticRegression.fit(X_train, y_train)
y_pred = logisticRegression.predict(X_test)
print('Logistic Regression')
print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test, y_pred))

# K Nearest Neighbor
from sklearn.neighbors import KNeighborsClassifier

kNN = KNeighborsClassifier(n_neighbors=10)  # 99.27
kNN.fit(X_train, y_train)
y_pred = kNN.predict(X_test)
print()
print('K Nearest Neighbors (K = 10)')
print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test, y_pred))

# Naive Bayes
from sklearn.naive_bayes import GaussianNB

naiveBayes = GaussianNB()  # 100.00
naiveBayes.fit(X_train, y_train)
y_pred = naiveBayes.predict(X_test)
print()
print('Naive Bayes')
print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test, y_pred))

# Linear SVM
from sklearn.svm import SVC

svc = SVC(kernel='linear')  # 98.54
svc.fit(X_train, y_train)
y_pred = svc.predict(X_test)
print()
print('Linear SVM')
print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test, y_pred))

# Kernel SVM
svck = SVC(kernel='rbf')  # 100.00
svck.fit(X_train, y_train)
y_pred = svck.predict(X_test)
print()
print('Kernel SVM')
print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test, y_pred))

# Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier

decisionTreeClassifier = DecisionTreeClassifier(criterion='entropy')  # 95.62
decisionTreeClassifier.fit(X_train, y_train)
y_pred = decisionTreeClassifier.predict(X_test)
print()
print('Decision Tree Classifier')
print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test, y_pred))

# Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier

randomForestClassifier = RandomForestClassifier(n_estimators=100, criterion='entropy')  # 100.00
randomForestClassifier.fit(X_train, y_train)
y_pred = randomForestClassifier.predict(X_test)
print()
print('Random Forest Classifier')
print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test, y_pred))