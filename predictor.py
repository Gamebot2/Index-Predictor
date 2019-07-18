import csv
import numpy as np
import scipy
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.svm import SVR
from datetime import datetime
from dateutil.parser import parse
import matplotlib.pyplot as plt

with open('data.csv', newline='') as csvfile:
    data = list(csv.reader(csvfile))

for i in range(1):
    print(str(i) + ": " + str(data[i]))

features=list(zip(*data))

#Values of the S&P 500 Index from 1870 for every month
labels = features[1][1:]
#Dates: one for each month since 1870
dates = features[0][1:]
#CPI
cpi = features[4][1:]
#Long interest rate
lir = features[5][1:]
#Election Years
elections = []

dataPoints = len(labels)

for a in range(len(dates)):
	year = int(parse(dates[a]).year)
	if year % 4 == 0:
		elections.append(1)
	else:
		elections.append(0)

x = []
xCpi = []
xYear = []
y = []


for a in range(dataPoints):
	x.append([float(cpi[a]), int(parse(dates[a]).year), float(lir[a])])
	xCpi.append([float(cpi[a])])
	xYear.append([int(parse(dates[a]).year)])
	y.append(float(labels[a]))

for i in range(5):
    print(str(i) + ": " + str(x[i]))

xTrain, xTest, yTrain, yTest = train_test_split(xCpi, y, test_size=.2, random_state=42)

#Options for model types are 'linear', 'poly', 'rbf'
clf = SVR(kernel='poly', gamma='scale', C=1.0, epsilon=0.1)
clf.fit(xTrain, yTrain)

#testPrediction = clf.predict(xTest)
print("Value of R^2: " + str(clf.score(xTest, yTest)))

#####ATTEMPT AT GRAPHING
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5), sharey=False)
axes[0].plot(xCpi, clf.fit(xCpi, y).predict(xCpi), color='m', lw=2, label='Test')
axes[0].scatter(xCpi, y, facecolor="none", edgecolor='b', s=50, label='Testing 2')
#axes[1].plot(xYear, clf.fit(xYear, y).predict(xYear), color='m', lw=2, label='Test')
axes[1].scatter(xYear, y, facecolor="none", edgecolor='b', s=50, label='Testing 2')

fig.text(0.3, 0.04, 'Consumer Price Index', ha='center', va='center')
fig.text(0.7, 0.04, 'Year', ha='center', va='center')
fig.text(0.06, 0.5, 'S&P500 Value', ha='center', va='center', rotation='vertical')
fig.suptitle("Support Vector Regression", fontsize=14)
plt.show()

