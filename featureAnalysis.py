import csv
import numpy as np
import scipy
from sklearn import linear_model
from datetime import datetime
from dateutil.parser import parse
import matplotlib.pyplot as plt
from helperFunctions import printRSquared
from helperFunctions import predictNext

with open('data.csv', newline='') as csvfile:
    data = list(csv.reader(csvfile))


features=list(zip(*data))

#Values of the S&P 500 Index from 1870 for every month
labels = features[1][1:]
#Dates: one for each month since 1870
dates = features[0][1:]
#CPI
cpi = features[4][1:]
#Long interest rate
lir = features[5][1:]
#Business Cycle Stage
bcs = features[9][1:]
#Election Years and Quarter number
elections = []
quarter = []

dataPoints = len(labels)

for a in range(len(dates)):
	year = int(parse(dates[a]).year)
	month = int(parse(dates[a]).month)
	if year % 4 == 0:
		elections.append(1)
	else:
		elections.append(0)
	
	if(month < 4):
		quarter.append(1)
	elif(month < 7):
		quarter.append(2)
	elif(month < 10):
		quarter.append(3)
	else:
		quarter.append(4)

x = []
xCpi = []
xYear = []
xBcs = []
xElection = []
xQuarter = []
y = []

for a in range(dataPoints):
	x.append([int(parse(dates[a]).year), quarter[a], elections[a]])
	xCpi.append([float(cpi[a])])
	xYear.append([int(parse(dates[a]).year)])
	xBcs.append([int(bcs[a])])
	xElection.append([elections[a]])
	xQuarter.append([quarter[a]])
	y.append(float(labels[a]))
print("Elements in X: CPI, Year, Quarter, Elections, Business Cycle\n")

goodFeatureList = []

#Options for SVR model types are 'linear', 'poly', 'rbf'
printRSquared(x, y, "X", goodFeatureList)
printRSquared(xCpi, y, "CPI", goodFeatureList)
printRSquared(xYear, y, "Year", goodFeatureList)
printRSquared(xBcs, y, "Business Cycle", goodFeatureList)
printRSquared(xElection, y, "Election Year", goodFeatureList)
printRSquared(xQuarter, y, "Quarter", goodFeatureList)
print("")

print("Strong Positive influence: " + str(goodFeatureList))

yearsToPredict = []
yearsToPredict.append(['2020'])
yearsToPredict.append(['2021'])


predictNext(xYear, y, "Year", yearsToPredict)

print(xYear[dataPoints - 1])



"""
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
"""

