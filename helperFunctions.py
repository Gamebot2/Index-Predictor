from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score

def printRSquared(xSet, ySet, feature, list):
	xTrain, xTest, yTrain, yTest = train_test_split(xSet, ySet, test_size=.2, random_state=42)

	#Mean Normalization: transforms train set and test set the same way
	scaler = preprocessing.StandardScaler().fit(xTrain)
	scaler.transform(xTrain)
	scaler.transform(xTest)

	#Creates and uses the regression model
	#clf = SVR(kernel='linear', gamma='scale', C=1.0, epsilon=0.1)
	clf = GradientBoostingRegressor(random_state=21, n_estimators=400)
	clf.fit(xTrain, yTrain)
	scores = cross_val_score(clf, xTest, yTest, cv=5)
	print(feature + " R^2 Avg. Value: %.3f" % scores.mean())
	if(scores.mean() > .5):
		list.append(feature)
    