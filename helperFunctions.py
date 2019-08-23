from sklearn.model_selection import train_test_split
from sklearn.svm import SVR

def printRSquared(xSet, ySet, feature):
	xTrain, xTest, yTrain, yTest = train_test_split(xSet, ySet, test_size=.2, random_state=42)
	clf = SVR(kernel='linear', gamma='scale', C=1.0, epsilon=0.1)
	clf.fit(xTrain, yTrain)
	print("Value of R^2 for the " + feature + " set: " + str(clf.score(xTest, yTest)))
    