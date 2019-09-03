import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import learning_curve


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

def predictNext(xSet, ySet, feature, predictInput):
	xTrain, xTest, yTrain, yTest = train_test_split(xSet, ySet, test_size=.2, random_state=42)

	clf = GradientBoostingRegressor(random_state=21, n_estimators=400)
	clf.fit(xTrain, yTrain)

	prediction = clf.predict(predictInput)
	print("Prediction for " + feature + " input: " + str(prediction))

def plotLearningCurve(estimator, title, X, y, ylim=None, cv=None, n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
	"""
	Generates a simple plot of the test and train learning curve

	Parameters
	-----------
	estimator: object that has the fit and predict methods (aka the Regressor)

	title: string- title for the chart

	X: array-like, shape (n_samples, n_features)
		Training vector

	y: array-like, shape (n_samples)
		Target vector

	yLim: tuple, shape (ymin, ymax), optional
		Defines minimum and maximum y values to plot

	cv: integer, cross validation generator, optional
		Number of folds to cross validate (default will be 3)

	n_jobs: integer, optional
		Number of jobs to run in parallel (default is 1)
	"""

	plt.figure()
	plt.title(title)
	if ylim is not None:
		plt.ylim(*ylim)
	plt.xlabel("Training Examples")
	plt.ylabel("Score")
	train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
	train_scores_mean = np.mean(train_scores, axis=1)
	train_scores_std = np.std(train_scores, axis=1)
	test_scores_mean = np.mean(test_scores, axis=1)
	test_scores_std = np.std(test_scores, axis=1)
	plt.grid()

	plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
	plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="g")
	plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training Score")
	plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-Validation Score")

	plt.legend(loc="best")
	return plt
	

    