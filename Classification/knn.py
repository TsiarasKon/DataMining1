import KNN_Classifier
from sklearn.model_selection import StratifiedKFold
import numpy as np

# prediction
def predict(train_features, train_categories, test_features, K):
	clf = KNN_Classifier.KNN(K)
	clf.fit(train_features, train_categories)
	test_categories_prediction = clf.predict(test_features)
	return test_categories_prediction


# 10-fold cross validation
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


# 10-fold cross validation
def crossvalidation(X, y, K):
	skf = StratifiedKFold(n_splits=10)
	scores = [[] for x in range(4)]		# 4 metrics
	for train_index, test_index in skf.split(X, y):
		X_train, X_test = X[train_index], X[test_index]
		y_train, y_test = y[train_index], y[test_index]
		clf = KNN_Classifier.KNN(K)
		clf.fit(X_train, y_train)
		y_pred = clf.predict(X_test)
		scores[0].append(accuracy_score(y_test, y_pred))
		scores[1].append(precision_score(y_test, y_pred, average='macro'))
		scores[2].append(recall_score(y_test, y_pred, average='macro'))
		scores[3].append(f1_score(y_test, y_pred, average='macro'))
	return np.mean(scores[0]), np.mean(scores[1]), np.mean(scores[2]), np.mean(scores[3])

def get_accuracy(X, y):
	skf = StratifiedKFold(n_splits=10)
	acc = []
	for train_index, test_index in skf.split(X, y):
		X_train, X_test = X[train_index], X[test_index]
		y_train, y_test = y[train_index], y[test_index]
		clf = KNN_Classifier.KNN(K)
		clf.fit(X_train, y_train)
		y_pred = clf.predict(X_test)
		acc.append(accuracy_score(y_test, y_pred))
	return np.mean(acc)
