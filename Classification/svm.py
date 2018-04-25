from sklearn import svm


# prediction
def predict(train_features, train_categories, test_features):
	clf = svm.SVC()
	clf.fit(train_features, train_categories)
	test_categories_prediction = clf.predict(test_features)
	return test_categories_prediction


# 10-fold cross validation
from sklearn.model_selection import cross_val_score

def crossvalidation(X, y, metrics):
	clf = svm.SVC()
	scores = []
	for metric in metrics:
		scores.append(cross_val_score(clf, X, y, cv=10, scoring=metric).mean())
	return scores

