from sklearn.ensemble import RandomForestClassifier


# prediction
def predict(train_features, train_categories, test_features):
	clf = RandomForestClassifier()
	clf.fit(train_features, train_categories)
	test_categories_prediction = clf.predict(test_features)
	return test_categories_prediction


# 10-fold cross validation
from sklearn.model_selection import cross_val_score

def crossvalidation(X, y, metrics):
	clf = RandomForestClassifier()
	scores = []
	for metric in metrics:
		scores.append(cross_val_score(clf, X, y, cv=10, scoring=metric).mean())
	return scores


def get_accuracy(X, y):
	clf = RandomForestClassifier()
	return cross_val_score(clf, X, y, cv=10, scoring='accuracy').mean()
