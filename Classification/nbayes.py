from sklearn.naive_bayes import MultinomialNB


# prediction
def predict(train_features, train_categories, test_features):
	clf = MultinomialNB()
	clf.fit(train_features, train_categories)
	test_categories_prediction = clf.predict(test_features)
	return test_categories_prediction


# 10-fold cross validation
from sklearn.model_selection import cross_val_score

def crossvalidation(X, y, metrics):
	clf = MultinomialNB()
	scores = []
	for metric in metrics:
		scores.append(cross_val_score(clf, X, y, cv=10, scoring=metric).mean())
	return scores
