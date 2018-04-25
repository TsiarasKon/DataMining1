import KNN_Classifier

# prediction
def predict(train_features, train_categories, test_features, K):
	clf = KNN_Classifier.KNN()
	clf.fit(train_features, train_categories)
	test_categories_prediction = clf.predict(test_features, K)
	return test_categories_prediction


# 10-fold cross validation
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


def crossvalidation(train_features, test_features, train_categories, test_categories, K):
	clf = KNN_Classifier.KNN()
	clf.fit(train_features, train_categories)
	test_categories_prediction = clf.predict(test_features, K)
	# Metrics are:
	acs = accuracy_score(test_categories, test_categories_prediction)
	ps = precision_score(test_categories, test_categories_prediction, average='macro')
	rs = recall_score(test_categories, test_categories_prediction, average='macro')
	f1s = f1_score(test_categories, test_categories_prediction, average='macro')
	return acs, ps, rs, f1s
