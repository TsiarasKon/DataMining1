from sklearn.ensemble import RandomForestClassifier


# prediction
def predict(train_features, train_categories, test_features):
	clf = RandomForestClassifier()
	clf.fit(train_features, train_categories)
	test_categories_prediction = clf.predict(test_features)
	return test_categories_prediction


# 10-fold cross validation
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


def crossvalidation(train_features, test_features, train_categories, test_categories):
	clf = RandomForestClassifier()
	clf.fit(train_features, train_categories)
	test_categories_prediction = clf.predict(test_features)
	# Metrics are:
	acs = accuracy_score(test_categories, test_categories_prediction)
	ps = precision_score(test_categories, test_categories_prediction, average='macro')
	rs = recall_score(test_categories, test_categories_prediction, average='macro')
	f1s = f1_score(test_categories, test_categories_prediction, average='macro')
	return acs, ps, rs, f1s
