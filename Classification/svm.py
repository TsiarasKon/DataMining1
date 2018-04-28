from sklearn import svm
from sklearn.model_selection import GridSearchCV

def find_best_params(X, y, metrics):
	params = {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['linear', 'rbf']}
	clf = GridSearchCV(svm.SVC(), params)
	clf.fit(X, y)
	return clf.best_params_


best_C = 10
best_gamma = 0.001
best_kernel = 'linear'


# prediction
def predict(train_features, train_categories, test_features):
	clf = svm.SVC(C=best_C, gamma=best_gamma, kernel=best_kernel)
	clf.fit(train_features, train_categories)
	test_categories_prediction = clf.predict(test_features)
	return test_categories_prediction


# 10-fold cross validation
from sklearn.model_selection import cross_val_score

def crossvalidation(X, y, metrics):
	clf = svm.SVC(C=best_C, gamma=best_gamma, kernel=best_kernel)
	scores = []
	for metric in metrics:
		scores.append(cross_val_score(clf, X, y, cv=10, scoring=metric).mean())
	return scores
