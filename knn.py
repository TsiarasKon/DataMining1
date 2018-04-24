from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn import preprocessing
from gensim.parsing.porter import PorterStemmer
import numpy as np
import pandas as pd
from heapq import heappush, heappop


""" KNN """

def manhattanDistance(v1, v2, dims):
    distance = 0.0
    for dim in range(dims):
        distance += np.square(abs(v1[dim] - v2[dim]))
    return np.sqrt(distance)


class KNN:
	def __init__(self):
		self.points = []  # list of 2-tuples all points learned and their categories
		self.categories = set()
	
	def learn(self, train_set, train_categories):     # aka fit
		self.categories = set(train_categories)
		for i in range(0, len(train_set)):
			self.points.append((train_set[i], train_categories[i]))

	def predict(self, test_set, K):
		prediction = []
		for newpoint in test_set:
			prediction.append(self.predict_for_one(newpoint, K))
		return prediction
	
	def predict_for_one(self, new_point, K):
		dim = len(new_point)
		if len(self.points[0][0]) != dim:
			print 'Error at predict_for_one: new point has wrong dimensions'
			return None
		heap = []               # minheap
		for point in self.points:
			heappush(heap, (manhattanDistance(point[0], new_point, dim), point[1]))
		category_count = {c:0 for c in self.categories}
		for i in range(0, K):   # only pop top-K (smallest K distances)
			_, c = heappop(heap)
			category_count[c] += 1
		max = -1
		maxcat = None
		for cat in self.categories:
			if category_count[cat] > max:
				max = category_count[cat]
				maxcat = cat
		return maxcat

""" KNN """


K = 5   # K parameter for KNN


def add_titles(content, titles):
	newcontent = []
	times = 3;
	for i in range(0, len(content)):
		titlemesh = (" " + titles[i]) * times;
		newcontent.append(content[i] + titlemesh)
	return newcontent;


custom_stopwords = set(ENGLISH_STOP_WORDS)
custom_stopwords.update(["say", "said", "saying", "just", "year"])

train_data = pd.read_csv('train_set.csv', sep="\t")
#train_data = train_data[0:100]
test_data = pd.read_csv('test_set.csv', sep="\t")
#test_data = test_data[0:25]
print "Loaded data"

titled_train_data = add_titles(train_data['Content'], train_data['Title'])
titled_test_data = add_titles(test_data['Content'], test_data['Title'])

p = PorterStemmer()
train_docs = p.stem_documents(titled_train_data)
test_docs = p.stem_documents(titled_test_data)

le = preprocessing.LabelEncoder()
le.fit(train_data["Category"])

y = le.transform(train_data["Category"])

count_vectorizer = CountVectorizer(stop_words=custom_stopwords)
X = count_vectorizer.fit_transform(train_docs)

svd_model = TruncatedSVD(n_components=2, n_iter=7, random_state=42)
X = svd_model.fit_transform(X)

clf = KNN()
clf.learn(X, y)


Test = count_vectorizer.transform(test_docs)
Test = svd_model.transform(Test)

y_pred = clf.predict(Test, K)

print le.inverse_transform(y_pred)

######

# 10-fold cross validation:
from sklearn.cross_validation import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=0.1)

clf = KNN()
clf.learn(X_train, y_train)
y_pred = clf.predict(X_test, K)

print "Metrics:"
ps = precision_score(y_test, y_pred, average='macro')
print " Precision Score: " + str(ps)
rs = recall_score(y_test, y_pred, average='macro')
print " Recall Score: " + str(rs)
f1s = f1_score(y_test, y_pred, average='macro')
print " F1 Score: " + str(f1s)
acs = accuracy_score(y_test, y_pred)
print " Accuracy Score: " + str(acs)
