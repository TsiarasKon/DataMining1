from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.parsing.porter import PorterStemmer
from sklearn.decomposition import TruncatedSVD
from sklearn.naive_bayes import MultinomialNB
from sklearn import preprocessing
import numpy as np
import pandas as pd

dataset_path = "../datasets/"

def add_titles(content, titles):
	newcontent = []
	mult = 0.001		# Title "weights" 10% of content length
	for i in range(0, len(content)):
		titlemesh = (" " + titles[i]) * int(len(content[i]) * mult);
		newcontent.append(content[i] + titlemesh)
	return newcontent;


# prediction
def predict(train_features, train_categories, test_features):
	clf = MultinomialNB()
	clf.fit(train_features, train_categories)
	test_categories_prediction = clf.predict(test_features)
	return test_categories_prediction


# 10-fold cross validation
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


def crossvalidation(train_features, test_features, train_categories, test_categories):
	clf = MultinomialNB()
	clf.fit(train_features, train_categories)
	test_categories_prediction = clf.predict(test_features)
	# Metrics are:
	acs = accuracy_score(test_categories, test_categories_prediction)
	ps = precision_score(test_categories, test_categories_prediction, average='macro')
	rs = recall_score(test_categories, test_categories_prediction, average='macro')
	f1s = f1_score(test_categories, test_categories_prediction, average='macro')
	return acs, ps, rs, f1s

################################

custom_stopwords = set(ENGLISH_STOP_WORDS)
custom_stopwords.update(["say", "says", "said", "saying", "just", "year"])

train_data = pd.read_csv(dataset_path + 'train_set.csv', sep="\t")
test_data = pd.read_csv(dataset_path + 'test_set.csv', sep="\t")
print "Loaded data."

le = preprocessing.LabelEncoder()
le.fit(train_data["Category"])
y = le.transform(train_data["Category"])

titled_train_data = add_titles(train_data['Content'], train_data['Title'])
titled_test_data = add_titles(test_data['Content'], test_data['Title'])

p = PorterStemmer()
train_docs = p.stem_documents(titled_train_data)
test_docs = p.stem_documents(titled_test_data)
print "Stemmed data."

vectorizer = TfidfVectorizer(stop_words=custom_stopwords)
X = vectorizer.fit_transform(train_docs)
Test = vectorizer.transform(test_docs)
print "Vectorized data"


# Prediction: 
Test_pred = le.inverse_transform(predict(X, y, Test))

predFile = open("./testSet_categories.csv", "w+")
predFile.write("Id,Category\n")
for i in range(len(Test_pred)):
	predFile.write(str(test_data['Id'][i]) + ',' + Test_pred[i] + '\n')
predFile.close()
