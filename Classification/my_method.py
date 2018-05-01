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
	mult = 0.001		
	for i in range(0, len(content)):
		titlemesh = (" " + titles[i]) * max(1, int(len(content[i]) * mult))
		newcontent.append(content[i] + titlemesh)
	return newcontent;


def preprocess_data(train_data, test_data):
	custom_stopwords = set(ENGLISH_STOP_WORDS)
	custom_stopwords.update(["say", "says", "said", "saying", "just", "year", "man", "men", "woman", \
		"women", "guy", "guys", "run", "running", "ran", "run", "do", "don't", "does", "doesn't" , \
		"doing", "did", "didn't",  "use", "used", "continue", "number", "great", "big", "good", "bad", \
		"better", "worse", "best", "worst", "actually", "fact", "way", "tell", "told", "include", "including", \
		"want", "wanting", "will", "won't", "give", "given", "month", "day", "place", "area", "look", \
		"looked", "far", "near", "get", "getting", "got", "know", "knows", "knew", "long", "week", "have", \
		"has", "haven't", "hasn't", "having", "had", "hadn't", "not", "think", "thinking", "Monday", \
		"Tuesday", "Wednesday", "Thursday", "Saturday", "Sunday", "high", "low", "thing", "there", "they're", \
		"It", "I've", "I'd", "He's", "She's", "They've", "I'm", "You're", "your", "their", "his", "hers", \
		"mine", "today", "yesterday", "it", "ve", "going", "go", "went", "lot", "don", "saw", "seen", "come", "came"])

	titled_train_data = add_titles(train_data['Content'], train_data['Title'])
	if test_data is not None:
		titled_test_data = add_titles(test_data['Content'], test_data['Title'])

	# Removing stopwords:
	new_train_data = []
	for doc in titled_train_data:
		doc_wordlist = doc.split()
		new_doc_wordlist = [word for word in doc_wordlist if word not in custom_stopwords]
		new_doc = ' '.join(new_doc_wordlist)
		new_train_data.append(new_doc)
	if test_data is not None:
		new_test_data = []
		for doc in titled_test_data:
			doc_wordlist = doc.split()
			new_doc_wordlist = [word for word in doc_wordlist if word not in custom_stopwords]
			new_doc = ' '.join(new_doc_wordlist)
			new_test_data.append(new_doc)

	p = PorterStemmer()
	train_docs = p.stem_documents(new_train_data)
	if test_data is not None:
		test_docs = p.stem_documents(new_test_data)
	print "my_method: Stemmed data."

	vectorizer = TfidfVectorizer()
	X = vectorizer.fit_transform(train_docs)
	if test_data is not None:
		Test = vectorizer.transform(test_docs)
	else:
		Test = None
	print "my_method: Vectorized data"


	svd_model = TruncatedSVD(n_components=80, random_state=42)
	X = svd_model.fit_transform(X)
	if test_data is not None:
		Test = svd_model.transform(Test)
	print "SVD'd data"


	return X, Test


# prediction
def predict(train_features, train_categories, test_features):
	clf = MultinomialNB()
	clf.fit(train_features, train_categories)
	test_categories_prediction = clf.predict(test_features)
	return test_categories_prediction


# 10-fold cross validation
from sklearn.model_selection import cross_val_score

def crossvalidation(train_data, y, metrics):
	X, _ = preprocess_data(train_data, None)
	clf = MultinomialNB()
	scores = []
	for metric in metrics:
		scores.append(cross_val_score(clf, X, y, cv=10, scoring=metric).mean())
	return scores

################################

if __name__ == "__main__":
	train_data = pd.read_csv(dataset_path + 'train_set.csv', sep="\t")
	test_data = pd.read_csv(dataset_path + 'test_set.csv', sep="\t")
	print "my_method: Loaded data."

	le = preprocessing.LabelEncoder()
	le.fit(train_data["Category"])
	y = le.transform(train_data["Category"])

	X, Test = preprocess_data(train_data, test_data)

	# Prediction: 
	import svm
	Test_pred = le.inverse_transform(svm.predict(X, y, Test))

	predFile = open("./testSet_categories.csv", "w+")
	predFile.write("Id,Category\n")
	for i in range(len(Test_pred)):
		predFile.write(str(test_data['Id'][i]) + ',' + Test_pred[i] + '\n')
	predFile.close()
