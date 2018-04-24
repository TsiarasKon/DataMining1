from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import preprocessing
from gensim.parsing.porter import PorterStemmer
import numpy as np
import pandas as pd


def add_titles(content, titles):
	newcontent = []
	times = 5;
	for i in range(0, len(content)):
		titlemesh = (" " + titles[i]) * times;
		newcontent.append(content[i] + titlemesh)
	return newcontent;


custom_stopwords = set(ENGLISH_STOP_WORDS)
custom_stopwords.update(["say", "said", "saying", "just", "year"])

train_data = pd.read_csv('train_set.csv', sep="\t")
train_data = train_data[0:100]
test_data = pd.read_csv('test_set.csv', sep="\t")
test_data = test_data[0:25]
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


clf = MultinomialNB()
clf.fit(X, y)

Test = count_vectorizer.transform(test_docs)

Test_pred = clf.predict(Test)

print le.inverse_transform(Test_pred)

################

# 10-fold cross validation:
from sklearn.cross_validation import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

clf = MultinomialNB()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print "Metrics:"
ps = precision_score(y_test, y_pred, average='macro')
print " Precision Score: " + str(ps)
rs = recall_score(y_test, y_pred, average='macro')
print " Recall Score: " + str(rs)
f1s = f1_score(y_test, y_pred, average='macro')
print " F1 Score: " + str(f1s)
acs = accuracy_score(y_test, y_pred)
print " Accuracy Score: " + str(acs)
