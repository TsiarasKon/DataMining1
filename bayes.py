from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn import preprocessing
from gensim.parsing.porter import PorterStemmer
import numpy as np
import pandas as pd

custom_stopwords = set(ENGLISH_STOP_WORDS)
custom_stopwords.update(["say", "said", "saying", "just", "year"])

train_data = pd.read_csv('train_set.csv', sep="\t")

test_data = pd.read_csv('test_set.csv', sep="\t")
print "Loaded data"

p = PorterStemmer()
train_docs = p.stem_documents(train_data['Content'])


le = preprocessing.LabelEncoder()
le.fit(train_data["Category"])

y = le.transform(train_data["Category"])

count_vectorizer = CountVectorizer(stop_words=custom_stopwords)
X = count_vectorizer.fit_transform(train_docs)

'''
svd_model = TruncatedSVD(n_components=80, n_iter=7, random_state=42)

svdX = svd_model.fit_transform(X)
print X.shape
print svdX.shape
'''

nb = MultinomialNB()
nb.fit(X, y)

test_docs = p.stem_documents(test_data['Content'])
Test = count_vectorizer.transform(test_docs)

Test_pred = nb.predict(Test)

print le.inverse_transform(Test_pred)

################

from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=0.1)

nb = MultinomialNB()
nb.fit(X_train, y_train)
y_pred = nb.predict(X_test)

print classification_report(y_test, y_pred, target_names=list(le.classes_))
