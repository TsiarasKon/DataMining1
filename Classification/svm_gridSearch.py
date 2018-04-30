from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.parsing.porter import PorterStemmer
from sklearn.decomposition import TruncatedSVD
from sklearn import preprocessing
import numpy as np
import pandas as pd
import svm

dataset_path = "../datasets/"

best_n_components = 80		# found from components_graph.py

def add_titles(content, titles):
	newcontent = []
	mult = 0.001
	for i in range(0, len(content)):
		titlemesh = (" " + titles[i]) * max(1, int(len(content[i]) * mult));
		newcontent.append(content[i] + titlemesh)
	return newcontent;


custom_stopwords = set(ENGLISH_STOP_WORDS)
custom_stopwords.update(["say", "says", "said", "saying", "just", "year"])

train_data = pd.read_csv(dataset_path + 'train_set.csv', sep="\t")
print "Loaded data."

le = preprocessing.LabelEncoder()
le.fit(train_data["Category"])
y = le.transform(train_data["Category"])

titled_train_data = add_titles(train_data['Content'], train_data['Title'])

# Removing stopwords:
new_train_data = []
for doc in titled_train_data:
	doc_wordlist = doc.split()
	new_doc_wordlist = [word for word in doc_wordlist if word not in custom_stopwords]
	new_doc = ' '.join(new_doc_wordlist)
	new_train_data.append(new_doc)

p = PorterStemmer()
train_docs = p.stem_documents(new_train_data)
print "Stemmed data."

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(train_docs)
print "Vectorized data"

svd_model = TruncatedSVD(n_components=best_n_components, n_iter=7, random_state=42)
svdX = svd_model.fit_transform(X)

metrics = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
best_params = svm.find_best_params(svdX, y, metrics)
print best_params
