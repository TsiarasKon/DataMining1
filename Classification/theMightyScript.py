from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.parsing.porter import PorterStemmer
from sklearn.decomposition import TruncatedSVD
from sklearn import preprocessing
import numpy as np
import pandas as pd

dataset_path = "../datasets/"

K = 5	# KNN parameter

def add_titles(content, titles):
	newcontent = []
	mult = 0.001		# Title "weights" 10% of content length
	for i in range(0, len(content)):
		titlemesh = (" " + titles[i]) * max(1, int(len(content[i]) * mult));
		newcontent.append(content[i] + titlemesh)
	return newcontent;


custom_stopwords = set(ENGLISH_STOP_WORDS)
custom_stopwords.update(["say", "says", "said", "saying", "just", "year"])

train_data = pd.read_csv(dataset_path + 'train_set.csv', sep="\t")
train_data = train_data[0:3000]
test_data = pd.read_csv(dataset_path + 'test_set.csv', sep="\t")
test_data = test_data[0:500]
print "Loaded data."

le = preprocessing.LabelEncoder()
le.fit(train_data["Category"])
y = le.transform(train_data["Category"])

titled_train_data = add_titles(train_data['Content'], train_data['Title'])
titled_test_data = add_titles(test_data['Content'], test_data['Title'])

# Removing stopwords:
new_train_data = []
for doc in titled_train_data:
	doc_wordlist = doc.split()
	new_doc_wordlist = [word for word in doc_wordlist if word not in custom_stopwords]
	new_doc = ' '.join(new_doc_wordlist)
	new_train_data.append(new_doc)
new_test_data = []
for doc in titled_test_data:
	doc_wordlist = doc.split()
	new_doc_wordlist = [word for word in doc_wordlist if word not in custom_stopwords]
	new_doc = ' '.join(new_doc_wordlist)
	new_test_data.append(new_doc)

p = PorterStemmer()
train_docs = p.stem_documents(new_train_data)
test_docs = p.stem_documents(new_test_data)
print "Stemmed data."

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(train_docs)
Test = vectorizer.transform(test_docs)
print "Vectorized data"

svd_model = TruncatedSVD(n_components=5, n_iter=7, random_state=42)
svdX = svd_model.fit_transform(X)
#svdTest = svd_model.transform(Test)
print "SVD'd data"

# Cross Validation:
import nbayes, forest, svm, knn
from my_method import crossvalidation as my_method_crossvalidation
from sklearn.cross_validation import train_test_split

metrics = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
metrics_print = ["Accuracy", "Precision", "Recall", "F-Measure"]
metrics_results = []

metrics_results.append(nbayes.crossvalidation(X, y, metrics))
metrics_results.append(forest.crossvalidation(svdX, y, metrics))
metrics_results.append(svm.crossvalidation(svdX, y, metrics))
metrics_results.append(knn.crossvalidation(svdX, y, K))
metrics_results.append(my_method_crossvalidation(train_data, y, metrics))

cvFile = open("./EvaluationMetric_10fold.csv", "w+")

cvFile.write("Statistic Measure\tNaive Bayes\tRandom Forest\tSVM\tKNN\tMy Method\n")
for i in range(len(metrics)):
	cvFile.write(metrics[i])
	for res in metrics_results:
		cvFile.write('\t' + str(res[i]))
	cvFile.write('\n')


cvFile.close()
