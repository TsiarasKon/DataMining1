from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import CountVectorizer
from gensim.parsing.porter import PorterStemmer
from sklearn.decomposition import TruncatedSVD
from sklearn import preprocessing
import numpy as np
import pandas as pd

K = 5	# KNN parameter

def add_titles(content, titles):
	newcontent = []
	times = 4;
	for i in range(0, len(content)):
		titlemesh = (" " + titles[i]) * times;
		newcontent.append(content[i] + titlemesh)
	return newcontent;


custom_stopwords = set(ENGLISH_STOP_WORDS)
custom_stopwords.update(["say", "says", "said", "saying", "just", "year"])

train_data = pd.read_csv('train_set.csv', sep="\t")
train_data = train_data[0:100]
test_data = pd.read_csv('test_set.csv', sep="\t")
test_data = test_data[0:25]
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

count_vectorizer = CountVectorizer(stop_words=custom_stopwords)
X = count_vectorizer.fit_transform(train_docs)
Test = count_vectorizer.transform(test_docs)
print "Vectorized data"

svd_model = TruncatedSVD(n_components=10, n_iter=7, random_state=42)
svdX = svd_model.fit_transform(X)
#svdTest = svd_model.transform(Test)
print "SVD'd data"


'''
# Prediction: 

Test_pred = predict(svdTest)

predFile = open("./testSet_categories.csv", "w+")
for i in range(Test_pred):
	predFile.write(test_data['Id'][i] + '\t' + Test_pred[i] + '\n')
predFile.close()
'''

# Cross Validation:
import nbayes, forest, svm, knn
from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=0.1)
svdX_train, svdX_test, svdy_train, svdy_test = train_test_split(svdX, y, random_state=1, test_size=0.1)

metrics = ["Accuracy", "Precision", "Recall", "F-Measure"]
metrics_results = []

metrics_results.append(nbayes.crossvalidation(X_train, X_test, y_train, y_test))
metrics_results.append(forest.crossvalidation(svdX_train, svdX_test, svdy_train, svdy_test))
metrics_results.append(svm.crossvalidation(svdX_train, svdX_test, svdy_train, svdy_test))
metrics_results.append(knn.crossvalidation(svdX_train, svdX_test, svdy_train, svdy_test, K))

cvFile = open("./EvaluationMetric_10fold.csv", "w+")

cvFile.write("Statistic Measure\tNaive Bayes\tRandom Forest\tSVM\tKNN\tMy Method\n")
for i in range(len(metrics)):
	cvFile.write(metrics[i])
	for res in metrics_results:
		cvFile.write('\t' + str(res[i]))
	cvFile.write('\n')


cvFile.close()
