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
	mult = 0.001	
	for i in range(0, len(content)):
		titlemesh = (" " + titles[i]) * max(1, int(len(content[i]) * mult))
		newcontent.append(content[i] + titlemesh)
	return newcontent;


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

train_data = pd.read_csv(dataset_path + 'train_set.csv', sep="\t")
test_data = pd.read_csv(dataset_path + 'test_set.csv', sep="\t")
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

svd_model5 = TruncatedSVD(n_components=5, random_state=42)
svdX5 = svd_model5.fit_transform(X)
svd_model50 = TruncatedSVD(n_components=50, random_state=13)
svdX50 = svd_model50.fit_transform(X)
svd_model75 = TruncatedSVD(n_components=250, random_state=13)
svdX75 = svd_model75.fit_transform(X)
print "SVD'd data"

# Cross Validation:
import nbayes, forest, svm, knn
from my_method import crossvalidation as my_method_crossvalidation
from sklearn.cross_validation import train_test_split

metrics = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
metrics_print = ["Accuracy", "Precision", "Recall", "F-Measure"]
metrics_results = []
'''
metrics_results.append(nbayes.crossvalidation(X, y, metrics))
print "NBayes metrics:" 
print metrics_results[0]
metrics_results.append(forest.crossvalidation(svdX50, y, metrics))
print "Forest metrics:"
print metrics_results[1]'''
metrics_results.append(svm.crossvalidation(svdX75, y, metrics))
print "SVM metrics:"
print metrics_results
metrics_results.append(knn.crossvalidation(svdX5, y, K))
'''print "KNN metrics:"
print metrics_results[3]
#metrics_results.append(my_method_crossvalidation(train_data, y, metrics))'''

cvFile = open("./EvaluationMetric_10fold.csv", "w+")

cvFile.write("Statistic Measure\tNaive Bayes\tRandom Forest\tSVM\tKNN\tMy Method\n")
for i in range(len(metrics)):
	cvFile.write(metrics_print[i])
	for res in metrics_results:
		cvFile.write('\t' + str(res[i]))
	cvFile.write('\t' + str(metrics_results[2][i]))		# "My Method"
	cvFile.write('\n')


cvFile.close()
