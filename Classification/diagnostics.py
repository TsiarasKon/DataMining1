from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.parsing.porter import PorterStemmer
from sklearn.decomposition import TruncatedSVD
from sklearn import preprocessing
import numpy as np
import pandas as pd

dataset_path = "../datasets/"

train_data = pd.read_csv(dataset_path + 'train_set.csv', sep="\t")
test_data = pd.read_csv(dataset_path + 'test_set.csv', sep="\t")
print "Loaded data."

category_count = {c:0 for c in set(train_data['Category'])}
for cat in train_data['Category']:
	category_count[cat] += 1

print category_count

N = len(train_data['Content'])
max_doc_len = -1
total_doc_len = 0

for doc in train_data['Content']:
	curr_doc_len = len(doc)
	total_doc_len += curr_doc_len
	if curr_doc_len > max_doc_len:
		max_doc_len = curr_doc_len

average_doc_len = total_doc_len / N

print "Number of samples: " + str(N)
print "Max content length: " + str(max_doc_len)
print "Average content length: " + str(average_doc_len)
