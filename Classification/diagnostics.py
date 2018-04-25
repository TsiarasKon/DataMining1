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
