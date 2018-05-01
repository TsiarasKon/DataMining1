from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.parsing.porter import PorterStemmer
from sklearn.decomposition import TruncatedSVD
from sklearn import preprocessing
import numpy as np
import pandas as pd
from forest import get_accuracy

dataset_path = "../datasets/"


def add_titles(content, titles):
	newcontent = []
	mult = 0.001
	for i in range(0, len(content)):
		titlemesh = (" " + titles[i]) * max(1, int(len(content[i]) * mult));
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

components = [25, 50, 75, 100, 125, 150, 175, 200]
points = []		# (components, accuracy)
for comp in components:
	svd_model = TruncatedSVD(n_components=comp)		# random_state=42
	svdX = svd_model.fit_transform(X)
	acc = get_accuracy(svdX, y)
	points.append((comp, acc))
	print "SVD'd data with n = {} components".format(comp)

import matplotlib.pyplot as plt

points_x = [p[0] for p in points]
points_y = [p[1] for p in points]

plt.plot(points_x, points_y)
plt.plot(points_x, points_y, 'or')
plt.grid(True)
plt.title("Random Forests Classifier")
plt.xlabel('Components')
plt.ylabel('Accuracy')
plt.savefig("components_plot.png")
plt.show()
