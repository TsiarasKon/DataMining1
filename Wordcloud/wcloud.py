from wordcloud import WordCloud
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import matplotlib.pyplot as plt
import pandas as pd

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

# Read the whole text.
train_data = pd.read_csv('../datasets/train_set.csv', sep="\t")

Categories = list(set(train_data['Category']))
category_text_list = [[] for x in Categories]

for x in zip(train_data['Content'],train_data['Category']):
	index = Categories.index(x[1])
	category_text_list[index].append(x[0])

for i in range(0, len(Categories)):
	category_text = ' '.join(category_text_list[i])
	wc = WordCloud(width=1280, height=720, stopwords=custom_stopwords, background_color="white").generate(category_text)
	fig = plt.figure(figsize=(13,8))
	plt.imshow(wc, interpolation="bilinear")
	plt.axis("off")	
	wc_path = Categories[i] + "_wc.png"
	fig.savefig(wc_path)
	plt.close()
	print "Created and saved " + Categories[i] + " workcloud at: ./" + wc_path
