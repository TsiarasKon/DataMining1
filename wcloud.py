from wordcloud import WordCloud
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import pandas as pd

custom_stopwords = set(ENGLISH_STOP_WORDS)
custom_stopwords.update(["say", "said", "saying", "just"])		# + year?

# Read the whole text.
train_data = pd.read_csv('train_set.csv', sep="\t")

Categories = list(set(train_data['Category']))
category_text_list = [[] for x in Categories]
img_masks = {"Business":"business.png", "Football":"football.png", "Politics":"politics.png", "Film":"film.png", "Technology":"technology.png"}

for x in zip(train_data['Content'],train_data['Category']):
	index = Categories.index(x[1])
	category_text_list[index].append(x[0])

for i in range(0, len(Categories)):
	category_text = ' '.join(category_text_list[i])
	mask = np.array(Image.open(img_masks[Categories[i]]))
	wc = WordCloud(width=1280, height=720, stopwords=custom_stopwords, mask=mask, background_color="white").generate(category_text)
	fig = plt.figure(figsize=(13,8))
	plt.imshow(wc, interpolation="bilinear")
	plt.axis("off")	
	fig.savefig(Categories[i] + "_wc.png")
	plt.close()


'''
for i in range(0, len(Categories)):
	text_list = [x[0] for x in zip(train_data['Content'],train_data['Category']) if x[1] == Categories[i]]
	text = ' '.join(text_list)
	wc = WordCloud(width=1280, height=720, stopwords=custom_stopwords, background_color="white").generate(text)
	fig = plt.figure(figsize=(13,8))
	plt.imshow(wc, interpolation="bilinear")
	plt.axis("off")	
	fig.savefig(Categories[i] + "_wc.png")
	plt.close()
'''
