from wordcloud import WordCloud
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import matplotlib.pyplot as plt
import pandas as pd

# Read the whole text.
train_data = pd.read_csv('train_set.csv', sep="\t")
train_data = train_data[0:50]

Categories = list(set(train_data['Category']))
for i in range(0, len(Categories)):
	text_list = [x[0] for x in zip(train_data['Content'],train_data['Category']) if x[1] == Categories[i]]
	text = ""
	for string in text_list:
		text += string + " "
	wc = WordCloud(width=1280, height=720, stopwords=set(ENGLISH_STOP_WORDS), background_color="white").generate(text)	
	fig = plt.figure(figsize=(13,8))
	plt.axis("off")
	plt.imshow(wc, interpolation="bilinear")
	fig.savefig("wc" + str(i+1) +".png")
	plt.close()

"""
# Generate a word cloud image
wc = WordCloud(width=2000, height=1000, stopwords=set(ENGLISH_STOP_WORDS), background_color="white").generate(text)

#image = wc.to_image()
#image.show()

fig = plt.figure(figsize=(20,10))
plt.imshow(wc, interpolation="bilinear")
plt.axis("off")
fig.savefig("wc.png")
plt.show()
plt.close()
"""
