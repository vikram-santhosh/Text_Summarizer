from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
from nltk.tokenize import sent_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.cluster import KMeans
from collections import Counter,OrderedDict
import numpy as np
import os
import re
import sys
import math

reload(sys)  
sys.setdefaultencoding('Cp1252')

class Summarizer:

	def preProcessor(self,text):

		tokens = word_tokenize(text)
		filtered_tokens = []

		for token in tokens:						#remove non alphabets
			if re.search('[a-zA-Z]', token):
				filtered_tokens.append(token)

		stopword_list = stopwords.words("english")
		words_minus_stopwords = []

		for w in filtered_tokens:					#remove stopwords
			if w not in stopword_list:
				words_minus_stopwords.append(w)
		
		word_lemmatizer = WordNetLemmatizer()
		lemmatized_words = []

		for w in words_minus_stopwords:				#lemmatize
			temp = word_lemmatizer.lemmatize(w)
			lemmatized_words.append(temp)

		return lemmatized_words

	def getSentences(self,input_path): # extracts sentences
		doc = open(input_path)
		text = doc.read()
		sent = sent_tokenize(text)
		return sent

	def TFIDF(self,corpus,sent):  #feature extraction

		#computing TF

		vocab = set(corpus)
		word_count = OrderedDict()
		tf = OrderedDict()

		for s in sent:
			words_in_sent = self.preProcessor(s)
			tf_count = Counter(words_in_sent)
			c = len(words_in_sent)
			l = []
			for word in vocab:
				if word in words_in_sent:
					tmp = float(tf_count[word]) / c
					l.append(tmp)
				else:
					l.append(0)
			tf[s] = l;

		#compute IDF
		n = len(sent)
		idf = OrderedDict()

		for word in vocab:
			idf[word] = 0


		for word in vocab:
			for s in sent:
				words_in_sent = self.preProcessor(s)
				for w in words_in_sent:
					if word == w:
						idf[word] +=1
						break

		#idf-matrix

		idf_matrix = OrderedDict()
		for s in sent:
			words_in_sent = self.preProcessor(s)
			l = []
			for w in vocab:
				if w in words_in_sent:
					l.append(idf[w])
				else:
					l.append(0)
			idf_matrix[s] = l

		#computing tf-idf

		tfidf = OrderedDict()

		for s in sent:
			words = self.preProcessor(s)
			l = []
			for i,j in zip(tf[s],idf_matrix[s]):
					tmp = math.log10(1.0+i) * math.log10(1+(n/(j+1.0)))
					l.append(tmp)
			tfidf[s] = l

		return tfidf

	def cosine_similarity(self,sent,tfidf_matrix): # computing cosine similarity

		n = len(sent)
		cosine_similarity = OrderedDict()

		for v1 in tfidf_matrix:
			l = []
			for v2 in tfidf_matrix:
				if v1 == v2:
					l.append(0)
				else:
					numo = np.dot(tfidf_matrix[v1],tfidf_matrix[v2])
					#deno = self.length(tfidf_matrix[v1]) * self.length(tfidf_matrix[v2])
					deno = math.sqrt(np.dot(tfidf_matrix[v1],tfidf_matrix[v1])) * math.sqrt(np.dot(tfidf_matrix[v2],tfidf_matrix[v2]))
					tmp = numo/deno
					l.append(tmp)

			cosine_similarity[v1] = l

		return cosine_similarity


	def cluster(self,cosine_similarity,num,k,sent): # cluster formation
		cosine_similarity = np.array(cosine_similarity.values()).reshape(num,num)
		km = KMeans(n_clusters=k)
		index = km.fit_predict(cosine_similarity)
		print index
		cluster = OrderedDict()
		for i in range(k):
			l = []
			for j in range(num):
				if i == index[j]: # if the jth sentence belongs to the ith cluster 
					l.append(sent[j])
			cluster[i] = l

		return cluster

	def display(self,cluster,k):
		for i in range(k):
			print cluster[i][0]
			
if __name__ == '__main__':

	import sys
	input_path = sys.argv[1]
	summarizer = Summarizer()

	doc = open(input_path)
	text = doc.read()

	corpus = summarizer.preProcessor(text)
	sent = summarizer.getSentences(input_path)
	tfidf_matrix = summarizer.TFIDF(corpus,sent)
	cosine_similarity = summarizer.cosine_similarity(sent,tfidf_matrix)
	k = int(len(sent)*0.2)
	cluster = summarizer.cluster(cosine_similarity,len(sent),k,sent)
	summarizer.display(cluster,k)