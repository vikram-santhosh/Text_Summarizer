from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
from nltk.tokenize import sent_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from collections import Counter
import numpy
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

	def getSentences(self,input_path):
		doc = open(input_path)
		text = doc.read()
		sent = sent_tokenize(text)
		return sent

	def TFIDF(self,corpus,sent):

		#computing TF

		vocab = set(corpus)
		word_count = {}
		tf = {}

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
		idf = {} 

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

		idf_matrix = {}
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
		tfidf = {}


		for s in sent:
			words = self.preProcessor(s)
			l = []
			for i,j in zip(tf[s],idf_matrix[s]):
					tmp = math.log10(1.0+i) * math.log10(1+(n/(j+1.0)))
					l.append(tmp)
			tfidf[s] = l

		return tfidf

	def dotproduct(self,v1, v2):
		s = 0
		for i,j in zip(v1,v2):
			s += (i*j)
	  	return s

	def length(self,v):
		return math.sqrt(self.dotproduct(v, v))

	def cosine_similarity(self,sent,tfidf_matrix):

		n = len(sent)
		cosine_similarity = {}

		for v1 in tfidf_matrix:
			l = []
			for v2 in tfidf_matrix:
				if v1 == v2:
					l.append(0)
				else:
					tmp = math.acos(self.dotproduct(tfidf_matrix[v1],tfidf_matrix[v2]) / 
						(self.length(tfidf_matrix[v1]) * self.length(tfidf_matrix[v2])))
					l.append(tmp)

			cosine_similarity[v1] = l

		return cosine_similarity

if __name__ == '__main__':

	import sys
	input_path = sys.argv[1]
	summarizer = Summarizer()

	doc = open(input_path)
	text = doc.read()

	corpus = summarizer.preProcessor(text)
	sent = summarizer.getSentences(input_path)
	tfidf_matrix = summarizer.TFIDF(corpus,sent)
	print tfidf_matrix
	cosine_similarity = summarizer.cosine_similarity(sent,tfidf_matrix)
	for i in cosine_similarity:
		for j in cosine_similarity[i]:
			print j,
		print "\n"