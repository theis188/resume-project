from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import os
from collections import defaultdict
from string import punctuation
import re
import json
import numpy as np
import pandas as pd
import scipy
from nltk.stem.porter import PorterStemmer
from gensim.models import word2vec 
from nltk.tokenize import sent_tokenize
from scipy.sparse.linalg import norm as sparse_norm
stemmer = PorterStemmer()

def clean_text(s,stem=False):
	"""Clean out the text"""
	ret = s.lower()
	ret = re.sub(r'[^a-z ]',' ',ret)
	ret = re.sub(r' +',' ',ret).strip()
	ret = re.sub(r'see more occupations related to this (activity|skill|task)','',ret)
	if stem:
		ret = ' '.join( stemmer.stem(word) for word in ret.split(' ') )
	return ret

def get_sentences_from_text(s):
	sentences = sent_tokenize(s)
	ret = [clean_text(sent) for sent in sentences]
	return ret

def process_all():
	"""Create a corpus and list of attributes for documents for Tfidf Fun"""
	files = os.listdir('records')
	files = [file for file in files if file not in ('.DS_Store','old')]
	attr_list = []
	corpus = []
	sentences = []
	corp_set = set()
	for file in files:
		with open('records/'+file) as f:
			attr_list, corpus, sentences  = proc_file(f,file,corpus,attr_list,corp_set,sentences)
	return attr_list,corpus,sentences

def proc_file(f,file,corpus,attr_list,corp_set,sentences):
	for line in f:
		jline = json.loads( line.strip() )
		text = proc_jline(jline)
		if not jline['title']: ##filter out empty titles
			continue
		if text:
		# if len(text.split(' '))>200:
			obj= {'level':jline['level']}
			obj['title'] = file.split('.')[0]
			if text not in corp_set:
				corp_set.add(text)
				corpus.append(text)
				attr_list.append(obj)
				new_sentences = get_sentences_from_jline(jline)
				sentences += new_sentences
	return attr_list, corpus, sentences

def get_sentences_from_jline(jline):
	clean_sent_list = [ get_sentences_from_text( we['description'] )  for we in jline['work_experience'] ]
	ret = [sent.split(' ') for sent_list in clean_sent_list for sent in sent_list]
	return ret

def proc_jline(jline):
	clean_text_list = [ clean_text( we['description'], stem=False )  for we in jline['work_experience'] ]
	text = ' '.join( ct for ct in clean_text_list if len(ct)<5000 ) ## Filter out overlong descriptions
	return text

def generate_df():
	attrs,corp,_ = process_all()
	data_dict = defaultdict(list)
	for attr in attrs:
		for k,v in attr.items():
			data_dict[k].append(v)
	data_dict['corpus']=corp
	ret_df = pd.DataFrame(data_dict)
	return ret_df

###############

df = generate_df()
df_sub = df[df.level.isin( ['More than 10 years','Less than 1 year'] )].reset_index()
del df_sub['index']

df_sub['experienced']=(df_sub.level=='More than 10 years')

from sklearn.linear_model import LogisticRegression
import random

def get_test_train_split(df_arg,frac):
	min_num = min(df_arg.groupby( ['level'] ).count().title )
	max_num = max(df_arg.groupby( ['level'] ).count().title )
	max_frac = frac*min_num/max_num
	def inner_fun(x):
		if x=='More than 10 years':
			return random.random()>max_frac
		else: return random.random()>frac
	df_arg['train'] = df_arg.level.apply(inner_fun)

get_test_train_split(df_sub,0.8)

logistic_regression = LogisticRegression(penalty='l1',C=1.0)
tfidf_vectorizer = TfidfVectorizer( max_features=1000, ngram_range=(1,1), 
								stop_words='english', min_df=0.02 )

tfidf = tfidf_vectorizer.fit_transform( df_sub.corpus )

train_indexes = [n for n,tf in enumerate(df_sub.train) if tf]
test_indexes = [n for n,tf in enumerate(df_sub.train) if not tf]

# df_sub[ df_sub.train ].experienced

logistic_regression.fit( tfidf[  train_indexes,: ],  df_sub[ df_sub.train ].experienced )

logistic_regression.predict_proba( tfidf[test_indexes,:] )

df_results=df_sub[ ~df_sub.train ].reset_index()
del df_results['index']

df_results['prediction'] = logistic_regression.predict( tfidf[test_indexes,:] )
probs = logistic_regression.predict_proba(tfidf[test_indexes,:])[:,1]
df_results['probs'] = probs


# df_results.groupby(['prediction','experienced']).count().level
((df_results.probs>0.95)==df_results.experienced).mean()

actual = df_results.experienced

inds = [np.argsort(logistic_regression.coef_)[0][::-1] ]

names = tfidf_vectorizer.get_feature_names()
logisitic_regression.coef_[k]

[word for k,word in enumerate(names) if k in inds[0] 
	if logisitic_regression.coef_[k]]