import time
t_load = time.time()
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import os
from collections import defaultdict
from string import punctuation
import re
import json
import numpy as np
import pandas as pd
import scipy
from nltk.stem.porter import *
import matplotlib.pyplot as plt
from gensim.models import word2vec 
from nltk.tokenize import sent_tokenize
from scipy.sparse.linalg import norm as sparse_norm
import random
from config import specific_recommendations

def load_corpus():
	with open('corpus.txt') as f:
		ret = [line.strip() for line in f]
	return ret

def get_score_dicts( ):
	with open('novice_words.txt') as f:
		nov = {word:float(score) for word,score in [line.strip().split('\t') for line in f]}
	with open('experienced_words.txt') as f:
		exp = {word:float(score) for word,score in [line.strip().split('\t') for line in f]}
	return nov,exp

def get_raw_score(res):
	words = set( res.split(' ') )
	scores = [nov_scores.get(word, 0.0) for word in words] + [0.52*exp_scores.get(word, 0.0) for word in words]
	raw_score = sum(scores)
	raw_score = 1000.*raw_score/max( len(words), 100 )
	return raw_score

def create_scorer(raw_score_distribution,n=100):
	in_order = sorted(raw_score_distribution)
	num = len(in_order)
	quantiles = [in_order[ int(1.*k*num/n) ] for k in range(n) ]
	return make_scorer_from_quantile(quantiles)

def make_scorer_from_quantile(qtile):
	def scorer(score):
		# num = len(qtile)
		prc = [k for k,qtile_val in enumerate(qtile) if qtile_val>score]
		if len(prc)==0:
			return len(qtile)
		return prc[0]
	return scorer 

def mean(l):
	return 1.*sum(l)/len(l)

def score_resume(res):
	raw_score = get_raw_score(res)
	adjusted_score = scorer(raw_score)
	return 100-adjusted_score

def long_tail_tfidf(corpus):
	tfidf_vectorizer = TfidfVectorizer( max_features=20000, stop_words='english', ngram_range=(1, 2), min_df=5 )
	tfidf = tfidf_vectorizer.fit_transform( corpus )
	tfidf_labels = tfidf_vectorizer.get_feature_names()
	return {
		'vectorizer':tfidf_vectorizer,
		'tfidf':tfidf,
		'labels':tfidf_labels,
	}

def find_missing_words(doc, n_compare = 12, n_words = 30):
	doc_vec = long_tfidf_dict['vectorizer'].transform([doc])
	sim_mat = np.dot( long_tfidf_dict['tfidf'] , doc_vec.transpose() )
	temp = np.array( sim_mat.todense().reshape(-1) )
	indices = np.argsort(temp)[0][-2:-2-n_compare:-1]
	feature_sums = np.sum( long_tfidf_dict['tfidf'][indices,:].todense(), axis=0 ) / n_compare
	# temp_row = long_tfidf_dict['tfidf'][k,:].todense()
	differences = feature_sums - doc_vec
	word_indices = np.array(np.argsort(differences))[0][-2:-2-n_words:-1]
	# res_score = score_resume(corpus[k])
	return [ long_tfidf_dict['labels'][k] for k in word_indices]

def process_resume(doc=None):
	t_start = time.time()
	if doc=='random':
		doc = random.choice(corpus)
	print("\nYour Resume:\n")
	print(doc)
	doc	= clean_text(doc)
	res_score = score_resume(doc)
	n_dashes = (res_score)//3
	print('\n0 Novice:'+'-'*n_dashes+' '+str(int(res_score))+' '+(33-n_dashes)*'-'+':Experienced 100')
	missing_words = find_missing_words(doc)
	print('\n\nWords that appear more frequently in similar resumes:')
	print( ', '.join(missing_words) )
	spec_recs_found = [{
						'match':re.search(rec['from'],doc),
						'to':rec['to']
						}
						 for rec in specific_recommendations
						 if re.search(rec['from'],doc)]
	print( '\n\nSpecific Recommendations:' )
	print( '\n'.join([
		rec['match'].group(0)+
		' -> '+rec['to'] 
		for rec in spec_recs_found] ) )
	print("Analysis in {:.3f}s".format(time.time()-t_start) )

def clean_text(s,stem=False):
	"""Clean out the text"""
	ret = s.lower()
	ret = re.sub(r'[^a-z ]',' ',ret)
	ret = re.sub(r' +',' ',ret).strip()
	ret = re.sub(r'see more occupations related to this (activity|skill|task)','',ret)
	if stem:
		ret = ' '.join( stemmer.stem(word) for word in ret.split(' ') )
	return ret

corpus = load_corpus()
nov_scores,exp_scores = get_score_dicts()
print( sum( nov_scores.values() ), sum(exp_scores.values()) )
all_resume_scores = [get_raw_score(res) for res in corpus]
scorer = create_scorer(all_resume_scores)
long_tfidf_dict = long_tail_tfidf(corpus)

if __name__ == "__main__":
	print('Startup in','{:.3f}s'.format(time.time()-t_load))
	while True:
		ret = input("Score Another?")
		if ret=="break":
			break
		elif ret=='':
			pass
		else:
			process_resume(doc = ret)


