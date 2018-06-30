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

def get_tfidf(corpus):
	tfidf_vectorizer = TfidfVectorizer( max_features=8000, stop_words='english', ngram_range=(1, 1), min_df=0.03 )
	tfidf = tfidf_vectorizer.fit_transform( corpus )
	count_vectorizer = CountVectorizer( max_features=8000, stop_words='english', ngram_range=(1, 1), min_df=0.03 )
	count = count_vectorizer.fit_transform( corpus )
	tfidf_labels = tfidf_vectorizer.get_feature_names()
	# tfidf_label_dict = { label:ind for ind,label in enumerate( tfidf_labels ) }
	count_labels = count_vectorizer.get_feature_names()
	count_label_dict = { label:ind for ind,label in enumerate( count_labels ) }
	# count_label_indices = [ tfidf_label_dict[label] for label in count_labels if label in tfidf_label_dict]
	new_count = np.zeros( tfidf.shape )
	for k,l in enumerate( tfidf_labels ):
		if l in count_label_dict:
			new_count[ : , k ] = count[ : , count_label_dict[ l ] ].todense().reshape( new_count[ : , k ].shape )
	return {
		'tfidf':tfidf,
		'count':scipy.sparse.csr_matrix(new_count),
		'labels':tfidf_labels,
	}

def create_corpus_file(corpus):
	with open('corpus.txt','w') as f:
		for doc in corpus:
			f.write(doc+'\n')

# levels = ['3-5 years', 'Less than 1 year', 'More than 10 years']

def get_corrected_counts(tfidf_dict,attr_list, counttype='tfidf', correct_for_number_collected=False):
	occs = set( a['title'] for a in attr_list )
	levels = set( a['level'] for a in attr_list )
	count_dict={}
	for occ in occs:
		subset = [a for a in attr_list if a['title']==occ ]
		for level in levels:
			count_dict[ (occ,level) ] = len( [ a for a in subset if a['level']==level ] )
	ret = np.zeros( tfidf_dict[counttype].shape )
	for i in range( ret.shape[0] ):
		correction_factor = 1.
		if correct_for_number_collected:
			correction_factor = count_dict[ ( attr_list[i]['title'] , attr_list[i]['level'] ) ]
		ret[i,:] = 1. * ( tfidf_dict[counttype][i,:] ).todense() / correction_factor
	return ret

def get_max_differences(tfidf_dict,level1,level2, attr_list, typename="count"):
	overall1 = np.array( tfidf_dict[typename][ [ k for k,o in enumerate(attr_list) if o['level']==level1 ] ,:].mean(axis=0) )[0]
	overall2 = np.array( tfidf_dict[typename][ [ k for k,o in enumerate(attr_list) if o['level']==level2 ] ,:].mean(axis=0) )[0]
	return sorted( [ ( v1-v2 , l )
				for l,v1,v2 in zip(tfidf_dict['labels'],overall1,overall2) ]
				, key=None, reverse=False)

def get_counts_from_corpus(corpus):
	ret = defaultdict(int)
	list_of_all_words = (' '.join(corpus)).split(' ')
	for word in list_of_all_words:
		ret[word]+=1
	return ret #


def get_words_adjacent(fun_input,subcorpus,pos):
	if isinstance(fun_input,str):
		return get_words_adjacent([fun_input],subcorpus,pos)
	list_of_all_words = (' '.join(subcorpus)).split(' ')
	subcorpus_word_counts=get_counts_from_corpus(subcorpus)
	word_count = len(list_of_all_words)
	words_ajacent = defaultdict(int)
	if isinstance(fun_input,list):
		check_len = len(fun_input)
		if pos > 0: pos = pos+check_len-1
		front_offset = (0 if pos > 0 else -pos)
		end_offset = (pos if pos > 0 else 0) + check_len - 1
		for k in range(front_offset, word_count - end_offset):
			if list_of_all_words[k:k+check_len]==fun_input:
				words_ajacent[ list_of_all_words [k+pos] ]+=1
	# corrected_words_after = { k : 
	# 					1.*v/subcorpus_word_counts[k]/subcorpus_word_counts[one_word]*word_count # lift for appearance after
	# 					for k,v in words_ajacent.items() }
	return sorted( [ (v,k) for k,v in words_ajacent.items() ] )

def select_corpus(corpus,attrname,attrval):
	ind_list = [ k for k,obj in enumerate(attr_list) if obj[attrname]==attrval ]
	return [corpus[k] for k in ind_list]

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
	raw_score = 1000.*raw_score/max( len(words), 200 )
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




if __name__ == '__main__':


	attr_list,corpus,sentences = process_all()
	# create_corpus_file(corpus)
	tfidf_dict = get_tfidf(corpus)

	tfidf_dict['corrected'] = scipy.sparse.csr_matrix( get_corrected_counts(tfidf_dict,attr_list, 
							counttype='tfidf',correct_for_number_collected=True) )
	tfidf_dict['exists'] = tfidf_dict['tfidf'] != 0

	result = get_max_differences(tfidf_dict,'Less than 1 year','More than 10 years',attr_list,typename='tfidf')

	result[:10]
	result[-10:-1]

	all_word_counts = get_counts_from_corpus(corpus)

	words_after = get_words_adjacent('communication',
								select_corpus(corpus,'level','Less than 1 year'),
								1)

	nov_scores,exp_scores = get_score_dicts()
	print( sum( nov_scores.values() ), sum(exp_scores.values()) )
	all_resume_scores = [get_raw_score(res) for res in corpus]
	scorer = create_scorer(all_resume_scores)
	in_order = sorted(all_resume_scores)
	num = len(in_order)

	sumanik = """I am working with the Data Science Team. Demand Media produces a lot of online content for various online brands like eHow.com, livestrong.com etc. So besides providing good quality content to users, we also need to provide related content to users for further improving their experience. My role in the team is to provide recommendations for content available on eHow.com, livestrong.com and cracked.com. Besides recommendation system, I am also working towards developing a model to automate the process of associating a relevant image to an article. 
	Currently I am working towards analysing the web crawl data available at commoncrawl.org to come up with some results that will further help us improve the content available on our websites."""
	clean_sumanik  = clean_text(sumanik)

	nov_res_scores = [score for score,attr in zip(all_resume_scores,attr_list) if attr['level']=="Less than 1 year"]
	exp_res_scores = [score for score,attr in zip(all_resume_scores,attr_list) if attr['level']=="More than 10 years"]

	mean(nov_res_scores)
	mean(exp_res_scores)

	all_adjusted_resume_scores = [ scorer(score) for score in all_resume_scores ]
	
	nov_res_lens = [len(res) for res,attr in zip(corpus,attr_list) if attr['level']=="Less than 1 year"]
	exp_res_lens = [len(res) for res,attr in zip(corpus,attr_list) if attr['level']=="More than 10 years"]

	res_lens = [len(res.split(' ')) for res in corpus]

	plt.hist(res_lens,bins=100)
	plt.show()

# with open('experienced_words.txt','w') as f:
# 	for score,word in result[:100]:
# 		f.write(word+'\t'+str(score)+'\n')

# with open('novice_words.txt','w') as f:
# 	for score,word in result[-100:-1]:
# 		f.write(word+'\t'+str(score)+'\n')




# model = word2vec.Word2Vec(sentences)
# model = word2vec.Word2Vec.load('word2vec_model')
# model.save('word2vec_model')


# def load_sample():
# 	with open(' ') as f:
# 		return (' '.join( clean_text( line ) for line in f ) )

# def load_glove_vec(n=100000):
# 	with open() as f:
# 		k=0
# 		while k<n:

# def one_fun():
# 	return 1

# def diag(word)
# 	kw_index = tfidf_dict['labels'].index(word)
# 	corpus_indices = tfidf_dict['tfidf'][:,kw_index].nonzero()[0]
# 	tfidf_dict['tfidf'][corpus_indices,kw_index].todense()
# 	for i in corpus_indices:
# 		print(corpus[i])

# ##################################################



########################


# def long_tail_tfidf(corpus):
# 	tfidf_vectorizer = TfidfVectorizer( max_features=20000, stop_words='english', ngram_range=(1, 2), min_df=5 )
# 	tfidf = tfidf_vectorizer.fit_transform( corpus )
# 	tfidf_labels = tfidf_vectorizer.get_feature_names()
# 	return {
# 		'tfidf':tfidf,
# 		'labels':tfidf_labels,
# 	}

# long_tfidf_dict = long_tail_tfidf(corpus)
# row_range = list(range( long_tfidf_dict['tfidf'].shape[0] ))
# norms=sparse_norm(long_tfidf_dict['tfidf'],axis=1)

# import random
# n_compare = 25
# n_words = 10
# sim_mat = np.dot( long_tfidf_dict['tfidf'] , long_tfidf_dict['tfidf'].transpose() )

# def find_missing_words():
# 	k = random.choice( row_range )
# 	print('Your resume:\n\n'+corpus[k])
# 	temp = np.array(sim_mat[:,k].todense().reshape(-1))
# 	indices = np.argsort(temp)[0][-2:-2-n_compare:-1]
# 	feature_sums = np.sum( long_tfidf_dict['tfidf'][indices,:].todense(), axis=0 ) / n_compare
# 	temp_row = long_tfidf_dict['tfidf'][k,:].todense()
# 	differences = feature_sums - temp_row
# 	word_indices = np.array(np.argsort(differences))[0][-2:-2-n_words:-1]
# 	res_score = score_resume(corpus[k])
# 	n_dashes = (res_score)//4
# 	print('\n0 Novice:'+'-'*n_dashes+' '+str(int(res_score))+' '+(25-n_dashes)*'-'+':Experienced 100')
# 	print('\n\nWords that appear more frequently in similar resumes:')
# 	print(', '.join( [long_tfidf_dict['labels'][k] for k in word_indices] ) )



