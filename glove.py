import numpy as np
from gensim.models import word2vec 

with open('experienced_words.txt') as f:
	exp_words = [line.strip().split('\t')[0] for line in f]

with open('novice_words.txt') as f:
	nov_words = [line.strip().split('\t')[0] for line in f]

all_words = set(exp_words) | set(nov_words)

vector_dict = {}
with open('C:\Users\matth\OneDrive\Desktop\Data\glove.6B.200d.txt') as f:
	counter = 0
	while counter<300000:
		if counter%10000==0: print(counter)
		counter+=1
		line = next(f)
		elements = line.strip().split(' ')
		word = elements[0]
		if word not in all_words:
			continue
		array = np.array( [float(n) for n in elements[1:]] )
		array = array/np.linalg.norm(array)
		vector_dict[word] = array
		if len(vector_dict) == len(all_words):
			break

from gensim.models import word2vec
model = word2vec.Word2Vec.load('word2vec_model')
vector_dict = {word:model.wv[word] for word in all_words}

[word for word in exp_words if word not in vector_dict]
[word for word in nov_words if word not in vector_dict]

exp_mat = np.array( [vector_dict[word] for word in exp_words] )
nov_mat = np.array( [vector_dict[word] for word in nov_words] )

sims = np.dot( exp_mat, nov_mat.transpose() )
for k,word in enumerate(nov_words):
	temp_sim = sims[:,k]
	indices = list(np.argsort(temp_sim)[::-1])
	top_n_words = [exp_words[n] for n in indices[:5]]
	print(word+' -> '+', '.join(top_n_words))