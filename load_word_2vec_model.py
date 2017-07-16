from gensim.models import Word2Vec
import numpy as np

model = Word2Vec.load('/home/rathee/projects/quora_similar_questions/data/300features_40minwords_10context_new')

def get_model_embeddings () :
	w2v = np.zeros((len(model.wv.index2word) + 1, 300))
	model_dict = {}
	index_to_word = {}
	w2v[0] = np.zeros(300)
	for i,word in enumerate(model.wv.index2word):
		w2v[i+1] = model[word]
		model_dict[word] = i + 1
		index_to_word[i+1] = word

	return w2v, model_dict, index_to_word

#w2v,model_dict, index_to_word = get_model_embeddings()
#print len(w2v)
