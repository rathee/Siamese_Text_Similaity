import numpy as np
import re
import itertools
from collections import Counter
import numpy as np
import time
import gc
from tensorflow.contrib import learn
from gensim.models.word2vec import Word2Vec
import gzip
from random import random
from preprocess import MyVocabularyProcessor
import sys
reload(sys)
sys.setdefaultencoding("utf-8")

from nltk.corpus import stopwords
import re
import csv
import sys
from gensim.parsing import PorterStemmer
global_stemmer = PorterStemmer()

stops = set(stopwords.words("english"))

def clean_sentence(sentence) :
	#review_text = re.sub("[^a-zA-Z0-9]"," ", sentence)
	review_text = text_to_wordlist(sentence)
	words = review_text.lower().split()
	words = [w for w in words if not w in stops]

	stemmed_words = []

	for w in words :
		stemmed = global_stemmer.stem(w)
		stemmed_words.append(stemmed)
	
	return stemmed_words


def text_to_wordlist(text):
    # Clean the text, with the option to remove stopwords and to stem words.
    
    # Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " is", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " n not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"60k", " 60000 ", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)

    return text
class InputHelper(object):
    
    def getTsvData(self, filepath):
        print("Loading training data from "+filepath)
        x1=[]
        x2=[]
        y=[]
        for line in open(filepath):
            l=line.strip().split("\t")
            s1 = clean_sentence(l[0].strip().lower())
            s2 = clean_sentence(l[1].strip().lower())
            if len(l)<3:
                continue
            if random() > 0.5:
               x1.append(s1)
               x2.append(s2)
            else:
               x1.append(s1)
               x2.append(s2)
            y.append(int(l[2].strip()))#np.array([0,1]))
        print len(x1), len(x2), len(y)
        return np.asarray(x1),np.asarray(x2),np.asarray(y)


    def getTsvTestData(self, filepath):
        x1=[]
        x2=[]
        y=[]
        count = 0
        for line in open(filepath):
            l=line.strip().split("\t")
            if len(l) != 3 :
               x1.append('india great country')
               x2.append('tax system wood sport')
               y.append(count)
               count += 1
               continue
            count += 1
            s1 = clean_sentence(l[0].strip().lower())
            s2 = clean_sentence(l[1].strip().lower())
            x1.append(s1)
            x2.append(s2)
            y.append(str(l[2].strip()))#np.array([0,1]))
        return np.asarray(x1),np.asarray(x2),np.asarray(y)
 
    def batch_iter(self, data, batch_size, num_epochs, shuffle=True):
        """
        Generates a batch iterator for a dataset.
        """
        data = np.asarray(data)
        print(data)
        print(data.shape)
        data_size = len(data)
        num_batches_per_epoch = int(len(data)/batch_size) + 1
        for epoch in range(num_epochs):
            # Shuffle the data at each epoch
            if shuffle:
                shuffle_indices = np.random.permutation(np.arange(data_size))
                shuffled_data = data[shuffle_indices]
            else:
                shuffled_data = data
            for batch_num in range(num_batches_per_epoch):
                start_index = batch_num * batch_size
                end_index = min((batch_num + 1) * batch_size, data_size)
                yield shuffled_data[start_index:end_index]
                
    def dumpValidation(self,x1_text,x2_text,y,shuffled_index,dev_idx,i):
        print("dumping validation "+str(i))
        x1_shuffled=x1_text[shuffled_index]
        x2_shuffled=x2_text[shuffled_index]
        y_shuffled=y[shuffled_index]
        x1_dev=x1_shuffled[dev_idx:]
        x2_dev=x2_shuffled[dev_idx:]
        y_dev=y_shuffled[dev_idx:]
        del x1_shuffled
        del y_shuffled
        with open('validation.txt'+str(i),'w') as f:
            for text1,text2,label in zip(x1_dev,x2_dev,y_dev):
                f.write(str(label)+"\t"+text1+"\t"+text2+"\n")
            f.close()
        del x1_dev
        del y_dev
    
    # Data Preparatopn
    # ==================================================
    
    
    def getDataSets(self, training_paths, max_document_length, percent_dev, batch_size, model_dict):
        x1_text, x2_text, y=self.getTsvData(training_paths)
       
        print len(x1_text), len(x2_text)
        # Build vocabulary
        i1=0
        train_set=[]
        dev_set=[]
        sum_no_of_batches = 0
        x1 = self.convert_all_sentences_to_indexes(model_dict, x1_text, max_document_length)
        x2 = self.convert_all_sentences_to_indexes(model_dict, x2_text, max_document_length)
        print len(x1), len(x2)
# Randomly shuffle data
        np.random.seed(131)
        shuffle_indices = np.random.permutation(np.arange(len(y)))
        x1_shuffled = x1[shuffle_indices]
        x2_shuffled = x2[shuffle_indices]
        y_shuffled = y[shuffle_indices]
        dev_idx = -1*len(y_shuffled)*percent_dev//100
        del x1
        del x2
        # Split train/test set
		#self.dumpValidation(x1_text,x2_text,y,shuffle_indices,dev_idx,0)
        # TODO: This is very crude, should use cross-validation
        x1_train, x1_dev = x1_shuffled[:dev_idx], x1_shuffled[dev_idx:]
        x2_train, x2_dev = x2_shuffled[:dev_idx], x2_shuffled[dev_idx:]
        y_train, y_dev = y_shuffled[:dev_idx], y_shuffled[dev_idx:]
        print("Train/Dev split for {}: {:d}/{:d}".format(training_paths, len(y_train), len(y_dev)))
        sum_no_of_batches = sum_no_of_batches+(len(y_train)//batch_size)
        train_set=(x1_train,x2_train,y_train)
        dev_set=(x1_dev,x2_dev,y_dev)
        gc.collect()
        return train_set,dev_set,sum_no_of_batches
    
    def getTestDataSet(self, data_path, max_document_length, model_dict):
        x1_text,x2_text,ids = self.getTsvTestData(data_path)

        x1 = self.convert_all_sentences_to_indexes(model_dict, x1_text, max_document_length)
        x2 = self.convert_all_sentences_to_indexes(model_dict, x2_text, max_document_length)
        # Build vocabulary
        return x1,x2, ids
    def convert_words_to_indexes(self,model_dict, sentence, sentence_length ) :
			indexes = []
			for word in sentence :
				index = model_dict.get(word,0) 
				indexes.append(index)
			for i in range(len(indexes), sentence_length) :
				indexes.append(0)
			return indexes[0:sentence_length]
    
    def convert_all_sentences_to_indexes (self, model_dict, sentences, sentence_length) :
			sentences_indexes = []

			for sentence in sentences :
					indexes = self.convert_words_to_indexes(model_dict, sentence, sentence_length)
					sentences_indexes.append(indexes)
			print len(sentences_indexes)
			return np.asarray(sentences_indexes)


