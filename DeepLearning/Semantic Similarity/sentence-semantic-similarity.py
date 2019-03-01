# -*- coding: utf-8 -*-
"""
@author: Sreenivas.J
"""
#http://alt.qcri.org/semeval2017/
#dataset: http://alt.qcri.org/semeval2017/task1/
from keras.layers import Dense, Dropout, concatenate, Input
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.models import Model

from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

import numpy as np
import os

#pip install nltk
#import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
import re #Regular Expression
#from sklearn.metrics import mean_squared_error

train_dir = 'D:/Data Science/Data/sentense semantics similarity/'
test_dir  = 'D:/Data Science/Data/sentense semantics similarity/'

#Global Vectors for Word Representation
glove_file = 'D:/Data Science/deeplearning/Glove.6B/glove.6B.50d.txt'
word_embed_size = 50 #Should be equial to Glove dimension 50/100/200/300
batch_size = 2
epochs = 5

def load_data(filepath):
    sent_left, sent_right, scores = [], [], []
    fsent = open(filepath,encoding='utf8')
    for line in fsent:
        #Separate/extract it by termination charecter
        left, right, score = line.strip().split("\t")
        sent_left.append(left)
        sent_right.append(right)
        scores.append(float(score))
    fsent.close()
    return sent_left, sent_right, scores

def cleanSentence(sentence):
     print("Actual Sentence: " + sentence)
     sentence_clean= re.sub("[^a-zA-Z]"," ", sentence)
     print(("\t"))
     print("Clean sentence: " + sentence_clean)
     sentence_clean = sentence_clean.lower()
     tokens = word_tokenize(sentence_clean)
     print(("\t"))
     print("Tokens: ")
     print(tokens)
     stop_words = set(stopwords.words("english"))
     print(("\t"))
     print("Stop Words: ")
     print(stop_words)
     sentence_clean_words = [w for w in tokens if not w in stop_words]
     print(("\t"))
     print("Sentence Clean Words: ")
     print(sentence_clean_words)
     print(("\t"))    
     print(("_________________________"))  
     return ' '.join(sentence_clean_words)
 
def buildVocabulary(sentence_left, sentence_right):
    text = list(set(sentence_left).union(sentence_right))
    print("Combined Union Text: ")
    print(text)
    print(("\t"))
    tokenizer = Tokenizer(lower=False, split=' ')
    print("Tokenizer: ")
    print(tokenizer)
    print(("\t"))    
    tokenizer.fit_on_texts(text)
    print(tokenizer)
    print(("\t"))    
    return tokenizer

def getTrainSequences(sentence_left, sentence_right, tokenizer):
    sent_left = tokenizer.texts_to_sequences(sentence_left)
    sent_right = tokenizer.texts_to_sequences(sentence_right)
    #Get the maximum length of the sentence so that all sentences can be labelled with same dimension
    sent_left_maxlen = max([len(s) for s in sent_left])
    sent_right_maxlen = max([len(s) for s in sent_right])
    seq_maxlen = max([sent_left_maxlen, sent_right_maxlen])
    #Pad with Zeros in empty spaces
    return np.array(pad_sequences(sent_left, maxlen=seq_maxlen)), np.array(pad_sequences(sent_right, maxlen=seq_maxlen))

def getTestSequences(sentence_left, sentence_right, tokenizer, seq_maxlen):
    sent_left = tokenizer.texts_to_sequences(sentence_left)
    sent_right = tokenizer.texts_to_sequences(sentence_right)
    return np.array(pad_sequences(sent_left, maxlen=seq_maxlen)), np.array(pad_sequences(sent_right, maxlen=seq_maxlen))
  

     
def loadGloveWordEmbeddings(glove_file):
    embedding_vectors = {}
    f = open(glove_file,encoding='utf8')
    for line in f:
        values = line.split()
        word = values[0]
        value = np.asarray(values[1:], dtype='float32')
        embedding_vectors[word] = value
    f.close()
    return embedding_vectors

def getEmbeddingWeightMatrix(embedding_vectors, word2idx): 
    #word_embed_size is Glove file dimension size
    word_embed_size = 50
    embedding_matrix = np.zeros((len(word2idx)+1, word_embed_size))
    for word, i in word2idx.items():
        embedding_vector = embedding_vectors.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix

#Load the train data from train file
#Left sentence and Right sentence
sentence_left_train, sentence_right_train, scores_train = load_data(os.path.join(train_dir, 'train.txt'))
print(len(sentence_left_train)) #2234 sentences are there in the train file
print(len(sentence_right_train)) 
print(len(scores_train)) 

#Make sure both left and right side has equal number of statements to compare
assert len(sentence_left_train) == len(sentence_right_train) and len(sentence_right_train) == len(scores_train)

#nltk.download('punkt')
#nltk.download('stopwords') #Some set of words which are not necessary. Like didn't

#clean the sentence pairs
sentence_left_train1 = list(map(cleanSentence, sentence_left_train))
sentence_right_train1 = list(map(cleanSentence, sentence_right_train))
assert len(sentence_left_train1) == len(sentence_right_train1)

#build vocabulary over all sentence pairs
#Tokenize means, assigning occurance ranking
tokenizer = buildVocabulary(sentence_left_train1, sentence_right_train1)
vocab_size = len(tokenizer.word_index) + 1
print(tokenizer.word_index)
print(vocab_size) #Unique words among all sentences

#get sequences for each sentence pair tuple
Xtrain_left, Xtrain_right = getTrainSequences(sentence_left_train1, sentence_right_train1, tokenizer)
ytrain = np.array(scores_train)
seq_maxlen = len(Xtrain_left[0])
print(seq_maxlen)

#load pre-trained word embeddings
embedding_vectors = loadGloveWordEmbeddings(glove_file)
print(len(embedding_vectors))

#get embedding layer weight matrix
#embedding_vectors: Vectors loaded from Glove file
#tokenizer.word_index: Each and every unique word from train file with the sequence
embedding_weight_matrix = getEmbeddingWeightMatrix(embedding_vectors, tokenizer.word_index)
#print(len(embedding_vectors.get("said")))
print(embedding_weight_matrix.shape) #Each word will be reprsented by a (50/100/200/300d) Glove dimensional weight matrix

#build model        
#vocab_size: 45
#word_embed_size: Glove file dimension (50/100/200/300)
#seq_maxlen: 16
#weights: Loaded Weight matrix
#left_input: 16
#Now build a Embedding layer
left_input = Input(shape=(Xtrain_left.shape[1],),dtype='int32')
#Build an Embedding layer
#Finally, we do not want to update the learned word weights in this model, 
#therefore we will set the trainable attribute for the model to be False.
left = Embedding(input_dim=vocab_size, output_dim=word_embed_size, 
                   input_length=seq_maxlen, weights=[embedding_weight_matrix], 
                   trainable = False) (left_input)
print(left_input)
left = LSTM(50, return_sequences=False)(left)
left = Dropout(0.1)(left) #To control over fitting

#Now do the same for right side as well
right_input = Input(shape=(Xtrain_right.shape[1],),dtype='int32')

right = Embedding(input_dim=vocab_size, output_dim=word_embed_size, 
                   input_length=seq_maxlen, weights=[embedding_weight_matrix], 
                   trainable = False) (right_input)
right = LSTM(50, return_sequences=False)(right)
right = Dropout(0.1)(right)

x = concatenate([left, right])
x = Dense(10, activation='relu')(x)
output = Dense(1)(x) #Tells the similarity

#Build the model
model = Model(inputs=[left_input, right_input], outputs=output)
print(model.summary())

model.compile(optimizer="sgd", loss="mean_squared_error")

model.fit([Xtrain_left, Xtrain_right], ytrain, batch_size=batch_size,
          epochs=epochs, validation_split=0.2)

#plot_loss(history)

#predict the sentence similarity on test data
sentence_left_test, sentence_right_test, scores_test = load_data(os.path.join(test_dir, 'test.txt'))

sentence_left_test1 = list(map(cleanSentence, sentence_left_test))
sentence_right_test1 = list(map(cleanSentence, sentence_right_test))

Xtest_left, Xtest_right = getTestSequences(sentence_left_test1, sentence_right_test1, tokenizer, seq_maxlen)
ytest = np.array(scores_test, dtype='float')

ytest_GV = model.predict([Xtest_left, Xtest_right])[:, 0]

print(ytest)  #ytest is systems score from test file
print(ytest_GV) #ytest_ is our DL predicted score on same test file
#Paerson correlation between original test and human test
#print(np.corrcoef(ytest, ytest_GV))
#print(mean_squared_error(ytest, ytest_GV))
