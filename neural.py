#!/usr/bin/python3

import pandas as pd
import numpy as np

from nltk.tokenize import word_tokenize
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM


def load_data():
	"""Load train and test data"""
	# sep = '\t'
	train = pd.read_csv("NLI2FOLI/SICK/SICK_train.txt", sep='\t')
	trial = pd.read_csv("NLI2FOLI/SICK/SICK_trial.txt", sep='\t')

	return train, trial


def load_embeddings():
	"""Load GloVe pre-trained word vectors"""
	glove_dict = {}
	with open("../glove.6B.100d.txt", 'r') as f:
		for line in f:
			values = line.split()
			word = values[0]
			vector = np.asarray(values[1:], "float32")
			glove_dict[word] = vector#

	return glove_dict


def create_embeddings(row, embeddings_dict):
	"""Create word embeddings for the sentences and return a 200d vector"""
	premise = word_tokenize(row['sentence_A'].lower())
	hypothesis = word_tokenize(row['sentence_B'].lower())

	premise_vectors = [embeddings_dict[w] for w in premise if w in embeddings_dict]
	premise_vec = sum(premise_vectors)

	hypothesis_vectors = [embeddings_dict[w] for w in hypothesis if w in embeddings_dict]
	hypothesis_vec = sum(hypothesis_vectors)

	concat_vec = np.concatenate((premise_vec, hypothesis_vec), axis=None)

	return concat_vec


def main():
	train, trial = load_data()
	embeddings_dict = load_embeddings()

	# Change labels to integers
	labels = {'ENTAILMENT':0, 'CONTRADICTION':1, 'NEUTRAL':2}
	train_labels = train['entailment_judgment'].map(labels).values
	trial_labels = trial['entailment_judgment'].map(labels).values

	# Get vectors and turn into array of arrays for Keras input
	train['embeddings'] = train.apply(lambda row: create_embeddings(row, embeddings_dict), axis=1) 
	train_vecs = np.stack(train.embeddings.values)
	train_vecs = np.reshape(train_vecs, (train_vecs.shape[0], 1, train_vecs.shape[1]))
	trial['embeddings'] = trial.apply(lambda row: create_embeddings(row, embeddings_dict), axis=1)
	trial_vecs = np.stack(trial.embeddings.values)
	trial_vecs = np.reshape(trial_vecs, (trial_vecs.shape[0], 1, trial_vecs.shape[1]))

	# Keras neural network model
	model = Sequential()
	model.add(LSTM(500, input_shape=((None, 200)), activation='tanh', recurrent_activation='hard_sigmoid'))
	model.add(Dropout(0.4))
	model.add(Dense(500, activation='relu'))
	model.add(Dropout(0.4))
	model.add(Dense(500, activation='tanh'))
	model.add(Dropout(0.4))
	model.add(Dense(500, activation='relu'))
	model.add(Dropout(0.4))
	model.add(Dense(500, activation='tanh'))
	model.add(Dropout(0.5))
	model.add(Dense(3, activation='softmax'))
	model.compile(optimizer=Adam(0.001, amsgrad=True), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

	model.fit(train_vecs, train_labels, epochs=50, batch_size=64)
	scores = model.evaluate(trial_vecs, trial_labels, verbose=0)

	print("Accuracy: %.2f%%" % (scores[1]*100))

if __name__ == '__main__':
	main()