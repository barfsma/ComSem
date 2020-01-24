#!/usr/bin/python3

import argparse
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import accuracy_score
from keras.layers import Dense, Dropout, LSTM, LSTMCell
from keras.models import Sequential
from keras.optimizers import Adam
from nltk.tokenize import word_tokenize


def parse_arguments():
	"""Read arguments from a command line"""
	parser = argparse.ArgumentParser(description='Read the SICK dataset files')
	parser.add_argument(
    '--sick', metavar='FILE', required=True,
        help='File containing SICK inference problems')
	parser.add_argument(
    '--out', metavar='FILE',
        help='File where predictions will be written')
	args = parser.parse_args()

	return args


def load_data(test_path):
	"""Load train and test data"""
	# sep = '\t'
	train = pd.read_csv("NLI2FOLI/SICK/SICK_train.txt", sep='\t')
	test = pd.read_csv(test_path, sep='\t')

	return train, test


def load_embeddings():
	"""Load GloVe pre-trained word vectors"""
	glove_dict = {}
	with open("../glove.42B.300d.txt", 'r') as f:
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
	args = parse_arguments()
	train, test = load_data(args.sick)
	embeddings_dict = load_embeddings()

	np.random.seed(2)
	tf.random.set_seed(2)

	# Change labels to integers
	labels = {'ENTAILMENT':0, 'CONTRADICTION':1, 'NEUTRAL':2}
	train_labels = train['entailment_judgment'].map(labels).values
	if 'entailment_judgment' in test.columns:
		test_labels = test['entailment_judgment'].map(labels).values

	# Get vectors and turn into array of arrays for Keras input
	train['embeddings'] = train.apply(lambda row: create_embeddings(row, embeddings_dict), axis=1) 
	train_vecs = np.stack(train.embeddings.values)
	train_vecs = np.reshape(train_vecs, (train_vecs.shape[0], 1, train_vecs.shape[1]))
	test['embeddings'] = test.apply(lambda row: create_embeddings(row, embeddings_dict), axis=1)
	test_vecs = np.stack(test.embeddings.values)
	test_vecs = np.reshape(test_vecs, (test_vecs.shape[0], 1, test_vecs.shape[1]))

	# Keras neural network model
	model = Sequential()
	model.add(LSTM(600, input_shape=((1, 600)), activation='tanh', recurrent_activation='hard_sigmoid'))
	model.add(Dropout(0.4))
	model.add(Dense(600, activation='relu'))
	model.add(Dropout(0.4))
	model.add(Dense(600, activation='tanh'))
	model.add(Dropout(0.4))
	model.add(Dense(600, activation='relu'))
	model.add(Dropout(0.4))
	model.add(Dense(600, activation='tanh'))
	model.add(Dropout(0.5))
	model.add(Dense(3, activation='softmax'))
	model.compile(optimizer=Adam(0.001, amsgrad=True), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

	model.fit(train_vecs, train_labels, epochs=200, batch_size=64)
	prediction = model.predict(test_vecs, batch_size=64)
	pred_classes = np.argmax(prediction, axis=1)

	# Write IDs and predictions to output file
	if args.out:
		with open(args.out, 'w') as f:
			mapping = {0:'ENTAILMENT', 1:'CONTRADICTION', 2:'NEUTRAL'}
			ids = test['pair_ID'].tolist()
			for pid, pred in zip(ids, pred_classes):
				f.write("{}:{}\n".format(pid,mapping[pred]))

if __name__ == '__main__':
	main()