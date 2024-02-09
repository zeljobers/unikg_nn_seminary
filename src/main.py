#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 21:20:51 2023

@author: zorin
"""

from csv import reader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import colorcet as cc
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler, LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report

def load_csv(filename):
	dataset = list()
	with open(filename, 'r') as file:
		csv_reader = reader(file)
		for row in csv_reader:
			if not row:
				continue
			dataset.append(row)
	return dataset

def str_column_to_float(dataset, column):
	for row in dataset:
		row[column] = float(row[column].strip())
		
def str_column_to_int(dataset, column):
	class_values = [row[column] for row in dataset]
	unique = set(class_values)
	lookup = dict()
	for i, value in enumerate(unique):
		lookup[value] = i
	for row in dataset:
		row[column] = lookup[row[column]]
	return lookup


def visualize(dataset):
	palette = sns.color_palette(cc.glasbey, n_colors=26)
	sns.pairplot(dataset, hue="lettr", palette=palette)
	plt.show()

def visualize_xbox_density(dataset):
	sns.displot(dataset, 
			     hue="lettr", 
				 palette="Spectral",
				 x=dataset['X-box'],
				 kind='kde',
				 fill=True)
	plt.show()


def visualize_xbox_ybox_correlation(dataset):
		sns.relplot(dataset, sizes=(0, 17), palette="Spectral", x="X-box", y="Y-box", hue="lettr")
		plt.show()



def baseline_model(optimizer='Nadam', units=10, init='uniform', activation='relu'):
	model = Sequential()
	model.add(Dense(units, input_dim = 16, kernel_initializer=init, activation = activation))
	model.add(Dropout(0.2))
	model.add(Dense(units, kernel_initializer=init, activation = activation))
	model.add(Dropout(0.2))
	model.add(Dense(26, kernel_initializer=init, activation = 'softmax'))
	model.compile(loss = 'categorical_crossentropy', 
			      optimizer = optimizer, 
				  metrics = ['accuracy'])
	return model

def evaluate_statistic_metrics_custom(x, y):
	
	x_train, x_test, y_train, y_test = train_test_split(
	 	x,y, 
	 	test_size = 0.2,
	 	random_state = 101)
	
	model = KerasClassifier(build_fn = baseline_model,
	 							epochs = 2000,
	 							batch_size = 256,
	 							verbose = 1
	 							)
	
	optimizers = ['Nadam']
	
	activations = ['tanh']
	
	init = ['normal']
	param_grid = {
	    'units': [15],
	    'init': init,
	    'activation': activations,
	    'optimizer' : optimizers
	}
	
	grid = GridSearchCV(estimator=model, 
						param_grid=param_grid, 
	 					cv= 2
						)
	grid_result = grid.fit(x_train, y_train, 
						   validation_data = (x_test, y_test))
	print("Best Parameters: ", grid_result.best_params_)
	print("Best Score: ", grid_result.best_score_)
	
	history = grid.best_estimator_.model.history
	
	plt.plot(history.history['accuracy'])
	plt.plot(history.history['val_accuracy'])
	plt.title('Accuracy')
	plt.legend(['Train', 'validation'])
	plt.show()
	
	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title('Loss')
	plt.legend(['Train', 'validation'])
	plt.show()
	
	
	predict_train = grid_result.predict(x_train)
	print(classification_report(y_train.argmax(axis=1),
	 							predict_train))
	
	cm_train = confusion_matrix(y_train.argmax(axis=1),
	 							predict_train)
	
	fig, ax = plt.subplots(figsize=(26,26))
	ax = sns.heatmap(cm_train, annot = True, cmap = 'Blues', cbar=False, linewidths=.5, ax=ax)
	ax.set_xlabel('Predicted Values')
	ax.set_ylabel('Actual Values')
	plt.show()
	
	predict_test = grid_result.predict(x_test)
	print(classification_report(y_test.argmax(axis=1),
	 							predict_test))

	cm_test = confusion_matrix(y_test.argmax(axis=1),
	 							predict_test)

	fig, ax = plt.subplots(figsize=(26,26))
	ax = sns.heatmap(cm_test, annot = True, cmap = 'Blues', cbar=False, linewidths=.5, ax=ax)
	ax.set_xlabel('Predicted Values')
	ax.set_ylabel('Actual Values')
	plt.show()



filename = 'letter.csv'
dataset =  load_csv(filename)



for i in range(len(dataset[0]) - 1):
 	str_column_to_float(dataset, i)


dataset = pd.DataFrame(dataset, columns=['X-box', 'Y-box', 'Width', 'High', 'Onpix', 'X-bar', 'Y-bar', 'X2bar', 'Y2bar', 'Xybar', 'X2ybr', 'Xy2br', 'X-ege', 'Xegvy', 'Y-ege', 'Yegvx', 'lettr'])

y = dataset.iloc[:, -1]

visualize(dataset)
visualize_xbox_density(dataset)
visualize_xbox_ybox_correlation(dataset)


encoder = LabelEncoder()
encoder.fit(y)
encoded_Y = encoder.transform(y)
dummy_Y = np_utils.to_categorical(encoded_Y)

df_norm = dataset[dataset.columns[0:-1]].apply(
		lambda x : (x - x.min()) / (x.max() - x.min())
 	)


x = df_norm
y = dummy_Y

# evaluate_statistic_metrics_custom(x,y)

x_train, x_test, y_train, y_test = train_test_split(
 	x,y, 
 	test_size = 0.2,
 	random_state = 101)

model = KerasClassifier(build_fn = baseline_model,
 							epochs = 50,
 							batch_size = 256,
 							verbose = 1
 							)

kfold = KFold(n_splits = 3,
 			  shuffle = True)


optimizers = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']

activations = ['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']

init = ['normal']
param_grid = {
    'units': [5, 10, 15],
    'init': init,
    'activation': activations,
    'optimizer' : optimizers
}

grid = GridSearchCV(estimator=model, 
 					param_grid=param_grid, 
 					cv= kfold
 					)
grid_result = grid.fit(x_train, y_train, 
					   validation_data = (x_test, y_test))
print("Best Parameters: ", grid_result.best_params_)
print("Best Score: ", grid_result.best_score_)

history = grid.best_estimator_.model.history

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Accuracy')
plt.legend(['Train', 'validation'])
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Loss')
plt.legend(['Train', 'validation'])
plt.show()


predict_train = grid_result.predict(x_train)
print(classification_report(y_train.argmax(axis=1),
 							predict_train))

cm_train = confusion_matrix(y_train.argmax(axis=1),
 							predict_train)

fig, ax = plt.subplots(figsize=(26,26))
ax = sns.heatmap(cm_train, annot = True, cmap = 'Blues', cbar=False, linewidths=.5, ax=ax)
ax.set_xlabel('Predicted Values')
ax.set_ylabel('Actual Values')
plt.show()

predict_test = grid_result.predict(x_test)
print(classification_report(y_test.argmax(axis=1),
 							predict_test))

cm_test = confusion_matrix(y_test.argmax(axis=1),
 							predict_test)

fig, ax = plt.subplots(figsize=(26,26))
ax = sns.heatmap(cm_test, annot = True, cmap = 'Blues', cbar=False, linewidths=.5, ax=ax)
ax.set_xlabel('Predicted Values')
ax.set_ylabel('Actual Values')
plt.show()