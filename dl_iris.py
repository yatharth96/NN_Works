import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.utils import np_utils
import numpy as np
from numpy import *
from sklearn.preprocessing import LabelEncoder
from random import shuffle
import random


def createDatasets():
	with open("iris.txt","r") as f:
		lines = f.readlines()

	indices = [0]*150
	train_indexes = random.sample(range(150),120)

	for i in train_indexes:
		indices[i] = 1

	training = []
	testing = []

	for i in range(len(indices)):
		if indices[i] == 0:
			
			testing.append(lines[i])
		else:
			training.append(lines[i])
	print(training)
	## Shuffling needed or not?
	shuffle(training)
	shuffle(testing)
	print(training)
	with open('training.txt','w') as f:
		f.write("".join(training))


	with open('testing.txt','w') as f:
		f.write("".join(testing))


#createDatasets()
X = np.genfromtxt("training.txt",delimiter = ",",usecols = (0,1,2,3))
Y = np.genfromtxt("training.txt",delimiter = ",",usecols = (4),dtype = None)

#print X
#print Y

encode = LabelEncoder()
encode.fit(Y)
encoded_Y = encode.transform(Y)

#print encoded_Y

Y_new = np_utils.to_categorical(encoded_Y)

#print Y_new


## KERAS

# Defining model
model = Sequential()
model.add(Dense(3,input_dim = 4, init = "uniform",activation = "sigmoid")) 	# only mention inp dim for first layer, rest figured out on own
model.add(Dense(3,init = "uniform",activation = "sigmoid"))

# Training model
model.compile(loss = "mse",optimizer = "sgd",metrics = ["accuracy"])
model.fit(X,Y_new,nb_epoch = 300,batch_size =1)


# Predicting with model
X_test = np.genfromtxt("testing.txt",delimiter = ",",usecols = (0,1,2,3))
Y_test = np.genfromtxt("testing.txt",delimiter = ",",usecols = (4),dtype = None) 


encode.fit(Y_test)
Y_test = encode.transform(Y_test)
Y_test = np_utils.to_categorical(Y_test)
predictions = model.predict(X_test)

accuracy = 0
i = 0 
for prediction in predictions:
	print("debug1:",prediction)
	print("debug2:",Y_test[i])
	print("debug3:",np.argmax(prediction))
	print("debug4:",np.argmax(Y_test[i]))
	if np.argmax(prediction)  == np.argmax(Y_test[i]):
		accuracy += 1/len(Y_test)
		i += 1

print("Accuracy:",accuracy)

