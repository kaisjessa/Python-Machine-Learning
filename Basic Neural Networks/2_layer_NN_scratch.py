#Tutorial from: http://iamtrask.github.io/2015/07/12/basic-python-network/

import numpy as np

#3 inputs, 1 output NN

#define the activation function
def sigmoid(x, derivative=False):
	if(derivative):
		return x*(1-x)
	else:
		return np.exp(x)/(1+np.exp(x))

#input data as a numpy matrix
#each row is a training example
#each column is one input node

X = np.array([
	[0,0,1],
	[0,1,1],
	[1,0,1],
	[1,1,1]
	])
#output data
#.T => transpose (1 row 4 columns => 4 rows 1 column)
#each row corresponds to the correct output of a row in X
y = np.array([[0,0,1,1]]).T

#constant random distribution
np.random.seed(1)

#3x1 matrix of weights, because of 3 inputs and 1 output
weights_0 = 2*np.random.random((3,1))-1
print("Weights:\n", weights_0)

for i in range(10000):

	#first layer = input dataset
	layer_0 = X
	#print("Layer 0:\n", layer_0)

	#second layer = matrix mutliplication of inputs and weights
	layer_1 = np.dot(layer_0, weights_0)
	#print("Layer 1:\n", layer_1)

	#apply the activation function on layer_1 so 0 < output < 1
	layer_1 = sigmoid(layer_1)
	#print("New Layer 1:\n", layer_1)

	#calculate the error in layer_1 using the difference between the correct outputs and the predictions
	l1_error = y - layer_1
	#print("Error:\n", l1_error)

	#use the slope of the sigmoid function to increase the magnitude of change for low-confidence (close to 0) predictions
	#use the error value to determine the "direction" of change and to weight the change
	#A large error increases the magnitude of the change
	#A low-confidence guess also increases the magnitude of the change
	#"Error weighted derivative"
	l1_delta = l1_error * sigmoid(layer_1, derivative=True)

	#Update the weight values using the matmul of l1_delta and layer_0
	#In other words, update each weight value by the product of its input and its delta (change)
	weights_0 += np.dot(layer_0.T, l1_delta)

	#This is simliar to the delta rule
	#l1_delta = l1_error * sigmoid'(layer_1) * layer_0.T


	#print("Updated Weights:\n", weights_0)
	if(i%1000==0):
		print("Current output:\n", layer_1)
		print("Deriv:\n", sigmoid(layer_1, derivative=True))

print("Results after training:\n", layer_1)
print("Correct output:\n", y)

#Should print ~1
print(float(sigmoid(np.dot([1,1,0], weights_0))))





