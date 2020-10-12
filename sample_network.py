import numpy as np;

def sigmoid(x):
	return 1/(1 + np.exp(-x))

# gradianet
def sigmoid_derivative(x):
	return x * (1-x)

cycles = 20000;

training_inputs = np.array([
	[0,0,1],
	[1,1,1],
	[1,0,1],
	[0,0,1],
	]);

testing_inputs = np.array([[1,0,0]])

labeled_outputs =  np.array([[0,1,1,0]]).T

np.random.seed(1)


# 3 by 1 matrix
# initial weight value
weights = 2 * np.random.random((3,1)) - 1

print('starting weights')
print(weights)

# training starts

for iteration in range(cycles):

	input_layer = training_inputs

	actual_outputs = sigmoid(np.dot(input_layer,weights))

	#backpropergation
	# error function 'error * input * gradient'
	error = labeled_outputs - actual_outputs

	adjustment = error * sigmoid_derivative(actual_outputs)

	# changing weights based on error(from backpropergation)
	weights += np.dot(input_layer.T, adjustment);

# training ends

print('cycles')
print(cycles)

print('trained weights')
print(weights)

print('outputs after training')
print(actual_outputs)

print('testing the network')

test_output = 	sigmoid(np.dot(testing_inputs,weights))

print(test_output)
