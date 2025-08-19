"""

To run, copy and paste into your terminal:
python .\multi_layer_perceptron.py


To test:
Change values of INITIAL_INPUTS and WEIGHTS_MATRIX

"""


import math, random


INITIAL_INPUTS = [1, 0, 1, 0, 1]


WEIGHTS_MATRIX = [
    [0.7, 0.6, 0.5, 0.3, 0.4],  # neuron 1
    [0.2, 0.8, 0.1, 0.5, 0.9],  # neuron 2
    [0.9, 0.4, 0.6, 0.7, 0.3],  # neuron 3
    [0.1, 0.3, 0.8, 0.6, 0.2],  # neuron 4 
    [0.5, 0.7, 0.2, 0.9, 0.4]   # neuron 5 
]

"""
 Generates a list of Random Biases based on how many neurons are going to be 
 generated in the layer_perceptron function.

 Number of layers to be generated can be referenced in the WEIGHTS_MATRIX.
"""
BIASES = [random.uniform(0, 0.2) for _ in range(len(WEIGHTS_MATRIX))]


# The core calculation for every neuron
def perceptron_calc(inputs, weights, bias):
    output = 0

    # Basically a summation:
    for i in range(len(inputs)):
        output += inputs[i] * weights[i]

    output += bias

    return output


# Sigmoid Activation Function (Better than using Step Activation Function)
def sigmoid(x):
    return 1 / (1 + math.exp(-x))



# One perceptron Calculation.
def perceptron(inputs, weights, bias):
    weighted_sum = perceptron_calc(inputs, weights, bias)
    return sigmoid(weighted_sum)



# Multi-Layer Perceptron Calculation
def layer_perceptron(inputs):
    outputs = []
    # Iterate through each neuron in the layer. Unzip weights and biases.
    for weight, bias in zip(WEIGHTS_MATRIX, BIASES):
        output = perceptron_calc(inputs, weight, bias)
        outputs.append(sigmoid(output))
    return outputs



print("Single Perceptron: ", perceptron(INITIAL_INPUTS, WEIGHTS_MATRIX[0], BIASES[0]))
print("\nMLP (Multi-level Perceptron): 5 neurons, 1 hidden layer \n", layer_perceptron(INITIAL_INPUTS))


