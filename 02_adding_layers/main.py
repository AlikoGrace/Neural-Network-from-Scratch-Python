import numpy as np

"""
2 layers
Layer 1: inputs → hidden
Layer 2: hidden → output
but in the case of the book, it's hidden 2 hidden layers 

 4 Neurons    ->     3 Neurons        ->     3 Neurons

 Sample features (4)                Hidden layer (3)                 Output layer (3)
x1 ──┐                              o───┐                           o───┐
x2 ──┼─── W1 (3×4), b1 (3) ───►     o───┼── W2 (3×3), b2 (3) ───►   o───●  y1
x3 ──┼--------------------------►    o───┘                      ►    o───●  y2
x4 ──┘                                                             ►  o───●  y3

"""

inputs = [[1.0, 2.0, 3.0, 2.5], 
          [2.0, 5.0, -1.0, 2.0],
          [-1.5, 2.7, 3.3, -0.8]] 

weights = [[0.2, 0.8, -0.5, 1.0], 
           [0.5, -0.91, 0.26, -0.5], 
           [-0.26, -0.27, 0.17, 0.87]] 

biases = [2, 3, 0.5] 

layer1_outputs=np.dot(inputs,np.array(weights).T)+biases
print(layer1_outputs) 

""" 
Hidden Layer 1 Outputs:

[[ 4.8    1.21   2.385] # Neuron 1 in Hidden Layer 2 will be receiving output data from 3 neurons in Hidden Layer 1
[ 8.9   -1.81   0.2  ] # Neuron 2 in Hidden Layer 2 will be receiving output data from 3 neurons in Hidden Layer 1
[ 1.41   1.051  0.026]] # Neuron 3 in Hidden Layer 2 will be receiving output data from 3 neurons in Hidden Layer 1
"""

weights2 = [[0.1, -0.14, 0.5], # Neuron 1 in Hidden Layer 2 weights 
            [-0.5, 0.12, -0.33], # Neuron 2 in Hidden Layer 2 weights
            [-0.44, 0.73, -0.13]] # Neuron 3 in Hidden Layer 2 weights

biases2 = [-1, 2, -0.5] # Neurons 1 -> 3 in Hidden Layer 2 biases

layer2_outputs = np.dot(layer1_outputs, np.array(weights2).T) + biases2
print(layer2_outputs)

""" 
Hidden Layer 2 Outputs:

[[ 0.5031  -1.04185 -2.03875]
[ 0.2434  -2.7332  -5.7633 ]
[-0.99314  1.41254 -0.35655]]
"""