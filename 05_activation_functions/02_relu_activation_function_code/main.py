import numpy as np
from nnfs.datasets import spiral_data
import nnfs

nnfs.init()

# Dense layer
class Layer_Dense:
    # Layer Initialization
    def __init__(self, n_inputs, n_neurons):
        # Initialize weights and biases
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

  # Forward Pass
    def forward(self, inputs):
        # Calculate output values from inputs, weights and biases
        self.output = np.dot(inputs, self.weights) + self.biases

        print(self.output[:5])
        # comparing with the last printout, you would see all the negative values have been clipped out or modified to be zero. that's why we used ReLU for hidden layers
       

class Activation_ReLU:
    # forward pass
    def forward(self,inputs):
        # calculate the values from the input
        self.output=np.maximum(0,inputs)


#applying it to the dense layer's ouputs in our code
# 
# create datasset
# 
X,y=spiral_data(samples=100,classes=3) 

# create a dense layer with two input features and three output values

dense1=Layer_Dense(2,3)
# create a ReLu function to be used with the dense layer
activation1=Activation_ReLU()

dense1.forward(X)

# Forward pass through activation func.
# Takes in output from previous layer
activation1.forward(dense1.output)

# Let's see output of the first few samples:
print(activation1.output[:5])
