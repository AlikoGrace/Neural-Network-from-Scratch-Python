import numpy as np
import nnfs    
from nnfs.datasets import spiral_data
nnfs.init()


class Layer_Dense:

    def __init__(self,n_imputs,n_neurons):
        # initializing weights and bias
        self.weights= 0.01* np.random.randn(n_imputs, n_neurons) 
        # np.random.randn produces a Gaussian distribution with amean of 0 and a variance of 1, multiply by 0.01 to make the numbers a magnitude smaller to reduce time taken by the model to fit.
        self.bias=np.zeros((1,n_neurons))
# since we are setting weights to inputs, neurons and not neurons input like the convention, there is no need to transpose weights.


    # forward pass When we pass data through a model from beginning to end
    def forward(self,inputs):
        # Calculate output values from inputs, weights and biases
        pass # using pass statement as placeholder
        self.output=np.dot(inputs, self.weights)+self.bias

    # Weights are initailly set randamly for a model and tweaked till they genralized but if you want to pre load a tria model you will initialize the parameters to whatever that pretrained model finished with


    # ----------------------------------------------/ let's try it out 

    import nnfs

    nnfs.init()
    n_inputs=2
    n_neurons=4

    weights=0.01 * np.random.randn(n_inputs,n_neurons)
    bias=np.zeros((1,n_neurons))

    print(weights)
    print(bias)


    #------------------------------------------------/using it on our generated data


    # create dataset
X,y=spiral_data(samples=100,classes=3)
# Create Dense layer with 2 input features and 3 output values/neurons, nummber of neurons can be anything you want.
dense1 = Layer_Dense( 2 , 3 )

# Perform a forward pass of our training data through this layer
dense1.forward(X)

print (dense1.output[: 5 ])