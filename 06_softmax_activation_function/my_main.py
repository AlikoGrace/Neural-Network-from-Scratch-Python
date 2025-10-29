from nnfs.datasets import spiral_data
import matplotlib.pyplot as plt
import numpy as np
import nnfs

# Set deterministic seed & float32 defaults (matches the NNFS book)
nnfs.init()


class Layer_dense:
    def __init__(self,n_inputs,n_neurons):
        self.weights=0.01*np.random.randn(n_inputs,n_neurons)
        self.bias=np.zeros((1,n_neurons))
        # just one bracket will give you an error, np.zeros expects a shaped tuple., not bringing it it treats it as two separaate values, thinking n_nerons as a dtype

    def forward(self,inputs):
        self.output=np.dot(inputs,self.weights)+self.bias


class ReLU_Activation:
 
    def forward(self,inputs):
                # 0 for negatives, pass positives through
        self.output=np.maximum(0,inputs)    


class SoftMax_Activation:
        
    def forward(self,inputs):
             # Stabilize: subtract row-wise max so exp() won't overflow
     exp_values=np.exp(inputs-np.max(inputs, axis=1, keepdims=True))
        # Normalize row-by-row to get probabilities that sum to 1

     probabilities=(exp_values/np.sum(exp_values,axis=1,keepdims=True))
     self.output=probabilities
   

# ---- Create dataset: 300 samples (100 per class), 2D inputs ----
X,y=spiral_data(samples=100,classes=3)

# ---- Build the network: 2->3 -> ReLU -> 3->3 -> Softmax ----
dense1=Layer_dense(2,3)
dense2=Layer_dense(3,3)

activation1=ReLU_Activation()

activation2= SoftMax_Activation()

dense1.forward(X)

activation1.forward(dense1.output)
dense2.forward(activation1.output)
activation2.forward(activation1.output)

print(activation2.output[:5])