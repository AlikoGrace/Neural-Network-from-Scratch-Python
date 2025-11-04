
# Forward Pass
x = [1.0, -2.0, 3.0] # input values
w = [-3.0, -1.0, 2.0] # weights
b = 1.0 # bias

# Multiplying inputs by weights
xw0 = x[0] * w[0] # 1.0 * -3.0
xw1 = x[1] * w[1] # -2.0 * -1.0
xw2 = x[2] * w[2] # 3.0 * 2.0

print("Inputs * weights: ", xw0, xw1, xw2) # -3.0, 2.0, 6.0

# Adding weighted inputs and a bias
z = xw0 + xw1 + xw2 + b
print("sum(Inputs * weights + bias): ", z) # 6.0

# ReLU activation function
y = max(z, 0)
print("ReLU output: ", y) # 6.0

# Backward Pass

# The derivative from the next layer
dvalue = 1.0

# explanation in the previous layer
drelu_dz = dvalue * (1. if z > 0 else 0.)

print("Gradient of ReLU: ", drelu_dz)

# Applying the chain rule to compute gradients of ReLU output w.r.t each weighted input and bias for backpropagation
dsum_dxw0 = 1
dsum_dxw1 = 1
dsum_dxw2 = 1
dsum_db = 1
drelu_dxw0 = drelu_dz * dsum_dxw0  
drelu_dxw1 = drelu_dz * dsum_dxw1
drelu_dxw2 = drelu_dz * dsum_dxw2
drelu_db = drelu_dz * dsum_db
# 1.0*1.0 for all
print("Partial derivatives (gradients) of ReLU output w.r.t xw0, xw1, xw2, and b:", drelu_dxw0, drelu_dxw1, drelu_dxw2, drelu_db)\


# Calculating partial derivatives of ReLU output w.r.t each original input x[i] and weight w[i]
# Here, we apply the chain rule again to backpropagate gradients through the multiplication operations in the forward pass.

# The partial derivative of f with respect to x equals y . The partial derivative of f with respect to y
# equals x . Following this rule, the partial derivative of the first weighted input with respect to the
# input equals the weight (the other input of this function). understand this before we go to the next step to complete backpropagaition

dmul_dx0 = w[0]
dmul_dx1 = w[1]
dmul_dx2 = w[2]
dmul_dw0 = x[0]
dmul_dw1 = x[1]
dmul_dw2 = x[2]
# # dmul_dx0: derivative of (x0 * w0) with respect to x0 (local gradient = w0)
# dmul_dw0: derivative of (x0 * w0) with respect to w0 (local gradient = x0)

drelu_dx0 = drelu_dxw0 * dmul_dx0 
# Chain rule: final gradient wrt x0 = gradient wrt xw0 * local gradient wrt x0 (w0)
# Multiply incoming gradient by each multiplication’s local derivative (chain rule) to get gradients for inputs and weights.

drelu_dw0 = drelu_dxw0 * dmul_dw0

drelu_dx1 = drelu_dxw1 * dmul_dx1
drelu_dw1 = drelu_dxw1 * dmul_dw1

drelu_dx2 = drelu_dxw2 * dmul_dx2
drelu_dw2 = drelu_dxw2 * dmul_dw2


print("Final gradients w.r.t original inputs and weights:", drelu_dx0, drelu_dw0, drelu_dx1, drelu_dw1, drelu_dx2, drelu_dw2)

# Partial derivatives of the multiplication, the chain rule
