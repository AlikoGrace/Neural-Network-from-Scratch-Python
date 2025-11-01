import math

softmax_output = [ 0.7 , 0.1 , 0.2 ]

target_output = [ 1 , 0 , 0 ]


# the formula simplified, categorical cross entropy forula
loss = - (math.log(softmax_output[ 0 ]) * target_output[0] +math.log(softmax_output[ 1 ])*target_output[1 ]+math.log(softmax_output[ 2 ]) * target_output[ 2 ] )


# you can also use loss=-math.log(softmax_output[0]) , you know shy right ?

print(loss)