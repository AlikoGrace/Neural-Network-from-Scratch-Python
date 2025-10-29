
inputs = [ 0 , 2 ,- 1 , 3.3 ,- 2.7 , 1.1 , 2.2 , - 100 ]
output = []
for i in inputs:
 if i > 0 :
  output.append(i)
 else :
  output.append( 0 )
print (output)


#---------More efficient way------------/

 
output.append(max(0,i))
print(output)
# since a relu function checks if i is greater than. zero it appends it to i and if smaller it makes it zero.

# using numpy

import numpy as np
output=np.maximum(0,inputs)
print(output)
