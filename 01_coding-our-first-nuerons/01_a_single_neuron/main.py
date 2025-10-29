input=[ 1.0 , 2.0 , 3.0 , 2.5 ]
weight=[ 0.2 , 0.8 ,- 0.5 , 1.0 ]
bias=2.0

output = 0

for i in range(len(input)):
    output+=input[i]*weight[i]
output +=bias

print(output)



