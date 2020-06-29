# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 16:12:53 2020

@author: thand
"""

"""
We are now going to train a neural
network model to classify this data.
(a) First we need to choose an architecture for the network. This data is 10D. 
We therefore need 10 input parameters for the model, and one output variable. 
It is a classification problem, so we can choose the final activation function 
to be a sigmoid.

 We know the decision boundary is
non-linear because we made the data – otherwise we may need to visualise some 
of it to figure this
out, and so we need two hidden layers. Let’s use 8 nodes on the 
first hidden layer, 4 nodes on the second hidden layer, and sigmoids
for all activation functions.
"""

LearningRate = 2

def sigmoid(z):
    return 1/(1+math.exp(-z))

#def derSigmoid(z):
 #   return sigmoid(z)*(1-sigmoid(z))

def derSigmoid(z):
    return math.exp(z)/math.pow(math.exp(z) + 1,2)

class Node:
    z = 0 # z is the sum of all the weights*inputs
    a = 0 # Activation value (value of z in activation function)
    level = 0
    layer = 0
    error = 0
    weight = []
    bias = 0
    gradient = 0
    
    def __init__(self,layer,level):
        self.level = level
        self.layer = layer
        
    def setZ(self,value):
        self.z = value
        self.a = sigmoid(self.z)
    
    def setError(self,NextError):
        gPrime = derSigmoid(self.z)
        self.error = 0
        for i in range(len(NextError)):
            self.error += gPrime*self.weight[i]*NextError[i]
        # Compute gradient
        self.gradient += self.a*self.error
        
        # Compute new weights
        for i in range(len(self.weight)):
            self.weight[i] -= LearningRate * self.gradient
        
        
def init_layer(num_neurons, num_next_neurons):
    # if 10 x 5
    weight_matrix = np.random.rand(num_next_neurons*num_neurons).reshape(num_neurons,num_next_neurons)
    
    # then bias vector is a 5-dimensional vector
    bias_vector = np.random.rand(num_next_neurons)
    
    return weight_matrix, bias_vector

def init_random_network(Layer_neurons):
    # () that is a tuple
    output = []
    if(len(Layer_neurons) > 1):
        for i in range(len(Layer_neurons) -1):
            w,b = init_layer( Layer_neurons[i], Layer_neurons[i+1])
            output.append((w,b))
    return output
        
"""
(b) Implement forward propagation for this network. Do this in a vectorised 
way, so we can generalise to different architectures. 
For any input vector, we need a vector of activations at every layer.
"""

# Function returns a 2D array where the first param is the layer and the second
# param is the node in that layer
def makeNodes(params, net):
    out = []
    for i in range(len(params)):
        out.append([])
        for k in range(params[i]):
            out[i].append(Node(i,k))
    
    # Set Weights
    for i in range(len(out)-1):
        for k in range(len(out[i])):
            tempArr = []
            for j in range(len(net[i][0][k])):
                 tempArr.append(net[i][0][k][j])
            out[i][k].weight = tempArr
            
    # Set Bias
    for i in range(len(out)-1):
        for k in range(len(out[i+1])):
            out[i+1][k].bias = net[i][1][k]
            
    return out

# Performs forward propagation on the network from the first layer of inputs

# Recursion starts from the beginning
def calLayer(num,node):
    if(num != 1):
        calLayer(num-1,node)
    levels = len(node[num])
    for i in range(levels):
        value = node[num][i].bias
        for k in range(len(node[num-1])):
            value += node[num-1][k].a * node[num-1][k].weight[i]
        node[num][i].setZ(value)

# Recursion starts from the end 
def calLayerErr(num, target, node):
    if(num != len(node)-1):
        calLayerErr(num+1,target,node)
    else:
        node[num][0].error = node[num][0].a - target
        return
    
    # We need a list of the errors of the next layer
    NextError = []
    for i in range(len(node[num+1])):
        NextError.append(node[num+1][i].error)
        
    # We set the error for each node in this layer
    for i in range(len(node[num])):
        node[num][i].setError(NextError)

def train_network(TData,TAnswers, node):
    for k in range(len(TData)):
        for i in range(len(node[0])):
            node[0][i].a = TData[k][i]
        calLayer(len(node)-1,node)
        calLayerErr(1,TAnswers[k][0],node)
        
        print("Result: ",node[len(node)-1][0].a)
        print("Ans: ",TAnswers[k][0])