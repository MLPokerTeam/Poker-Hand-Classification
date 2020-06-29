# -*- coding: utf-8 -*-
"""
@author: Thando Peter 1908664@students.wits.ac.za
@author: Tieho Ramphore 1908649@students.wits.ac.za
@author: 
"""

"""
Information of the structure of the data (All this information can be found in the poker-hand.name file):
    
Attribute Information:

1) S1 "Suit of card #1"
Ordinal (1-4) representing {Hearts, Spades, Diamonds, Clubs}

2) C1 "Rank of card #1"
Numerical (1-13) representing (Ace, 2, 3, ... , Queen, King)

3) S2 "Suit of card #2"
Ordinal (1-4) representing {Hearts, Spades, Diamonds, Clubs}

4) C2 "Rank of card #2"
Numerical (1-13) representing (Ace, 2, 3, ... , Queen, King)

5) S3 "Suit of card #3"
Ordinal (1-4) representing {Hearts, Spades, Diamonds, Clubs}

6) C3 "Rank of card #3"
Numerical (1-13) representing (Ace, 2, 3, ... , Queen, King)

7) S4 "Suit of card #4"
Ordinal (1-4) representing {Hearts, Spades, Diamonds, Clubs}

8) C4 "Rank of card #4"
Numerical (1-13) representing (Ace, 2, 3, ... , Queen, King)

9) S5 "Suit of card #5"
Ordinal (1-4) representing {Hearts, Spades, Diamonds, Clubs}

10) C5 "Rank of card 5"
Numerical (1-13) representing (Ace, 2, 3, ... , Queen, King)

11) CLASS "Poker Hand"
Ordinal (0-9)

0: Nothing in hand; not a recognized poker hand                      Rank
1: One pair; one pair of equal ranks within five cards               Rank
2: Two pairs; two pairs of equal ranks within five cards             Rank
3: Three of a kind; three equal ranks within five cards              Rank
4: Straight; five cards, sequentially ranked with no gaps            Rank
5: Flush; five cards with the same suit                              Suit
6: Full house; pair + different rank three of a kind                 Rank
7: Four of a kind; four equal ranks within five cards                Rank
8: Straight flush; straight + flush                                  Rank + Suit
9: Royal flush; {Ace, King, Queen, Jack, Ten} + flush                Specific + Suit



The order of cards is important, which is why there
     are 480 possible Royal Flush hands as compared to 4 (one for each
     suit � explained in more detail below).
     
Number of Instances: 25010 training, 1,000,000 testing

Number of Attributes: 10 predictive attributes, 1 goal attribute

9. Class Distribution:

      The first percentage in parenthesis is the representation
      within the training set. The second is the probability in the full domain.

      Training set:

      0: Nothing in hand, 12493 instances (49.95202% / 50.117739%)
      1: One pair, 10599 instances, (42.37905% / 42.256903%)
      2: Two pairs, 1206 instances, (4.82207% / 4.753902%)
      3: Three of a kind, 513 instances, (2.05118% / 2.112845%)
      4: Straight, 93 instances, (0.37185% / 0.392465%)
      5: Flush, 54 instances, (0.21591% / 0.19654%)
      6: Full house, 36 instances, (0.14394% / 0.144058%)
      7: Four of a kind, 6 instances, (0.02399% / 0.02401%)
      8: Straight flush, 5 instances, (0.01999% / 0.001385%)
      9: Royal flush, 5 instances, (0.01999% / 0.000154%)

      The Straight flush and Royal flush hands are not as representative of
      the true domain because they have been over-sampled. The Straight flush
      is 14.43 times more likely to occur in the training set, while the
      Royal flush is 129.82 times more likely.

      Total of 25010 instances in a domain of 311,875,200.

      Testing set:

	The value inside parenthesis indicates the representation within the test
      set as compared to the entire domain. 1.0 would be perfect representation,
      while <1.0 are under-represented and >1.0 are over-represented.

      0: Nothing in hand, 501209 instances,(1.000063)
      1: One pair, 422498 instances,(0.999832)
      2: Two pairs, 47622 instances, (1.001746)
      3: Three of a kind, 21121 instances, (0.999647)
      4: Straight, 3885 instances, (0.989897)
      5: Flush, 1996 instances, (1.015569)
      6: Full house, 1424 instances, (0.988491)
      7: Four of a kind, 230 instances, (0.957934)
      8: Straight flush, 12 instances, (0.866426)
      9: Royal flush, 3 instances, (1.948052)

      Total of one million instances in a domain of 311,875,200.


10. Statistics

      Poker Hand       # of hands	Probability	    # of combinations
      Royal Flush      4		    0.00000154	    480
      Straight Flush   36		    0.00001385	    4320
      Four of a kind   624		    0.0002401	    74880
      Full house       3744		    0.00144058	    449280
      Flush            5108		    0.0019654	    612960
      Straight         10200		0.00392464	    1224000
      Three of a kind  54912		0.02112845	    6589440
      Two pairs        123552		0.04753902	    14826240
      One pair         1098240	    0.42256903	    131788800
      Nothing          1302540	    0.50117739	    156304800

      Total            2598960	    1.0             311875200

      The number of combinations represents the number of instances in the entire domain.
"""

import numpy as np
import re #regex import that allows string splitting via multiple delimeters to get rid of \n
"""
We need to load in the data from the .data files.
We have two files:

1. poker-hand-testing.data
2. poker-hand-training-true.data
"""
#reading in training data
trainingData = open('poker-hand-training-true.data', 'r')
trainingString = trainingData.read()
fullTrainingData = np.array(re.split(',|\n', trainingString))[:-1].astype(np.int64).reshape(25010, 11)
print("The training set has ", fullTrainingData.shape[0], "data points with ", fullTrainingData.shape[1], "attributes")


"""
Split the training data into validation and blind testing (500 000 each)

"""
#reading in testing data
testingData = open('poker-hand-testing.data', 'r')
testingString = testingData.read()
fullTestingData = np.array(re.split(',|\n', testingString))[:-1].astype(np.int64).reshape(1000000, 11)
print("The testing set has ", fullTestingData.shape[0], "data points with ", fullTestingData.shape[1], "attributes")

#Splitting up into validation and blind testing data
splitValData = fullTestingData[:500000]
splitTestData = fullTestingData[500000:]
print("The validation set has ", splitValData.shape[0], "data points with ", splitValData.shape[1], "attributes")
print("The blind testing set has ", splitTestData.shape[0], "data points with ", splitTestData.shape[1], "attributes")
"""
Cut the split validation and blind data into a workable chunk

Take 10% of each array and use it to code. This will make sure that testing
 code is fast and ease the coding process. The full data-set will be used in the final training.
 
 It would be best if you just deleted data from the above arrays, so that later we can just 
 comment out this section of the code to have the full dataset.

"""
#Getting workable data sizes (10%)
splitValData = splitValData[:50000]
splitTestData = splitTestData[:50000]

#Splitting attributes from results
trainAtt = fullTrainingData[:, :10]
trainRes = fullTrainingData[:, 10:]
valAtt = splitValData[:, :10]
valRes = splitValData[:, 10:]
testAtt = splitTestData[:, :10]
testRes = splitTestData[:, 10:]
print("The training attribute set has ", trainAtt.shape[0], "data points with ", trainAtt.shape[1], "attributes")
print("The training result set has ", trainRes.shape[0], "data points with ", trainRes.shape[1], "attribute")

# Seperates the the suit input and the rank input
# Since the suit only influences 3 out of the 10 possible hands and the occurances
# it influences dim in comparison to the occurances only influenced by the rank
# the occurances influenced by the suit become outliers.
def getRankAndSuit(arr):
    arr2 = arr.T
    Rank = []
    Suit = []
    for i in range(len(arr2)):
        if(i%2==0):
            Suit.append(arr2[i])
        else:
            Rank.append(arr2[i])
    return np.array(Rank).T,np.array(Suit).T

# Returns input and result arrays with evenly distributed variations.
# The number of entries of each variation is dependent on the variation with
# the smallest occurances.
def Level_data(a,b):
    limit = 5
    limitArr = [0,0,0,0,0,0,0,0,0,0]
    newA = []
    newB = []
    temp = -1
    for i in range(len(b)):
        temp = -1
        for k in range(10):
            if(b[i] == [k] and limitArr[k] <limit):
                temp=k
        if(temp!=-1):
            limitArr[temp]+=1
            newA.append(a[i])
            newB.append(b[i])
    return np.array(newA), np.array(newB)


class NeuralNetwork(object):
    def __init__(self, x, y, shape):
        
        self.shape = shape
        
        # Turns all the values into a percentage based on the max number in 
        # each list
        self.Xm = np.amax(x,axis=0)
        self.Ym = np.amax(y,axis=0)
        self.X = x/self.Xm
        self.Y = y/self.Ym
        
        # The number of hidden layers in the neural network
        self.HiddenLayers = len(shape) - 1 
        
        # Weights
        self.Weights = []
        for i in range(self.HiddenLayers):
            self.Weights.append(np.random.randn(shape[i], shape[i+1]))
        
    def feedForward(self):
        # Moves the initial values through the nueral network and sets the values
        # of the final layer
        self.z = []
        for i in range(self.HiddenLayers):
            if(i==0):
                # Calculates the activation function of the first hidden layer
                # from the input
                self.z.append(self.sigmoid(np.dot(self.X, self.Weights[i])))
            else:
                # Calculates the activation function of next hidden layer
                # from the previous layer's activation function
                self.z.append(self.sigmoid(np.dot(self.z[i-1], self.Weights[i])))
        return self.z[len(self.z)-1]
        
    def sigmoid(self, s, deriv = False):
        if(deriv == True):
            return s*(1-s) # Returns the derivative of the sigmoid function
        return 1/(1+np.exp(-s))
    
    def backward(self,LearningRate):
        self.error = []
        self.delta = []
        for i in range(self.HiddenLayers):
            if(i==0):
                # Subtracts the calculated final result from the intended result
                # to get the error
                self.error.append(self.Y - self.z[self.HiddenLayers-i-1])
            else:
                # Calculates how much the layer weights of our hidden layers 
                # contribute to the output error
                self.error.append( self.delta[i-1].dot(self.Weights[self.HiddenLayers-i].T))
                
            # Applying the derivative of the sigmoid function to the error
            self.delta.append(self.error[i] * self.sigmoid(self.z[self.HiddenLayers-i-1], deriv = True))
        
        # After calculating the errors for all the weights in all the layers we
        # adjust the weights to reduce the error
        for i in range(self.HiddenLayers):
            if(i==0):
                # Input layer to the hidden layer
                self.Weights[i] += LearningRate * self.X.T.dot(self.delta[self.HiddenLayers-i-1])
            else:
                # Hidden layer to next layer (Hidden/Output layer)
                self.Weights[i] += LearningRate * self.z[i-1].T.dot(self.delta[self.HiddenLayers-i-1])
        
    def train(self,loops, learningRate):
        p = 0
        for i in range(loops):
            if(round(i*100/loops)!=p):
                p = round(i*100/loops)
                print(p,"%")
            out = self.feedForward()
            self.backward(learningRate)
    
    # Returns Input Data Results
    def getYArray(self):
        return self.Y*self.Ym
    
    # Returns Input Data Input
    def getXArray(self):
        return self.X*self.Xm
    
    # Returns the result after the input data is processed through the network
    def getNetworkYArray(self):
        return self.feedForward()*self.Ym
    
    def getWeights(self):
        return self.Weights
    
    def loadWeights(self,val):
        self.Weights = val
        
    def loadInput(self, xInput, yInput):
        self.Xm = np.amax(xInput,axis=0)
        self.Ym = np.amax(yInput,axis=0)
        self.X = xInput/self.Xm
        self.Y = yInput/self.Ym
        
    def printOutput(self):
        arrY = self.getYArray()
        arrYAns = self.getNetworkYArray()
        arrDiff = arrY - arrYAns
        for i in range(len(arrY)):
            print(str(arrY[i])," - ",str(np.round(arrYAns[i]))," = ", str(np.round(arrDiff[i])))
            
    def printError(self):
       arrDiff = self.getYArray() - self.getNetworkYArray()
       print("Loss: ",np.mean(np.square(arrDiff)),'\n')
       
    def printConfusionMatrix(self):
        confusion = []
        YArray = self.getYArray()
        for i in range(int(np.max(YArray))+1):
            confusion.append(np.zeros(int(np.max(YArray))+1))
        
        YNetwork = self.getNetworkYArray()
        Correct = 0
        Wrong = 0
        for i in range(len(YArray)):
            confusion[int(round(YArray[i][0]))][int(round(YNetwork[i][0]))] += 1
            if(round(YArray[i][0]) == round(YNetwork[i][0])):
                Correct += 1
            else:
                Wrong += 1
                
        print(confusion)
        print("Correct Percentage: ",Correct*100/(Wrong+Correct))
        print("Wrong Percentage: ",Wrong*100/(Wrong+Correct))
            

# Combines the output of two neural Networks so that they can serve as input 
# Into a third
def CombineOutput(a,b):
    EndInput = []
    EndInput.append((np.array(a).T)[0])
    EndInput.append((np.array(b).T)[0])
    return np.array(EndInput).T
    
# Function takes in the rank network, the suit network and the end network
# and propagates the input data through the entire network.
# It then returns the loss of the network
def PropThroughRankAndSuit(inputA,inputB,RankNetwork,SuitNetwork,EndNetwork):
    ARank, ASuit = getRankAndSuit(inputA)
    
    RankNetwork.loadInput(ARank,inputB)
    RankOut = RankNetwork.feedForward()
    
    SuitNetwork.loadInput(ASuit,inputB)
    SuitOut = SuitNetwork.feedForward()
    
    EndInput = CombineOutput(RankOut,SuitOut)
    
    EndNetwork.loadInput(EndInput,inputB)
    EndNetwork.feedForward()
    #EndNetwork.printOutput()
    EndNetwork.printError()
    EndNetwork.printConfusionMatrix()
    
# For the ones with suit it is either the same or nothing. There are 3 of these types.
# Theres the flush, straight flush and the royal flush

# It would be best to seperate these two

# There's enough layers to figure out the complexity of poker hands but not too
    # much that it will overfit to the training data



# Gets equal amounts of each unique output as data
levelTrainAtt, levelTrainRes = Level_data(trainAtt,trainRes)
np.append(levelTrainAtt,trainAtt[50:2000])
np.append(levelTrainRes,trainRes[50:2000])
"""
# Extracts the rank and the suit
trainRank, trainSuit = getRankAndSuit(levelTrainAtt)

# Loss does not get better after 10 000 iterations
TrainingIterations = 20000
RankShape = [5,15,10,15,8,1]
NNRank = NeuralNetwork(trainSuit,levelTrainRes,RankShape)
NNRank.train(TrainingIterations, 1.2)
NNRank.printError()
NNRank.printConfusionMatrix()

TrainingIterations = 10000
SuitShape = [5,8,6,4,1]
NNSuit = NeuralNetwork(trainRank,levelTrainRes,SuitShape)
NNSuit.train(TrainingIterations, 1)
NNSuit.printError()
NNSuit.printConfusionMatrix()

TrainingIterations = 50000
EndShape = [2,16,8,4,1]
NNEnd = NeuralNetwork(CombineOutput(NNRank.feedForward(),NNSuit.feedForward()),levelTrainRes,EndShape)
NNEnd.train(TrainingIterations, 1)

PropThroughRankAndSuit(testAtt,testRes,NNRank,NNSuit,NNEnd)


"""
TrainingIterations = 1000
NetworkShape = [10, 8, 6, 4, 1]
NN = NeuralNetwork(trainAtt,trainRes,NetworkShape)
NN.train(TrainingIterations, 1)
NN.printError()
NN.loadInput(testAtt,testRes)
NN.printError()
NN.printConfusionMatrix()
