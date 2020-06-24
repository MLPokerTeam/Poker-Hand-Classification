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

0: Nothing in hand; not a recognized poker hand
1: One pair; one pair of equal ranks within five cards
2: Two pairs; two pairs of equal ranks within five cards
3: Three of a kind; three equal ranks within five cards
4: Straight; five cards, sequentially ranked with no gaps
5: Flush; five cards with the same suit
6: Full house; pair + different rank three of a kind
7: Four of a kind; four equal ranks within five cards
8: Straight flush; straight + flush
9: Royal flush; {Ace, King, Queen, Jack, Ten} + flush

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
import math
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

#TODO: make a decision tree
#format data entry to needed attributes
def areConsec(arrRank):
    arrRank = np.sort(arrRank)
    isConsec = True
    print("they're consecutive")
    for i in range (len(arrRank)-1): #if the difference is greater than 1 then they are not consecutive
        a = arrRank[i]
        b = arrRank[i+1]
        if b - a != 1 and b-a < 9:
            isConsec = False
            print("turns out they are not")
            break
    return isConsec


def areRoyal(arrRank):
    isRoyal = True
    print("they're all royal")
    for i in arrRank:
        if i == 1: #the ace can be both royal and non-royal, so its existence should not decide if set is royal
            continue
        if i < 10: #anything below 10 is non-royal making the set as a whole non-royal
            isRoyal = False
            print("oh wait, they're not")
            break
    return isRoyal


def areDuplicate(arrRank):
    duos = 0
    fTrio = False
    fDuo = False
    fDuoDuo = False
    for i in arrRank:
        if np.sum(arrRank == i) == 3:
            fTrio = True
        elif (np.sum(arrRank == i) == 2):
            duos -=- 1
    if duos == 2: #if there is more than one pair then the dual duo is prioritised
        fDuoDuo = True
        fDuo = True
    elif duos == 1:
        fDuo = True
    print("trio: ", fTrio)
    print("single pair:", fDuo)
    print("two pairs: ", fDuoDuo)
    return (fTrio, fDuo, fDuoDuo)


def areSuited(arrSuit):
    print("same suit:",len(np.unique(arrSuit)) == 1)
    return len(np.unique(arrSuit)) == 1


def attributeFix(entry):
    arrAtt = np.empty([0,0])
    arrRank = entry[1::2]
    arrSuit = entry[::2]
    #if cards are consecutive
    arrAtt = np.append(arrAtt, areConsec(arrRank))
    #if cards royal
    arrAtt = np.append(arrAtt, areRoyal(arrRank))
    #if cards are trio, double duo, or single duo
    arrAtt = np.append(arrAtt, areDuplicate(arrRank))
    #if cards are same suit
    arrAtt = np.append(arrAtt, areSuited(arrSuit))
    return arrAtt


arrPractice = np.array([2,13,2,1,4,4,1,5,2,11])
print(attributeFix(arrPractice))
#TODO: make a confusion matrix
