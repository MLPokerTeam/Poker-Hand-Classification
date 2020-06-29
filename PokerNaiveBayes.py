import numpy as np

# -*- coding: utf-8 -*-
"""
@author: Thando Peter 1908664@students.wits.ac.za
@author: Tieho Ramphore 1908649@students.wits.ac.za
@author: Olebogeng Maleho 1862666@students.wits.ac.za
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

      Poker Hand       # of hands	Probability	# of combinations
      Royal Flush      4		0.00000154	480
      Straight Flush   36		0.00001385	4320
      Four of a kind   624		0.0002401	74880
      Full house       3744		0.00144058	449280
      Flush            5108		0.0019654	612960
      Straight         10200		0.00392464	1224000
      Three of a kind  54912		0.02112845	6589440
      Two pairs        123552		0.04753902	14826240
      One pair         1098240	0.42256903	131788800
      Nothing          1302540	0.50117739	156304800

      Total            2598960	1.0		311875200

      The number of combinations represents the number of instances in the entire domain.
"""

"""
We need to load in the data from the .data files.
We have three files:

1. poker-hand-testing.data
2. poker-hand-training-true.data
"""

"""
Split the training data into validation and blind testing (500 000 each)
"""

"""
Cut the split validation and blind data into a workable chunk

Take 5% of each array and use it to code. This will make sure that testing
 code is fast and ease the coding process. The full data-set will be used in the final training.
 
 It would be best if you just deleted data from the above arrays, so that later we can just 
 comment out this section of the code to have the full dataset.

"""


def read_file(data,size):
    training_data1 = np.array([line.rstrip('\n') for line in open(data)])

    training_data2 = np.zeros((size, 11))

    for i in range(size):

        for j in range(11):
            str1 = training_data1[i].split(",")
            training_data2[i][j] = str1[j]

    return training_data2


training_data = np.array(read_file("poker-hand-training-true.data",25010))

# prior probabilities

priors = np.array(
    [12493 / 25010, 10599 / 25010, 1206 / 25010, 513 / 25010, 93 / 25010, 54 / 25010, 36 / 25010, 6 / 25010, 5 / 25010,
     5 / 25010])


suit = {1.0: "Hearts", 2.0: "Spades", 3.0: "Diamonds", 4.0: "Clubs"}

rank = {1.0: "Ace of ", 2.0: "2 of ", 3.0: "3 of ", 4.0: "4 of ", 5.0: "5 of ", 6.0: "6 of ", 7.0: "7 of ", 8.0: "8 of ", 9.0: "9 of ", 10.0: "10 of ", 11.0: "Jack of ", 12.0: "Queen of ", 13.0: "King of "}

outcomes = {0.0: " NIH", 1.0: " 1 Pair", 2.0: " 2 Pairs", 3.0: " Three of Kind", 4.0: " Straight" , 5.0: " flush", 6.0: " full house", 7.0: " 4 of Kind" , 8.0: " Straight Flush" , 9.0: " Royal flush"}

no_of_outcomes = {" NIH": 12493, " 1 Pair": 10599, " 2 Pairs": 1206, " Three of Kind": 513, " Straight": 93, " flush": 54, " full house": 36, " 4 of Kind": 6, " Straight Flush": 5, " Royal flush": 5}

li = [["0" for col in range(5)] for row in range(25010)]

for p in range(25010):

    for k in range(5):

        i = 2*k
        j = 2*k+1
        Str = rank[training_data[p][j]]+suit[training_data[p][i]]
        li[p][k] = Str

matrix = np.array(li)
new_matrix = matrix.reshape(125050,1)
unique1, counts1 = np.unique(new_matrix,return_counts=True)

hands = [["0" for col in range(5)] for row in range(25010)]

for p in range(25010):

    for k in range(5):

        i = 2*k
        j = 2*k+1
        Str = rank[training_data[p][j]]+suit[training_data[p][i]]

        if training_data[p][10] == 0.0:
            Str = Str + "," + outcomes[training_data[p][10]]
            hands[p][k] = Str

        if  training_data[p][10] == 1.0:
            Str = Str + "," + outcomes[training_data[p][10]]
            hands[p][k] = Str

        if  training_data[p][10] == 2.0:
            Str = Str + "," + outcomes[training_data[p][10]]
            hands[p][k] = Str

        if  training_data[p][10] == 3.0:
            Str = Str + "," + outcomes[training_data[p][10]]
            hands[p][k] = Str

        if  training_data[p][10] == 4.0:
            Str = Str + "," + outcomes[training_data[p][10]]
            hands[p][k] = Str

        if  training_data[p][10] == 5.0:
            Str = Str + "," + outcomes[training_data[p][10]]
            hands[p][k] = Str

        if  training_data[p][10] == 6.0:
            Str = Str + "," + outcomes[training_data[p][10]]
            hands[p][k] = Str

        if  training_data[p][10] == 7.0:
            Str = Str + "," + outcomes[training_data[p][10]]
            hands[p][k] = Str

        if  training_data[p][10] == 8.0:
            Str = Str + "," + outcomes[training_data[p][10]]
            hands[p][k] = Str

        if  training_data[p][10] == 9.0:
            Str = Str + "," + outcomes[training_data[p][10]]
            hands[p][k] = Str

hands_and_outcomes = np.array(hands)
hands_and_outcomes = hands_and_outcomes.reshape(125050,1)
unique2, counts2 = np.unique(hands_and_outcomes,return_counts=True)


def cond_prob(key,value):

    deck = [0]*52
    cond_probabilities = [0]*52
    denominator = value
    length = unique2.size

    for i in range(52):

        for j in range(length):
            str1 = unique2[j].split(",")
            if unique1[i] == str1[0] and key == str1[1]:
               deck[i] = counts2[j]

    for k in range(52):
        cond_probabilities[k] = (deck[k] + 1) / (denominator + 10)

    cond_probabilities = np.array(cond_probabilities)

    return cond_probabilities


def encode_data_point(training_data,i):

    data_point = training_data[i]
    temp_data_point = ["0"]*5
    final_data_point = [0]*52

    for eta in range(5):
        phi = 2*eta
        psi = 2*eta + 1
        Str = rank[data_point[psi]]+suit[data_point[phi]]
        temp_data_point[eta] = Str

    for alpha in range(52):

        for beta in range(5):

            if unique1[alpha] == temp_data_point[beta]:
               final_data_point[alpha] = 1

    return final_data_point


def naive_bayes(priors,training_data,i):

    prod = 1
    posterior_prob = np.zeros(10)
    anterior_prob = np.zeros(10)
    data_point = encode_data_point(training_data,i)

    for x in range(10):

        for y in range(52):

           if data_point[y] == 0:
              prod = prod * (1 - cond_prob(outcomes[x],no_of_outcomes[outcomes[x]])[y])

           else:
                prod = prod * (cond_prob(outcomes[x],no_of_outcomes[outcomes[x]])[y])

        posterior_prob[x] = prod
        prod = 1

    for z in range(10):
        anterior_prob[z] = posterior_prob[z] * priors[z] / ( posterior_prob[0]*priors[0] + posterior_prob[1]*priors[1] + posterior_prob[2]*priors[2] + posterior_prob[3]*priors[3] + posterior_prob[4]*priors[4] + posterior_prob[5]*priors[5] + posterior_prob[6]*priors[6] + posterior_prob[7]*priors[7] + posterior_prob[8]*priors[8] + posterior_prob[9]*priors[9] )

    max_prob = np.amax(anterior_prob)
    result = ""

    for i in range(10):

        if anterior_prob[i] == max_prob:
           result = outcomes[i]

    return result

#Testing the algorithm


test_n_valid = np.array(read_file("poker-hand-testing.data",1000000))

testing_data = np.array(test_n_valid[:500000])

validation_data = np.array(test_n_valid[50000:])

hands = [["0" for col in range(5)] for row in range(500000)]

for p in range(500000):

    for k in range(5):

        i = 2*k
        j = 2*k+1
        Str = rank[testing_data[p][j]]+suit[testing_data[p][i]]

        if testing_data[p][10] == 0.0:
            Str = Str + "," + outcomes[testing_data[p][10]]
            hands[p][k] = Str

        if  testing_data[p][10] == 1.0:
            Str = Str + "," + outcomes[testing_data[p][10]]
            hands[p][k] = Str

        if  testing_data[p][10] == 2.0:
            Str = Str + "," + outcomes[testing_data[p][10]]
            hands[p][k] = Str

        if  testing_data[p][10] == 3.0:
            Str = Str + "," + outcomes[testing_data[p][10]]
            hands[p][k] = Str

        if  testing_data[p][10] == 4.0:
            Str = Str + "," + outcomes[testing_data[p][10]]
            hands[p][k] = Str

        if  testing_data[p][10] == 5.0:
            Str = Str + "," + outcomes[testing_data[p][10]]
            hands[p][k] = Str

        if  testing_data[p][10] == 6.0:
            Str = Str + "," + outcomes[testing_data[p][10]]
            hands[p][k] = Str

        if  testing_data[p][10] == 7.0:
            Str = Str + "," + outcomes[testing_data[p][10]]
            hands[p][k] = Str

        if  testing_data[p][10] == 8.0:
            Str = Str + "," + outcomes[testing_data[p][10]]
            hands[p][k] = Str

        if  testing_data[p][10] == 9.0:
            Str = Str + "," + outcomes[testing_data[p][10]]
            hands[p][k] = Str

hands_and_outcomes = np.array(hands)
hands_and_outcomes = hands_and_outcomes.reshape(2500000,1)
unique4, counts4 = np.unique(hands_and_outcomes,return_counts=True)

sum = 0.0
for g in range(500000):

    res = naive_bayes(priors,testing_data,g)
    temp =  hands_and_outcomes[g][0].split(",")
    if res == temp[1]:
       sum = sum + 1.0

accuracy = (sum / 500000)*100
print("The reported accuracy is {x}%".format(x = accuracy))