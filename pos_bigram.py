#from __future__ import division
import nltk
import pickle
# nltk.download('averaged_perceptron_tagger')
from nltk import pos_tag
from numpy.random import choice
import numpy as np

bigram_p = {}

START_SYM = "<s>"
PERCENTAGE = 0.10
DTYPE_ERROR = "Dytpe does not exist."

# TODO: add a right skewed prob dist
def createDist(possible, dtype="uniform"):
    if(dtype=="uniform"):
        dist = []
        for i in range(len(possible)):
            dist.append(1.0/len(possible))

        return dist

    if(dtype=="right_skewed"):
        total = 0
        for it in possible:
            total += it[1]
        dist = []
        for i in range(len(possible)):
            dist.append(possible[i][1]/total)

        return dist

    else:
        return DTYPE_ERROR

def bigramSort(listOfBigrams):
    return sorted(listOfBigrams, key=lambda x: x[1], reverse=True)

def createListOfBigrams():   
    f = open("./data/annotated.txt", "r")
    corpus = f.readlines()
    for sentence in corpus:
        tokens = sentence.split()
        tokens = [START_SYM] + tokens 
        tokens_tag = pos_tag(tokens)
        bigrams = (tuple(nltk.bigrams(tokens_tag)))
        
        for bigram in bigrams:
            if(bigram[0][0]=="<s>" or bigram[1][0]=="<s>"):
                continue
            if(bigram[0][0]=="(pause)" or bigram[1][0]=="(pause)" or \
                bigram[0][0]=="(uh)" or bigram[1][0]=="(uh)" or \
                bigram[0][0]=="(um)" or bigram[1][0]=="(um)"):
                if bigram not in bigram_p:
                    # print(bigram)
                    bigram_p[bigram] = 1
                else:
                    bigram_p[bigram] += 1

    listOfBigrams = [(k, v) for k, v in bigram_p.items()]
    return bigramSort(listOfBigrams)
    

def searchDraw(word, draw):
    for it in draw:
        if( (it[0][0][0] == word) or (it[0][1][0] == word) ):
            return 1 
    return 0

def returnDraw(word, draw):
    for it in draw:
        if( (it[0][0][0] == word) or (it[0][1][0] == word) ):
            return it[0]

def cleanInput(sent):
    sent = sent.lower()
    return sent.replace(".", "") \
                .replace(",", "") \
                .replace("\"", "")

def getPOS(sentence, listOfBigrams):
    sentence = sentence.lower()
    tokens = sentence.split()
    tokens = pos_tag(tokens)
    possibleBigrams = []
    for token in tokens:
        for j in range(len(listOfBigrams)):
            if( (token == listOfBigrams[j][0][0]) or (token == listOfBigrams[j][0][1]) ):
                possibleBigrams.append(listOfBigrams[j])
    return bigramSort(possibleBigrams)

    
if __name__ == "__main__":
    inputSentence = cleanInput(input("Input Sentence: "))

    # bigrams = createListOfBigrams()
    # outfile = open('./obj/pos_bigram', 'wb')
    # pickle.dump(bigrams, outfile)
    # outfile.close()

    infile = open('./obj/pos_bigram', 'rb')
    bigrams = pickle.load(infile)
    infile.close()
   
    choices = np.array(getPOS(inputSentence, bigrams))
    print(choices)
    draw = choices[choice(choices.shape[0], int(PERCENTAGE*(inputSentence.count(" ")+1)), p=createDist(choices, dtype="right_skewed"))]
    print(draw)
    

    for word in list(inputSentence.split()):
        if(searchDraw(word, draw)==1):
            tup = returnDraw(word, draw)
            print(tup[0][0]+" ", tup[1][0]+" ", end="")
        else:
            print(word + " ", end="")
    print()