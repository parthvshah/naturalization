#from __future__ import division
import nltk
import pickle
from numpy.random import choice
import numpy as np
from numpy import array
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, load_model
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense
from keras.layers import LSTM, GRU
from keras.layers import Embedding
from random import sample
 

bigram_p = {}

START_SYM = "<s>"
PERCENTAGE = 0.095
DTYPE_ERROR = "Dytpe does not exist."

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
    f = open("./data/reduced.txt", "r")
    corpus = f.readlines()

    for sentence in corpus:
        tokens = sentence.split()
        tokens = [START_SYM] + tokens 
        bigrams = (tuple(nltk.bigrams(tokens)))
        for bigram in bigrams:
            if(bigram[0]=="(pause)" or bigram[1]=="(pause)" or \
                bigram[0]=="(uh)" or bigram[1]=="(uh)" or \
                bigram[0]=="(um)" or bigram[1]=="(um)"):
                if bigram not in bigram_p:
                    bigram_p[bigram] = 1
                else:
                    bigram_p[bigram] += 1

    listOfBigrams = [(k, v) for k, v in bigram_p.items()]
    return bigramSort(listOfBigrams)
    

def possibleAlt(sentence, listOfBigrams):
    sentence = sentence.lower()
    tokens = sentence.split()
    # tokens = [START_SYM] + tokens
    possibleBigrams = []
    for token in tokens:
        for j in range(len(listOfBigrams)):
            # FIXME: could be an 'in', clean RHS string 
            if( (token == listOfBigrams[j][0][0]) or (token == listOfBigrams[j][0][1]) ):
                possibleBigrams.append(listOfBigrams[j])
    return bigramSort(possibleBigrams)

def searchDraw(word, draw):
    for it in draw:
        if( (it[0][1] == word) or (it[0][0] == word) ):
            return 1 
    return 0

def returnDraw(word, draw):
    for it in draw:
        if( (it[0][1] == word) or (it[0][0] == word) ):
            return it[0]

def cleanInput(sent):
    sent = sent.lower()
    return sent.replace(".", "") \
                .replace(",", "") \
                .replace("\"", "")


def gen_sentences(sent, choices):
    # Number of choices
    print(choices)
    formed_sentences = []
    for bigram in choices:
        next_word = ''
        prev_word = ''
        sentence = []
        op_sentence = []
        outputSentence = []

        if(bigram[0][0] != "(uh)" and bigram[0][0] != "(um)" and bigram[0][0] != "(pause)"):
            prev_word = bigram[0][0]
            next_word = bigram[0][1]
            print("Previous word is ",prev_word)
            # print(next_word)
            pred_word = next_word.strip("()")
            print("Next word is ", pred_word)
            
            for word in list(sent.split()):
                if(word == prev_word):
                    outputSentence.append(word)
                    gen_word = generate_word(model, tokenizer, max_length-1, outputSentence)
                    break
                else:
                    outputSentence.append(word)
        
            print("Generated word is:", gen_word)
            op_sentence= outputSentence.append(gen_word)
            print("output: ",outputSentence)
            
            if(pred_word == gen_word):
                sentence = ' '.join(word for word in outputSentence)
                print("Correct Prediction \n ")
                formed_sentences.append(sentence)
            else:
                print("Incorrect Prediction \n")

    return formed_sentences
# May the force be with you on this fateful day padawan
# It is our choices ... that show what we truly are, far more than our abilities.
def bigramDriver(inputSentence):
    inputSentence = cleanInput(inputSentence)
    infile = open('./obj/bigram', 'rb')
    bigrams = pickle.load(infile)
    infile.close()
    choices = np.array(possibleAlt(inputSentence, bigrams))

    # percentage = int(PERCENTAGE*(inputSentence.count(" ")+1))
    # draw = choices[choice(choices.shape[0], percentage, p=createDist(choices))]

    outputSentence = []
    for word in list(inputSentence.split()):
        outputSentence.append(word)

    return ' '.join(word for word in outputSentence), choices

def generate_seq(model, tokenizer, max_length, seed_text, n_words):
	in_text = seed_text
	for _ in range(n_words):
		encoded = tokenizer.texts_to_sequences([in_text])[0]
		encoded = pad_sequences([encoded], maxlen=max_length, padding='pre')
		yhat = model.predict_classes(encoded, verbose=0)
		out_word = ''
		for word, index in tokenizer.word_index.items():
			if index == yhat:
				out_word = word
				break
		in_text += ' ' + out_word
	return in_text

def generate_word(model, tokenizer, max_length, seed_text):
    in_text = seed_text
    encoded = tokenizer.texts_to_sequences([in_text])[0]
    encoded = pad_sequences([encoded], maxlen=max_length, padding='pre')
    yhat = model.predict_classes(encoded, verbose=0)
    out_word = ''
    for word, index in tokenizer.word_index.items():
        if index == yhat:
            out_word = word
            break
    return out_word
 
 
def load_data(file_name, size):
    data = ''
    with open(file_name) as f:
        head = [next(f) for x in range(size)]
    data = ''.join(head)
    return data

def model_tol(max_length, vocab_size, X, y, load=True):
    if(load):
        
        model = load_model('./data/model_LSTM_h.h5')
        print("Loaded model from disk.")
        return model
    else:
        model = Sequential()
        model.add(Embedding(vocab_size, 10, input_length=max_length-1))
        model.add(LSTM(100))
        model.add(Dense(vocab_size, activation='softmax'))

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.fit(X, y, epochs=300) # Epochs: 500

        model.save('./data/model_LSTM_h.h5')
        return model

def load_tokenizer():
    tokenizer = Tokenizer()
    data = load_data('./data/reduced.txt', 500) # Max: 10000
    tokenizer.fit_on_texts([data])
    encoded = tokenizer.texts_to_sequences([data])[0]

    vocab_size = len(tokenizer.word_index) + 1
    print('Vocabulary Size: %d' % vocab_size)

    sequences = list()
    for i in range(2, len(encoded)):
        sequence = encoded[i-2:i+1]
        sequences.append(sequence)
    print('Total Sequences: %d' % len(sequences))

    max_length = max([len(seq) for seq in sequences])
    sequences = pad_sequences(sequences, maxlen=max_length, padding='pre')
    print('Max Sequence Length: %d' % max_length)

    sequences = array(sequences)
    X, y = sequences[:,:-1],sequences[:,-1]
    y = to_categorical(y, num_classes=vocab_size)

    return tokenizer, max_length, vocab_size, X, y

def flatten(a):
    forFlatten = []
    for sent in a:
        forFlatten.append(list(sent.split()))
    
    res = np.array(forFlatten)
    res = res.T
    print(res)
    prevPrinted = ""
    finalSentence = []
    for group in res:
        for word in group:
            if(prevPrinted != word):
                prevPrinted = word
                finalSentence.append(word)
    return ' '.join(finalSentence)     

def indexing(a, org):
    indexDict = dict()
    listOrg = list(org.split(' '))
    # print(listOrg)
    inserted = ""
    for sent in a:
        if('pause' in list(sent.split(' '))):
            indexDict[sent] = (list(sent.split(' ')).index('pause'), 'pause')
            inserted = 'pause'
        if('uh' in list(sent.split(' '))):
            indexDict[sent] = (list(sent.split(' ')).index('uh'), 'uh')
            inserted = 'uh'
        if('um' in list(sent.split(' '))):
            indexDict[sent] = (list(sent.split(' ')).index('um'), 'um')
            inserted = 'um'
            
    inOrder = {k: v for k, v in sorted(indexDict.items(), key=lambda item: item[1])}
    # print(inOrder)
    offset = 0
    for key in inOrder:
        listOrg.insert(inOrder[key][0]+offset, inOrder[key][1])
        offset += 1
    
    return ' '.join(listOrg)

def final_sent(formed,sentence,percentage):
    returned_l = sample(formed,percentage)
    return(indexing(returned_l, sentence))

def gen_entire_sentence(sent_list,sentence):
    formedSent = []
    s1 = list(sentence.split(' '))
    l1 = len(s1)
    for sent in sent_list:
        s = list(sent.split(' '))
        l2 = len(s)
        l3 = l2 - 1
        for i in range(l3,l1):
            s.append(s1[i])    
        final = ' '.join(word for word in s)
        formedSent.append(final)
    return formedSent

def gen_p(lf,p):
    lp = p
    if(lp > lf):
        while(lp>lf):
            lp-=1

    return lp


if __name__ == "__main__":
    inputSentence = cleanInput(input())
    choices = []
    sentence , choices = bigramDriver(inputSentence)
    tokenizer, max_length, vocab_size, X, y = load_tokenizer()
    model = model_tol(max_length, vocab_size, X, y, load=True)
    # print(model.summary())
    sent_list = []
    sent_list = (gen_sentences(sentence, choices))
    # print(sent_list)
    print( "Sentences are: ")
    for sent in sent_list:
        print(sent)
    print()
    print()
    # print("sentence is ",sentence)
    print("Formed:")
    formed = gen_entire_sentence(sent_list,sentence)
    print(formed)
    lf = len(formed)
    # print(lf, "is the length")
    percentage = int(PERCENTAGE*(sentence.count(" ")+1))
    # print(percentage)
    fp = gen_p(lf,percentage)
    # print(fp)
    final = final_sent(formed, sentence, fp)
    print("FINAL:")
    print(final)