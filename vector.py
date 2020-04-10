import numpy as np
from scipy import spatial
import pickle

from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from keras.models import model_from_json

gloveFile = './data/glove.6B.300d.txt'

def loadGloveModel():
    # f = open(gloveFile, 'r')
    model = {}
    # for line in f:
    #     splitLine = line.split()
    #     word = splitLine[0]
    #     embedding = np.array([float(val) for val in splitLine[1:]])
    #     model[word] = embedding
    # print("Done.", len(model), " words loaded.")
    # pickle.dump( model, open( "./data/model.p", "wb" ) )
    model = pickle.load( open("./data/model.p", "rb") )
    return model



def clean(sentence):
    return sentence.replace(r'"', '') \
                    .replace(r'!', '') \
                    .replace(r'?', '') \
                    .replace(r',', '') \
                    .replace(r')', '') \
                    .replace(r'(', '') \
                    .lower()

def create_vectors(s1, s2, model):
    skipVal = 0
    s1 = list(s1.split())
    s2 = list(s2.split())

    total = len(s1)+len(s2)

    # s1Inner = [model[word] for word in s1]
    s1Inner = []
    for word in s1:
        try:
            s1Inner.append(model[word])
        except KeyError:
            skipVal += 1
            continue

    # s2Inner = [model[word] for word in s2]
    s2Inner = []
    for word in s2:
        try:
            s2Inner.append(model[word])
        except KeyError:
            skipVal += 1
            continue
   
    vector_1 = np.mean(s1Inner,axis=0)
    vector_2 = np.mean(s2Inner,axis=0)
    # print("Skipped:", skipVal/total)
    if((skipVal/total)==1.0):
        return -1
    return [vector_1, vector_2]

def cosine_distance(vector_1, vector_2):
    cosine = spatial.distance.cosine(vector_1, vector_2)
    return (1-cosine)

def train(ann, obs, load=True):
    ann = normalize(np.array(ann), axis=1)
    obs = normalize(np.array(obs), axis=1)
    x = obs
    y = ann

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, 
                                                        random_state=11)

    if(load==False):
        choice = 'tanh'

        model = Sequential()
        model.add(Dense(128, activation=choice, input_dim=300))
        model.add(Dense(512, activation=choice))
        model.add(Dense(1024, activation='relu'))
        model.add(Dense(512, activation=choice))
        model.add(Dense(300, activation=choice))

        model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

        model.fit(x_train, y_train, epochs=20, batch_size=64, shuffle=True, 
                  validation_split=0.1)

        model_json = model.to_json()
        with open("model.json", "w") as json_file:
            json_file.write(model_json)
        model.save_weights("model.h5")
        print("Saved model to disk.")

    else:
        json_file = open('model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        # load weights into new model
        model.load_weights("model.h5")
        print("Loaded model from disk.")
    
    # test(model, x_test, y_test)
    print("x_test", x_test.shape)
    return (model, x_test, y_test)

def test(model, x_test, y_test):
    pred_arr = model.predict(x_test)
    distances = []
    for x, y in zip(pred_arr, y_test):
        distances.append(cosine_distance(x, y))
    
    print("Avg. Dist:", round(sum(distances)/len(distances), 4))
    # errors = []
    # print("Pred", "Actual", "Error", "Error %", sep="\t")
    # for i, j in zip(pred_arr, y_test):
    #     print(round(i[0], 2), j, round(j-i[0], 2), round((j-i[0])/j*100, 2), sep="\t")
    #     errors.append(abs(round((j-i[0])/j*100, 2)))

    # print("Average error %", round(sum(errors)/len(errors), 2))

def create_vector(s, model):
    skipVal = 0
    total = len(s)

    # s1Inner = [model[word] for word in s1]
    sInner = []
    for word in s:
        try:
            sInner.append(model[word])
        except KeyError:
            skipVal += 1
            continue
   
    vector = np.mean(sInner,axis=0)
    # print("Skipped:", skipVal/total)
    if((skipVal/total)==1.0):
        return -1
    return vector

def generate_sentence(sentence, ann_model, glove_model, add_word="pause"):
    listOfSentence = list(sentence.split())
    possibleSentences = []
    for i in range(len(listOfSentence)+1):
        listOfSentence.insert(i, add_word)
        possibleSentences.append(listOfSentence[:])
        listOfSentence.remove(add_word)

    vectors = []
    for sentenceArr in possibleSentences:
        res = create_vector(sentenceArr, glove_model)
        try:
            tempLen = len(res)
        except:
            print("Error.")
            exit()
        vectors.append(res[:])


    target_vector = create_vector(listOfSentence, glove_model)
    target = ann_model.predict(normalize(np.array([target_vector]), axis=1))[0]

    pred_arr = ann_model.predict(normalize(np.array(vectors), axis=1))
    distances = []
    for xy in pred_arr:
        distances.append(np.sum(np.absolute(target - xy)))
    
    for dist in distances:
        print(dist)

if(__name__=="__main__"):
    glove_model = loadGloveModel()

    annVectors = pickle.load( open("./data/annVectors.p", "rb") )
    obsVectors = pickle.load( open("./data/obsVectors.p", "rb") )

    ann_model, x_test, y_test = train(annVectors, obsVectors, load=True)

    # test(model, x_test, y_test)
    sentRic = "but i and"
    generate_sentence(sentRic, ann_model, glove_model)




