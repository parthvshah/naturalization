import numpy as np

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

def create(sentence, add_word="pause"):
    listOfSentence = list(sentence.split())
    possibleSentences = []
    for i in range(len(listOfSentence)+1):
        listOfSentence.insert(i, add_word)
        possibleSentences.append(listOfSentence[:])
        listOfSentence.remove(add_word)

    vectors = []
    for sentenceArr in possibleSentences:
        res = create_vector(sentenceArr, model)
        if(res!=-1):
            vectors.append(res)
        else:
            print("Error.")
            exit()
    
    print(np.array(vectors).shape)
    
    

create("but i gave it absolutely everything and more today i pushed so hard and i really didn't have a lot of pace")