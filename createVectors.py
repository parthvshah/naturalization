from vector import create_vectors, loadGloveModel
import pickle 

model = loadGloveModel()

ann = open('./data/annotated.txt', 'r')
obs = open('./data/obscured.txt', 'r')
annLines = ann.readlines()
obsLines = obs.readlines()

annVectors = []
obsVectors = []

for x, y in zip(annLines, obsLines):
    res = create_vectors(x, y, model)
    if(res==-1):
        continue
    else:
        try:
            compute = len(res[0]) - len(res[1])
        except TypeError:
            continue
        
        annVectors.append(res[0])
        obsVectors.append(res[1])

# annVectors = np.array(annVectors)
# obsVectors = np.array(obsVectors)

pickle.dump( annVectors, open( "./data/annVectors.p", "wb" ) )
pickle.dump( obsVectors, open( "./data/obsVectors.p", "wb" ) )
