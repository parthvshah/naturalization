from bigram import bigramDriver

inFile = 'in.txt'

IF = open(inFile, 'r')

inputLines = IF.readlines()

for line in inputLines:
    print(bigramDriver(line))

