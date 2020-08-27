from random import random

f = open('op.txt', 'r')

lines = f.readlines()

for line in lines:
    noOfPauses = line.count('pause')

    if(noOfPauses==1):
        print(line.replace('pause', '(pause)'), end="")
    else:
        rand = random()
        if(rand < 0.6363):
            print(line.replace('pause', '(uh)', 2).replace('pause', '(pause)'), end="")
        else:
            print(line.replace('pause', '(um)', 1).replace('pause', '(pause)'), end="")
