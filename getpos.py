from nltk import pos_tag


f = open('reduced.txt', 'r')
lines = f.readlines()
for line in lines:
    tokens = line.split()
    tokens_tag = tuple(pos_tag(tokens))
    for token in tokens_tag:
        if (token[0] == "(uh)" or token[0] == "(um)" or token[0] == "(pause)"):
            print(token[0], end = " ")
        else:
            print(token[1], end = " ")
    print()
