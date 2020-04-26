from hybrid import hybrid_driver

def sanitize(line):
    return line.replace("(pause)", "") \
                .replace("(um)", "") \
                .replace("(uh)", "")

inFile = 'in.txt'

IF = open(inFile, 'r')

inputLines = IF.readlines()

for line in inputLines:
    if(len(line)<=2):
        print()
        continue
    sanitizedLine = sanitize(line)
    print(hybrid_driver(sanitizedLine))

