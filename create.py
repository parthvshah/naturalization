import re

f = open("data.txt", "r")
lines = f.readlines()

annFD = open("annotated.txt", "w")
obFD = open("obscured.txt", "w")

def ann(sent):


    annFD.write(sent)

def ob(sent):
    repSent = sent.replace(".", " ") \
                    .replace("..", " ") \
                    .replace("...", " ") \
                    .replace("-", " ") \
                    .replace("--", " ") \
                    .replace("---", " ") \
                    .replace("uh", " ") \
                    .replace("um", " ")

    repSentList = list(repSent.split())
    repSentWOSpaces = ' '.join(repSentList)

    obFD.write(repSentWOSpaces+'\n')

for line in lines:
    low = line.lower()
    if( ('..' in low) or 
        ('--' in low) or 
        ('uh' in low) or 
        ('um' in low)):
        rActions = re.sub(r"\((.*?)\)", "", low)
        ann(rActions)
        # ann.write(low)
        ob(rActions)
        # ob.write(low)
