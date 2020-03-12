import re

f = open("./data/data.txt", "r")
lines = f.readlines()

annFD = open("./data/annotated.txt", "w")
obFD = open("./data/obscured.txt", "w")

def ann(sent):
    repSent = sent.replace("...", " (pause) ") \
                  .replace("..", " (pause) ") \
                  .replace(".", " ") \
                  .replace("---", " (pause) ") \
                  .replace("--", " (pause) ") \
                    

    repSent1 = re.sub(r"^uh+", "(uh)", repSent) 
    repSent2 = re.sub(r"^um+", "(um)", repSent1)
    repSent3 = re.sub(r"\suh+", " (uh)", repSent2) 
    repSent4 = re.sub(r"\sum+", " (um)", repSent3)
    # TODO: multiple spaces
    repSent5 = re.sub(r" - ", "(pause)", repSent4)  
    repSent6 = re.sub(r"-uh+", "(uh)", repSent5) 
    repSent7 = re.sub(r"-um+", "(um)", repSent6)              


    repSentList = list(repSent7.split())
    repSentWOSpaces = ' '.join(repSentList)

    annFD.write(repSentWOSpaces+'\n')
    #annFD.write(sent)

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
