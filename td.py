import re


with open("trainingData_Old.csv","r") as f:
    #terms = f.read().replace("[MEDIA]","[!media!]").replace("[LINK]","[!link!]")
    terms = f.read()
terms = re.sub(r'\s+\[[A-Z_]+\]\s+',' ',terms)
terms = re.sub(r'\s+\[[A-Z_]+\]','',terms)
terms = re.sub(r'\[[A-Z_]+\]\s+','',terms)
#terms = terms.replace("[!media!]","[MEDIA]").replace("[!link!]","[LINK]")

with open("trainingData.csv","w") as f:
    f.write(terms)