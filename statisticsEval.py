with open("lulonaro_training.csv","r") as f:
    lines = f.read().strip().split("\n")


annotators = {} 
for line in lines:
    url, user, id, data, sentence, response, appreciation = line.split("|")
    chosenAppreciation, appreciators = appreciation.split(";")
    appreciators = appreciators.split(",")
    for appreciator in appreciators:
        agent, appType = appreciator.split(":")
        if not agent in annotators: 
            annotators[agent] = {
                "Total": 0,
                "Apreciativo": {
                    "Total": 0,
                    "Correct": 0,
                    "Wrong": 0
                },
                "Não_Apreciativo": {
                    "Total": 0,
                    "Correct": 0,
                    "Wrong": 0
                }
            }
        annotators[agent]["Total"] += 1
        annotators[agent][appType]["Total"] += 1
        annotators[agent][appType][("Correct" if chosenAppreciation == appType else "Wrong")] += 1



with open("lulonaro_training_doubt.csv","r") as f:
    lines = f.read().strip().split("\n")


annotators_Doubt = {}

for line in lines:
    url, user, id, data, sentence, response, appreciation = line.split("|")
    appreciators = appreciation.split(";")[1].split(",")
    typeOfAppreciation = "".join(sorted([appreciator.split(":")[1][0] for appreciator in appreciators]))
    for appreciator in appreciators:
        agent, appType = appreciator.split(":")
        if not agent in annotators_Doubt: 
            annotators_Doubt[agent] = {
                "Total": 0,
                "Apreciativo": {
                    "Total" : 0,
                    "AAI" : 0,
                    "AIN" : 0,
                    "AAN" : 0,
                    "II" : 0,
                    "AII" : 0,
                    "IIN" : 0,
                    "ANN" : 0,
                    "INN" : 0                
                },
                "Não_Apreciativo": {
                    "Total" : 0,
                    "AAI" : 0,
                    "AIN" : 0,
                    "AAN" : 0,
                    "II" : 0,
                    "AII" : 0,
                    "IIN" : 0,
                    "ANN" : 0,
                    "INN" : 0                
                },
                "Indefinido": {
                    "Total" : 0,
                    "AAI" : 0,
                    "AIN" : 0,
                    "AAN" : 0,
                    "II" : 0,
                    "AII" : 0,
                    "IIN" : 0,
                    "ANN" : 0,
                    "INN" : 0                
                }
            }
        annotators_Doubt[agent]["Total"] += 1
        annotators_Doubt[agent][appType]["Total"] += 1
        annotators_Doubt[agent][appType][typeOfAppreciation] += 1



appreciationDict = {
    "AAI" : "Apreciativo_Apreciativo_Indefinido",
    "AAN" : "Apreciativo_Apreciativo_NaoApreciativo",
    "AII" : "Apreciativo_Indefinido_Indefinido",
    "AIN" : "Apreciativo_Indefinido_NaoApreciativo",
    "ANN" : "Apreciativo_NaoApreciativo_NaoApreciativo",
    "II" : "Indefinido_Indefinido",
    "IIN" : "Indefinido_Indefinido_NaoApreciativo",
    "INN" : "Indefinido_NaoApreciativo_NaoApreciativo"
}



def printAnnotator(name,annotation):
    print(f"{name}:")
    print(f"\tTotal Evaluations:{annotation['Total']}")

    print(f"\tAppreciation:")
    print(f"\t\tTotal Appreciations:{annotation['Apreciativo']['Total']}")
    print(f"\t\t\t'Correct':{annotation['Apreciativo']['Correct']}")
    print(f"\t\t\t'Wrong':{annotation['Apreciativo']['Wrong']}")

    print(f"\tNon Appreciation:")
    print(f"\t\tTotal Non Appreciations:{annotation['Não_Apreciativo']['Total']}")
    print(f"\t\t\t'Correct':{annotation['Não_Apreciativo']['Correct']}")
    print(f"\t\t\t'Wrong':{annotation['Não_Apreciativo']['Wrong']}")


def printAnnotatorDoubt(name,annotation):
    print(f"{name}:")
    print(f"\tTotal Evaluations:{annotation['Total']}")

    print(f"\tAppreciation:")
    print(f"\t\tTotal Appreciations: {annotation['Apreciativo']['Total']}")
    print(f"\t\t\tApreciativo/Apreciativo/Indefinido: {annotation['Apreciativo']['AAI']}")
    print(f"\t\t\tApreciativo/Indefinido/Não Apreciativo: {annotation['Apreciativo']['AIN']}")
    print(f"\t\t\tApreciativo/Apreciativo/Não Apreciativo: {annotation['Apreciativo']['AAN']}")
    print(f"\t\t\tIndefinido/Indefinido: {annotation['Apreciativo']['II']}")
    print(f"\t\t\tApreciativo/Indefinido/Indefinido: {annotation['Apreciativo']['AII']}")
    print(f"\t\t\tIndefinido/Indefinido/Não Apreciativo: {annotation['Apreciativo']['IIN']}")
    print(f"\t\t\tApreciativo/Não Apreciativo/Não Apreciativo: {annotation['Apreciativo']['ANN']}")
    print(f"\t\t\tIndefinido/Não Apreciativo/Não Apreciativo: {annotation['Apreciativo']['INN']}")

    print(f"\tNon Appreciation:")
    print(f"\t\tTotal Non Appreciations: {annotation['Não_Apreciativo']['Total']}")
    print(f"\t\t\tApreciativo/Apreciativo/Indefinido: {annotation['Não_Apreciativo']['AAI']}")
    print(f"\t\t\tApreciativo/Indefinido/Não Apreciativo: {annotation['Não_Apreciativo']['AIN']}")
    print(f"\t\t\tApreciativo/Apreciativo/Não Apreciativo: {annotation['Não_Apreciativo']['AAN']}")
    print(f"\t\t\tIndefinido/Indefinido: {annotation['Não_Apreciativo']['II']}")
    print(f"\t\t\tApreciativo/Indefinido/Indefinido: {annotation['Não_Apreciativo']['AII']}")
    print(f"\t\t\tIndefinido/Indefinido/Não Apreciativo: {annotation['Não_Apreciativo']['IIN']}")
    print(f"\t\t\tApreciativo/Não Apreciativo/Não Apreciativo: {annotation['Não_Apreciativo']['ANN']}")
    print(f"\t\t\tIndefinido/Não Apreciativo/Não Apreciativo: {annotation['Não_Apreciativo']['INN']}")

    print(f"\t\tTotal Undefined: {annotation['Indefinido']['Total']}")
    print(f"\t\t\tApreciativo/Apreciativo/Indefinido: {annotation['Indefinido']['AAI']}")
    print(f"\t\t\tApreciativo/Indefinido/Não Apreciativo: {annotation['Indefinido']['AIN']}")
    print(f"\t\t\tApreciativo/Apreciativo/Não Apreciativo: {annotation['Indefinido']['AAN']}")
    print(f"\t\t\tIndefinido/Indefinido: {annotation['Indefinido']['II']}")
    print(f"\t\t\tApreciativo/Indefinido/Indefinido: {annotation['Indefinido']['AII']}")
    print(f"\t\t\tIndefinido/Indefinido/Não Apreciativo: {annotation['Indefinido']['IIN']}")
    print(f"\t\t\tApreciativo/Não Apreciativo/Não Apreciativo: {annotation['Indefinido']['ANN']}")
    print(f"\t\t\tIndefinido/Não Apreciativo/Não Apreciativo: {annotation['Indefinido']['INN']}")

print("###Correct Annotations###")
for annotator in annotators.keys():
    printAnnotator(annotator,annotators[annotator])

print("\n###Undefined Annotations###")
for annotator in annotators_Doubt.keys():
    printAnnotatorDoubt(annotator,annotators_Doubt[annotator])

for line in lines:
    url, user, id, data, sentence, response, appreciation = line.split("|")
    appreciators = appreciation.split(";")[1].split(",")
    typeOfAppreciation = appreciationDict["".join(sorted([appreciator.split(":")[1][0] for appreciator in appreciators]))]
    with open(f"inDoubt/{typeOfAppreciation}.txt","a") as f: f.write(f"{line}\n")
