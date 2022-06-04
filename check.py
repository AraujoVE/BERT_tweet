with open("trainingData.csv","r") as f:
    lines = f.read().strip().split("\n")
del lines[0]

lula = 0
bolsonaro = 0
both = 0

appreciate = 0

iter = 0

types = [0]*6
for line in lines: types[int(line.split("|")[-1])] += 1

print(f"Appreciation = {sum(types[::2])}\nNon Appreciation = {sum(types[1::2])}\n") 
print(f"Lula =\t\t{sum(types[:2])}\n\tAppreciation =\t\t{types[0]}\n\tNon Appreciation =\t{types[1]}\n")
print(f"Bolsonaro =\t{sum(types[2:4])}\n\tAppreciation =\t\t{types[2]}\n\tNon Appreciation =\t{types[3]}\n")
print(f"Both =\t\t{sum(types[4:])}\n\tAppreciation =\t\t{types[4]}\n\tNon Appreciation =\t{types[5]}\n")
