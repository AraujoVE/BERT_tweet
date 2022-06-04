import pandas as pd
import random
import emoji
import re

#Translate specified dicts and emoticons in text
def translatePatterns(text):
    replaceDict = {"<MEDIA>":"[MEDIA]","<LINK>":"[LINK]"}

    df = pd.DataFrame(text.split('\n'))[0].str.split('|',expand=True) #Create a DataFrame out of the text

    df[4] = df[4].str.replace(r'(:)(\S+?)(:)', r'\2', regex=True) #Replace string with the (:)(\S+?)(:) pattern
    
    df[4] = df[4].apply(lambda text : emoji.demojize(text)) #Convert emojis to text

    extraVocab =  [f"[{emot.upper()}]" for emot in re.findall(r'(?<=:)(\S+?)(?=:)','\n'.join(list(df[4])))] + list(replaceDict.values()) #Save used emoji names + replaceDict values names

    df[4] = df[4].str.replace(r'(:)(\S+?)(:)', lambda x : r'['+x.group(2).upper()+']', regex=True) #Fix emoji names to a pattern

    df[0] = df[df.columns[:]].apply(lambda x: '|'.join(x.dropna().astype(str)),axis=1) #Join columns

    text = '\n'.join(list(df[0])) #Join lines

    for initSymbol,endSymbol in replaceDict.items():
        text = text.replace(initSymbol,endSymbol) #Replace replace dict values

    with open("./extraVocab.csv",'w') as f:
        f.write('vocab\n'+'\n'.join(list(set(extraVocab)))) #Save extra vocab

    return text

#Change binary parts to 0 and 1    
def fixAppreciation(text):
    splittedText = text.split("|")
    splittedText[-1] = "1" if splittedText[-1].split(";")[0] == "Apreciativo" else "0"
    splittedText[-2] = "1" if splittedText[-2] == "True" else "0"
    return "|".join(splittedText)

#Set paths
inputPath = "./lulonaro_training.csv"
outputPath = "./trainingData.csv"

#Generate heading
heading = "link|user|id|date|sentence|response|appreciative".split("|")
headingCols = dict([tuple([i,heading[i]]) for i in range(len(heading))])

with open(inputPath,'r') as f: #Read and fix the data
    textLines = [fixAppreciation(line) for line in translatePatterns(f.read().strip('\n')).split("\n")]    

random.shuffle(textLines)
df = pd.DataFrame(textLines)[0].str.split('|',expand=True)
df.rename(columns=headingCols,inplace=True)

dfAppreciativeLula = df.loc[(df['appreciative'] == '1') & (~df['sentence'].str.contains("bolsonaro",case=False)) & (df['sentence'].str.contains("lula",case=False))].reset_index(drop=True)
dfAppreciativeBolsonaro = df.loc[(df['appreciative'] == '1') & (df['sentence'].str.contains("bolsonaro",case=False)) & (~df['sentence'].str.contains("lula",case=False))].reset_index(drop=True)
dfAppreciativeBoth = df.loc[(df['appreciative'] == '1') & (df['sentence'].str.contains("bolsonaro",case=False)) & (df['sentence'].str.contains("lula",case=False))].reset_index(drop=True)

dfNonAppreciativeLula = df.loc[(df['appreciative'] == '0') & (~df['sentence'].str.contains("bolsonaro",case=False)) & (df['sentence'].str.contains("lula",case=False))].reset_index(drop=True)
dfNonAppreciativeBolsonaro = df.loc[(df['appreciative'] == '0') & (df['sentence'].str.contains("bolsonaro",case=False)) & (~df['sentence'].str.contains("lula",case=False))].reset_index(drop=True)
dfNonAppreciativeBoth = df.loc[(df['appreciative'] == '0') & (df['sentence'].str.contains("bolsonaro",case=False)) & (df['sentence'].str.contains("lula",case=False))].reset_index(drop=True)

minAppreciativeLulaBolsonaro = int(min(
    len(dfAppreciativeLula),
    len(dfAppreciativeBolsonaro),
    len(dfNonAppreciativeLula),
    len(dfNonAppreciativeBolsonaro)
)*0.9)

minAppreciativeBoth = int(min(
    len(dfAppreciativeBoth),
    len(dfNonAppreciativeBoth)
)*0.9)

df = pd.concat([
    dfAppreciativeLula[:minAppreciativeLulaBolsonaro],
    dfNonAppreciativeLula[:minAppreciativeLulaBolsonaro],
    dfAppreciativeBolsonaro[:minAppreciativeLulaBolsonaro],
    dfNonAppreciativeBolsonaro[:minAppreciativeLulaBolsonaro],
    dfAppreciativeBoth[:minAppreciativeBoth],
    dfNonAppreciativeBoth[:minAppreciativeBoth]
]).reset_index(drop=True)
df = df.sample(frac=1).reset_index(drop=True)

df["expandedLabel"] = 0
df.loc[(df['appreciative'] == '1') & (~df['sentence'].str.contains("bolsonaro",case=False)) & (df['sentence'].str.contains("lula",case=False)),"expandedLabel"] = 0
df.loc[(df['appreciative'] == '0') & (~df['sentence'].str.contains("bolsonaro",case=False)) & (df['sentence'].str.contains("lula",case=False)),"expandedLabel"] = 1
df.loc[(df['appreciative'] == '1') & (df['sentence'].str.contains("bolsonaro",case=False)) & (~df['sentence'].str.contains("lula",case=False)),"expandedLabel"] = 2
df.loc[(df['appreciative'] == '0') & (df['sentence'].str.contains("bolsonaro",case=False)) & (~df['sentence'].str.contains("lula",case=False)),"expandedLabel"] = 3
df.loc[(df['appreciative'] == '1') & (df['sentence'].str.contains("bolsonaro",case=False)) & (df['sentence'].str.contains("lula",case=False)),"expandedLabel"] = 4
df.loc[(df['appreciative'] == '0') & (df['sentence'].str.contains("bolsonaro",case=False)) & (df['sentence'].str.contains("lula",case=False)),"expandedLabel"] = 5




df.to_csv(outputPath,sep='|',index=False)