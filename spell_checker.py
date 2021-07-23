
import pandas as pd
import parameters
import re
from collections import Counter

def words(text): return re.findall(r'\w+', text.lower())



fdf = pd.read_excel(parameters.training_data)
#message_col=list(fdf['message'])
tadaa = " ".join(list(fdf["message"]))
   

#tadaa = open('/home/rajput/Documents/Fasttext_final/testting/fastText-0.9.1/fastText-0.9.1/saddam70M').read()
tadaa1 = open(parameters.spell_checker_file).read()
tadaa+=tadaa1
word_list=tadaa.split()
words_dict={}
for i in range(len(word_list)):
    words_dict[word_list[i]]=i
# print(type(tadaa))
# print(tadaa)

WORDS = Counter(words(tadaa))

def P(word, N=sum(WORDS.values())): 
    "Probability of `word`."
    return WORDS[word] / N

def correction(word): 
    "Most probable spelling correction for word."
    return max(candidates(word), key=P)

def correction1(word):
    return max(candidates1(word), key=P)



def candidates1(word): 
    "Generate possible spelling corrections for word."
    #return (known([word]) or known(edits1(word)) or known(edits2(word)) or known(edit3(word)) or [word])
    return (known([word]) or known(edits1(word)) or [word])


def candidates(word): 
    "Generate possible spelling corrections for word."
    #return (known([word]) or known(edits1(word)) or known(edits2(word)) or known(edit3(word)) or [word])
    return (known([word]) or known(edits1(word)) or known(edits2(word)) or [word])

def known(words): 
    "The subset of `words` that appear in the dictionary of WORDS."
    return set(w for w in words if w in WORDS)

def edits1(word):
    "All edits that are one edit away from `word`."
    letters    = 'abcdefghijklmnopqrstuvwxyz'
    splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
    deletes    = [L + R[1:]               for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
    replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
    inserts    = [L + c + R               for L, R in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)

def edits2(word): 
    "All edits that are two edits away from `word`."
    return (e2 for e1 in edits1(word) for e2 in edits1(e1))

# def edit3(word):
#     return (e3 for e2 in edits2(word) for e3 in edits2(e2))






def spell_checker(text):
    #print('enter text')
    #text1=input()
    text=text.split()
    modified_text=[]
    for word in text:
        if len(word)<=3:
            modified_text.append(word)

        elif len(word)==4:
            if word not in words_dict:
                modified_text.append(correction1(word))
            else:
                modified_text.append(word)
        
        elif len(word)>4:
            if word not in words_dict:
                modified_text.append(correction(word))
            else:
                modified_text.append(word)
        
    return " ".join(modified_text)
#print(correction('recharg'))

# while True:
#     text=input()
#     print(spell_checker(text))
# while True:
#     print('enter text')
#     text1=input()
#     text=text1.split()
#     modified_text=[]
#     for word in text:
#         if len(word)<=3:
#             modified_text.append(word)
#         else:
#             modified_text.append(correction(word))
#     print(" ".join(modified_text))
#     print(text1)
# #print(correction('recharg'))


