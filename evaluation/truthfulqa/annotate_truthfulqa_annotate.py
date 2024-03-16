import json
import pandas as pd
from pathlib import Path
import os
import numpy as np

len_arc = 817
n = 3
with open('annotations_emre.json', 'r') as json_file:
    cache1 = json.load(json_file)["false_indexes"]
    res1 = [(0 if i in cache1 else 1) for i in range(len_arc)]

with open('annotations_oguz.json', 'r') as json_file:
    cache2 = json.load(json_file)["false_indexes"]
    res2 = [(0 if i in cache2 else 1) for i in range(len_arc)]

with open('annotations_mete.json', 'r') as json_file:
    cache3 = json.load(json_file)["false_indexes"]
    res3 = [(0 if i in cache3 else 1) for i in range(len_arc)]

union2 = cache1 + cache2 + cache3
all2 = np.sort(list(set(list((union2)))))

with open('truthfulqa_tr.jsonl', 'r') as json_file:
    json_list = list(json_file)
arc_tr = []
for json_str in json_list:
    arc_tr.append(json.loads(json_str))

with open('tfqa_corrected_oguz.jsonl', 'r') as json_file2:
    json_list2 = list(json_file2)
arc_tr2 = []
for json_str2 in json_list2:
    arc_tr2.append(json.loads(json_str2))

with open('truthfulqa_en.jsonl', 'r') as json_file:
    json_list = list(json_file)
arc_en = []
for json_str in json_list:
    arc_en.append(json.loads(json_str))
    

length = len(arc_tr)
valid_indexes = []
jj = 0

while jj<len(all2):
    i = all2[jj]
    print(" ------------------ Sample: ", i, " ------------------")
    en_sample = arc_en[i]
    tr_sample = arc_tr[i]
    tr2_sample = arc_tr2[jj]
    print("Q-En: ", en_sample["question"])
    print("Q-Tr: ", tr2_sample["question"])
    print("Q-Tr: ", tr_sample["question"])
    print()

    for j in range(len(en_sample["mc1_targets"]["choices"])):
        print("mc1|En-",str(j+1),": ",en_sample["mc1_targets"]["choices"][j])
        print("mc1|Tr-",str(j+1),": ",tr2_sample["mc1_targets"]["choices"][j])
        print("mc1|Tr-",str(j+1),": ",tr_sample["mc1_targets"]["choices"][j])
        print()
    
    for j in range(len(en_sample["mc2_targets"]["choices"])):
        print("mc2|En-",str(j+1),": ",en_sample["mc2_targets"]["choices"][j])
        print("mc2|Tr-",str(j+1),": ",tr2_sample["mc2_targets"]["choices"][j])
        print("mc2|Tr-",str(j+1),": ",tr_sample["mc2_targets"]["choices"][j])
        print()
        
    print()
    print("Is translation valid?:")
    res = input("Write 't' for true, 'f' for false, 'r' to return previous, 'e' to save and exit: ")
    if(len(res)<1):
        i-=1
    else:
        if(res[0] == "t"):
            jj+=0
        elif(res[0] == "r"):
            jj-=2
        else:
            jj-=1
   
    os.system("clear")
    jj+=1

