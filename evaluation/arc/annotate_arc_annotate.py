import json
import pandas as pd
from pathlib import Path
import os
import numpy as np

len_arc = 1172
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

union = cache1 + cache2 + cache3
all = np.sort(list(set(list((union)))))


with open('arc_corrected_oguz.jsonl', 'r') as json_file:
    json_list = list(json_file)
arc_tr = []
for json_str in json_list:
    arc_tr.append(json.loads(json_str))

with open('arc-tr.jsonl', 'r') as json_file2:
    json_list2 = list(json_file2)
arc_tr2 = []
for json_str2 in json_list2:
    arc_tr2.append(json.loads(json_str2))

with open('arc-en.jsonl', 'r') as json_file:
    json_list = list(json_file)
arc_en = []
for json_str in json_list:
    arc_en.append(json.loads(json_str))

length = len(arc_tr)
valid_indexes = []
jj = 0

while jj<len(all):
    i = int(all[jj])
    print(" ------------------ Sample: ", i, "/",jj," ------------------")
    en_sample = arc_en[i]
    tr_sample = arc_tr[jj]
    tr_sample2 = arc_tr2[i]
    print("Q-En: ", en_sample["question"]["stem"])
    print("Q-Tr: ", tr_sample["question"])
    print("Q-T2: ", tr_sample2["question"])
    print()

    for j in range(len(en_sample["question"]["choices"])):
        print("En: ",en_sample["question"]["choices"][j]["label"], ": ",en_sample["question"]["choices"][j]["text"])
        print("Tr: ",tr_sample["choices"]["label"][j], ": ",tr_sample["choices"]["text"]["text"][j])
        print("T2: ",tr_sample2["choices"]["label"][j], ": ",tr_sample2["choices"]["text"]["text"][j])
        print()
        
    print()
    print("Is translation valid?:")
    res = input("Write 't' for true, 'f' for false, 'r' to return previous, 'e' to save and exit: ")
    if(len(res)<1):
        jj-=1
    else:
        if(res[0] == "t"):
            jj+=0
        elif(res[0] == "r"):
            jj-=2
        else:
            jj-=1
   
    os.system("clear")
    jj+=1

