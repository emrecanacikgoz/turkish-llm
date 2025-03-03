import json
import pandas as pd
from pathlib import Path
import os

cache = {"left_index":0, "false_indexes":[]}
already_saved = False
my_file = Path("annotations.json")
if my_file.is_file():
    results_saved = True
    with open('annotations.json', 'r') as json_file:
        cache = json.load(json_file)

with open("annotations.json", "w") as file:
    json.dump(cache, file)

with open('arc-challange-test-tr-231221.jsonl', 'r') as json_file:
    json_list = list(json_file)
arc_tr = []
for json_str in json_list:
    arc_tr.append(json.loads(json_str))

with open('arc-challenge-test-en.jsonl', 'r') as json_file:
    json_list = list(json_file)
arc_en = []
for json_str in json_list:
    arc_en.append(json.loads(json_str))

length = len(arc_tr)
start = cache["left_index"]
valid_indexes = []
i = start

while i<len(arc_en):
    print(" ------------------ Sample: ", i, " ------------------")
    en_sample = arc_en[i]
    tr_sample = arc_tr[i]
    print("Q-En: ", en_sample["question"]["stem"])
    print("Q-Tr: ", tr_sample["question"])
    print()

    for j in range(len(en_sample["question"]["choices"])):
        print("En: ",en_sample["question"]["choices"][j]["label"], ": ",en_sample["question"]["choices"][j]["text"])
        print("Tr: ",tr_sample["choices"]["label"][j], ": ",tr_sample["choices"]["text"]["text"][j])
        print()
        
    print()
    print("Is translation valid?:")
    res = input("Write 't' for true, 'f' for false, 'r' to return previous, 'e' to save and exit: ")
    if(len(res)<1):
        i-=1
    else:
        if(res[0] == "t"):
            if(i in cache["false_indexes"]):
                cache["false_indexes"].remove(i)
            with open("annotations.json", "w") as file:
                json.dump(cache, file)
        elif(res[0] == "f"):
            if(i in cache["false_indexes"]):
                cache["false_indexes"].remove(i)
            cache["false_indexes"].append(i)
            with open("annotations.json", "w") as file:
                json.dump(cache, file)
        elif(res[0] == "e"):
            with open("annotations.json", "w") as file:
                json.dump(cache, file)
            break
        elif(res[0] == "r"):
            i-=2
        else:
            i-=1
   
    cache["left_index"] = i
    os.system("clear")
    i+=1

