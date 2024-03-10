import json
import pandas as pd
from pathlib import Path
import os

cache = {"left_index":0, "false_indexes":[]}
already_saved = False
my_file = Path("annotations_qa.json")
if my_file.is_file():
    results_saved = True
    with open('annotations_qa.json', 'r') as json_file:
        cache = json.load(json_file)

with open("annotations_qa.json", "w") as file:
    json.dump(cache, file)

with open('tfqa_tr_correct.jsonl', 'r') as json_file:
    json_list = list(json_file)
arc_tr = []
for json_str in json_list:
    arc_tr.append(json.loads(json_str))

with open('tfqa_tr.jsonl', 'r') as json_file2:
    json_list2 = list(json_file2)
arc_tr2 = []
for json_str2 in json_list2:
    arc_tr2.append(json.loads(json_str2))

with open('tfqa_en.jsonl', 'r') as json_file:
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
    tr2_sample = arc_tr2[i]
    print("Q-En: ", en_sample["question"])
    print("Q-Tr: ", tr_sample["question"])
    print()

    for j in range(len(en_sample["mc1_targets"]["choices"])):
        print("mc1|En-",str(j+1),": ",en_sample["mc1_targets"]["choices"][j])
        print("mc1|Tr-",str(j+1),": ",tr_sample["mc1_targets"]["choices"][j])
        print()
    
    for j in range(len(en_sample["mc2_targets"]["choices"])):
        print("mc2|En-",str(j+1),": ",en_sample["mc2_targets"]["choices"][j])
        print("mc2|Tr-",str(j+1),": ",tr_sample["mc2_targets"]["choices"][j])
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
            with open("annotations_qa.json", "w") as file:
                json.dump(cache, file)
        elif(res[0] == "f"):
            if(i in cache["false_indexes"]):
                cache["false_indexes"].remove(i)
            cache["false_indexes"].append(i)
            with open("annotations_qa.json", "w") as file:
                json.dump(cache, file)
        elif(res[0] == "e"):
            with open("annotations_qa.json", "w") as file:
                json.dump(cache, file)
            break
        elif(res[0] == "r"):
            i-=2
        else:
            i-=1
   
    cache["left_index"] = i
    os.system("clear")
    i+=1

