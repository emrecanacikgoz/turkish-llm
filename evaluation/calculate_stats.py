import numpy as np
import json
import pandas as pd
from pathlib import Path
import os

len_arc = 1172
n = 3
with open('arc/annotations_emre.json', 'r') as json_file:
    cache1 = json.load(json_file)["false_indexes"]
    res1 = [(0 if i in cache1 else 1) for i in range(len_arc)]

with open('arc/annotations_oguz.json', 'r') as json_file:
    cache2 = json.load(json_file)["false_indexes"]
    res2 = [(0 if i in cache2 else 1) for i in range(len_arc)]

with open('arc/annotations_mete.json', 'r') as json_file:
    cache3 = json.load(json_file)["false_indexes"]
    res3 = [(0 if i in cache3 else 1) for i in range(len_arc)]

union = cache1 + cache2 + cache3
all = np.sort(list(set(list((union)))))
print("arc:")
print("Union false samples: ", len(all))

tot = 0
r = n*len_arc
P = np.zeros((len_arc))
all_ones = 0
all_zeros = 0
for i in range(len_arc):
    ones = 0
    zeros = 0
    if(res1[i] == 1):
        ones += 1
        all_ones += 1
    else:
        zeros += 1
        all_zeros += 1

    if(res2[i] == 1):
        ones += 1
        all_ones += 1
    else:
        zeros += 1
        all_zeros += 1

    if(res3[i] == 1):
        ones += 1
        all_ones += 1
    else:
        zeros += 1
        all_zeros += 1
    P[i] = (1/6)*(ones**2 + zeros**2 -3)
    if res1[i] == res2[i] and res2[i] == res3[i] and res1[i] == res3[i]:
        tot +=1

P_head = np.sum(P)/len_arc
P_head_e = (all_ones/(3*len_arc))**2 + (all_zeros/(3*len_arc))**2
kappa = (P_head - P_head_e)/(1 - P_head_e)
print("Raw Agreement: ", tot/len_arc)
print("Fleiss' Kappa: ", kappa)
print()


len_arc = 817
n = 3
with open('truthfulqa/annotations_emre.json', 'r') as json_file:
    cache1 = json.load(json_file)["false_indexes"]
    res1 = [(0 if i in cache1 else 1) for i in range(len_arc)]

with open('truthfulqa/annotations_oguz.json', 'r') as json_file:
    cache2 = json.load(json_file)["false_indexes"]
    res2 = [(0 if i in cache2 else 1) for i in range(len_arc)]

with open('truthfulqa/annotations_mete.json', 'r') as json_file:
    cache3 = json.load(json_file)["false_indexes"]
    res3 = [(0 if i in cache3 else 1) for i in range(len_arc)]

union2 = cache1 + cache2 + cache3
all2 = np.sort(list(set(list((union2)))))
print("truthfulqa:")
print("Union false samples: ", len(all2))

tot = 0
r = n*len_arc
P = np.zeros((len_arc))
all_ones = 0
all_zeros = 0
for i in range(len_arc):
    ones = 0
    zeros = 0
    if(res1[i] == 1):
        ones += 1
        all_ones += 1
    else:
        zeros += 1
        all_zeros += 1

    if(res2[i] == 1):
        ones += 1
        all_ones += 1
    else:
        zeros += 1
        all_zeros += 1

    if(res3[i] == 1):
        ones += 1
        all_ones += 1
    else:
        zeros += 1
        all_zeros += 1
    P[i] = (1/6)*(ones**2 + zeros**2 -3)
    if res1[i] == res2[i] and res2[i] == res3[i] and res1[i] == res3[i]:
        tot +=1

P_head = np.sum(P)/len_arc
P_head_e = (all_ones/(3*len_arc))**2 + (all_zeros/(3*len_arc))**2
kappa = (P_head - P_head_e)/(1 - P_head_e)
print("Raw Agreement: ", tot/len_arc)
print("Fleiss' Kappa: ", kappa)
print()