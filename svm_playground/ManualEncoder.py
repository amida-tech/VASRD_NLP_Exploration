'''
Created on Oct 30, 2019

@author: gregory
'''
import numpy as np

entities = dict()

def read_all_entities(filename):
    with open(filename, 'r', encoding='utf-8') as fs:
        lines = fs.readlines()
    index = 0
    for line in lines:
        entity_value = line.strip()
        entities[entity_value] = index
        index += 1

read_all_entities('../output/aws_entity_list')

def multilabel_encode(ners):
    print(ners)
    X = np.zeros((len(ners), len(entities)))
    row = 0
    for ner_tuple in ners:
        for items in ner_tuple:
            for item in items:
                if item not in entities:
                    continue
                X[row][entities[item]]
        row +=1
    return X
