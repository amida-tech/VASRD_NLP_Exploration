'''
Created on Oct 30, 2019

@author: gregory
'''

import json
from os import listdir
from os.path import isfile, join

value_set = set()

def process_entity(content):
    # we just care about text for right now
    entity_text = content['Text']
    value_set.add(entity_text.lower())

def main(data_dir, entity_file):
    onlyfiles = [f for f in listdir(data_dir) if isfile(join(data_dir, f))]
    for onlyfile in onlyfiles:
        with open(join(data_dir, onlyfile), 'r', encoding='utf-8') as fs:
            content = json.load(fs)
        process_entity(content)

    
    value_list = list(value_set)
    value_list.sort()
    
    with open(entity_file, 'w', encoding='utf-8') as fs:
        for vl in value_list:
            fs.write(vl + '\n')

if __name__ == '__main__':
    data_dir = '../output/aws_entities'
    all_entity_file = '../output/aws_entity_list'
    main(data_dir, all_entity_file)