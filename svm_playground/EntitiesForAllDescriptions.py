'''
Created on Oct 30, 2019

@author: gregory
'''

# This file should only be run once!  It captures all the entities from the AWS
# MEDICAL COMPREHEND service on disc which can then be accessed at a later time
# in whatever fashion we require

import boto3
import json
from os.path import join

client = boto3.client(service_name='comprehendmedical', region_name='us-west-2')

ratings = []

def is_number(some_string):
    try: 
        int(some_string)
        return True
    except ValueError:
        return False

def read_ratings_information(ratings_file):
    with open(ratings_file, 'r', encoding='utf-8') as fs:
        lines = fs.readlines()
    for line in lines:
        # first capture diagnostic code
        parts = line.strip().split('   ')
        diagnostic_code = parts[0]
        rest_of_line = parts[1]
        # secondly capture ratings.  There should be either 1 or 2
        minor_parts = rest_of_line.split(' ')
        minor_code = minor_parts[len(minor_parts) - 1]
        rest_of_line = ' '.join(minor_parts[0:-1])
        major_code = None
        if len(minor_parts) > 2:
            minor_parts = rest_of_line.split(' ')
            last_item = minor_parts[len(minor_parts) - 1]
            if is_number(last_item):
                major_code = last_item
                rest_of_line = ' '.join(minor_parts[0:-1])
        #@ third description of rating
        description = rest_of_line
        
        ratings.append((diagnostic_code, description, major_code, minor_code))


def tag_entities(ratings_file):
    read_ratings_information(ratings_file)
    loop = 1
    for rating in ratings:
        result = client.detect_entities_v2(Text = rating[1])
        entities = result['Entities'];
        inner_loop = 1
        for entity in entities:
            with open(join('..', 'output', 'aws_entities', 'doc_' + str(loop)) + '_' + str(inner_loop), 'w', encoding='utf-8') as fs:
                json.dump(entity, fs)
            inner_loop += 1
        loop += 1

if __name__ == '__main__':
    ratings_file = '../data/all_ratings_good.txt'
    tag_entities(ratings_file)