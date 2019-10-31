'''
Created on Oct 31, 2019

@author: gregory
'''

from os import listdir
from os.path import isfile, join

def main(section_dir):
    total_lines = 0
    onlyfiles = [f for f in listdir(section_dir) if isfile(join(section_dir, f))]
    for onlyfile in onlyfiles:
        with open(join(section_dir, onlyfile), 'r', encoding='utf-8') as fs:
            lines = fs.readlines()
        for line in lines:
            if len(line) > 0:
                total_lines += 1

    print(total_lines)

if __name__ == '__main__':
    section_dir = '/Users/gregory/eclipse-workspace/Boto/output/sample_texts/vasrd'
    main(section_dir)