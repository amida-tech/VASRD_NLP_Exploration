'''
Created on Oct 29, 2019

@author: gregory
'''

import numpy as np

def create_toy_diagnostic_code():
    ratings = []
    ratings.append(('6500', 'Acute heart failure', '100', '50'))
    return ratings

# These roughly correspond to the 4 ner rows, this I am creating 4 documents
def create_toy_docs():
    docs = []
    # no relevant medical terms (I hope)
    docs.append('The Astros were one game away from winning their second World Series title in three years, before the Nationals won two straight games in Houston to clinch the 2019 title. The club is expected to return the majority of their potent lineup in 2020, and they will once again be the team to beat in the 2020 season. Gerrit Cole will be a free agent at the conclusion of the 2019 season, and even if he departs, Houston will be bringing back Justin Verlander and Zack Greinke to lead the pitching staff.')
    docs.append('24 yo male who lives in Ontario Canada has history of recurrent apraxia when he goes outside in the cold during the winter months for the past few winters.  He had a history of apraxia of the skull and had been on aspirin 100 mg at noon for four years. He was scheduled to undergo colonscopy.')
    docs.append('The clavicle is a long bone. There are two clavicles, one on the left and one on the right. The clavicle is the only long bone in the body that lies horizontally.  The 5 ft 7 in woman had a clavicle.')
    docs.append('Perform wedge-shape amputation on the radial bone according to the above procedure. If stroke is indicated, perform amputation on the femur instead. Amputation should not be performed if the patient has influenza.')
    return docs

# The NER Types are as follows
# ANATOMY, MEDICAL_CONDITION, MEDICATION, PROTECTED_HEALTH_INFORMATION and TEST_TREATMENT_PROCEDURE
def create_toy_ners():
    ners = []
    ners.append(([], [], [], [], []))
    ners.append((['Skull'], ['Apraxia'], ['Aspirin'], ['24 yo'], ['Colonscopy']))
    ners.append((['Clavicle'], [], [], ['5 ft 7 in'], []))
    ners.append((['Radial Bone', 'Femur'], ['Stroke', 'Influenza', 'Amputation'], [], [], []))
    return ners

def create_toy_ners_numpy():
    ners = np.zeros(shape=(4,5))
    ners = np.vstack([ners, ([], [], [], [], [])])
    ners = np.vstack([ners, (['Skull'], ['Apraxia'], ['Aspirin'], ['24 yo'], ['Colonscopy'])])
    ners = np.vstack([ners, (['Clavicle'], [], [], ['5 ft 7 in'], [])])
    ners = np.vstack([ners, (['Radial Bone', 'Femur'], ['Stroke', 'Influenza', 'Amputation'], [], [], [])])
    return ners

def create_toy_ners_predictions():
    y = [['1000', '50'],['2000', '30'],['1000', '30'],['3000', '40']]
    return y