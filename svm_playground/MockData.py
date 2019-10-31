'''
Created on Oct 29, 2019

@author: gregory
'''

import numpy as np

def create_toy_diagnostic_code():
    ratings = []
    ratings.append(('6500', 'Acute heart failure', '100', '50'))
    return ratings

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