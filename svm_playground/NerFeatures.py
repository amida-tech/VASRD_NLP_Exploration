'''
Created on Oct 29, 2019

@author: gregory
'''

from joblib import dump, load
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import LinearSVC
from svm_playground import ManualEncoder, MockData

ratings = []
features = []
vectorizer = CountVectorizer()

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

def get_bag_of_words_feature():
    descriptions = [rating[1] for rating in ratings]
    X = vectorizer.fit_transform(descriptions)
    dump(vectorizer, 'model/cv')
    return X

def get_ner_feature(ners):
    x = ManualEncoder.multilabel_encode(ners)    
    return x

def featurize():
    features.append(get_bag_of_words_feature())
    features.append(get_ner_feature())

#TODO
def combine_features(selected):
    pass

def train(x, ner_predictions, model_output):
    y = np.array(ner_predictions)

    svm_classifier = LinearSVC(random_state=0, tol=1e-5, max_iter=100000)
    classifier = MultiOutputClassifier(svm_classifier, n_jobs=-1)
    classifier.fit(x, y)

    dump(classifier, model_output)     
    
    print(classifier.predict(x))

def main(model_output):
    #read_ratings_information(ratings_file)
    ners = MockData.create_toy_ners()
    ner_predictions = MockData.create_toy_ners_predictions()
    X = get_ner_feature(ners)
    train(X, ner_predictions, model_output)

if __name__ == '__main__':
    model_output = '../model/svm_ner_model'
    main(model_output)