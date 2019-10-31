'''
Created on Oct 29, 2019

@author: gregory
'''

from joblib import dump
import numpy as np
import scipy as sp
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.svm import LinearSVC
from svm_playground import ManualEncoder, MockData

ratings = []
vectorizer = CountVectorizer()
t_vectorizer = TfidfVectorizer(use_idf=False)
ti_vectorizer = TfidfVectorizer(use_idf=True)

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

def get_bag_of_words_from_rating_feature():
    descriptions = [rating[1] for rating in ratings]
    X = vectorizer.fit_transform(descriptions)
    dump(vectorizer, 'model/cv')
    return X

def get_tf_idf_feature():
    descriptions = [rating[1] for rating in ratings]
    X = ti_vectorizer.fit_transform(descriptions)
    dump(ti_vectorizer, 'model/tf_idf')
    return X
    
def get_normalized_tf_feature():
    descriptions = [rating[1] for rating in ratings]
    X = t_vectorizer.fit_transform(descriptions)
    dump(t_vectorizer, 'model/tf_nn')
    return X

def get_bag_of_words_from_mock_data_feature(count_vectorizer_model):
    descriptions = MockData.create_toy_docs()
    X = vectorizer.fit_transform(descriptions)
    dump(vectorizer, count_vectorizer_model)
    return X

def get_tf_idf_from_mock_data_feature(tf_idf_vectorizer_model):
    descriptions = MockData.create_toy_docs()
    X = ti_vectorizer.fit_transform(descriptions)
    dump(ti_vectorizer, tf_idf_vectorizer_model)
    return X

def get_normalized_tf_from_mock_data_feature(tf_nn_vectorizer_model):
    descriptions = MockData.create_toy_docs()
    X = t_vectorizer.fit_transform(descriptions)
    dump(t_vectorizer, tf_nn_vectorizer_model)
    return X

def get_ner_feature(ners):
    X = ManualEncoder.multilabel_encode(ners)  
    return X

def combine_features(selected):
    # this is hard to generalized based upon the extent of the features
    # however, right now we assume 1 is bag of words and 0 is ner
    # we SHOULD be able to add other features as we need to without too many more tears.
    X = sp.sparse.hstack((selected[1],selected[0]),format='csr')
    return X

def train(x, ner_predictions, model_output):
    y = np.array(ner_predictions)

    svm_classifier = LinearSVC(random_state=0, tol=1e-5, max_iter=100000)
    classifier = MultiOutputClassifier(svm_classifier, n_jobs=-1)
    classifier.fit(x, y)

    dump(classifier, model_output)
    
    print(classifier.predict(x))

def main(model_output, count_vectorizer_model, tf_idf_vectorizer_model, tf_nn_vectorizer_model):
    ners = MockData.create_toy_ners()
    ner_predictions = MockData.create_toy_ners_predictions()
    X1 = get_ner_feature(ners)
    X2 = get_tf_idf_from_mock_data_feature(tf_idf_vectorizer_model)
    X = combine_features([X1, X2])
    train(X, ner_predictions, model_output)

if __name__ == '__main__':
    model_output = '../model/svm_ner_model'
    count_vectorizer_model = '../model/cv_model'
    tf_idf_vectorizer_model = '../model/tf_idf_model'
    tf_nn_vectorizer_model = '../model/tf_nn_model'
    main(model_output, count_vectorizer_model, tf_idf_vectorizer_model, tf_nn_vectorizer_model)