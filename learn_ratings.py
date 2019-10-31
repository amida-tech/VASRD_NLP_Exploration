'''
Created on Oct 25, 2019

@author: gregory
'''

from joblib import dump, load
from keras.callbacks.callbacks import EarlyStopping
from keras.models import load_model, Sequential
from keras.layers import Dense, LSTM
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
import numpy as np
from os import listdir
from os.path import isfile, join
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.svm import LinearSVC

ratings = []
features = []
vectorizer = CountVectorizer()
t = Tokenizer()

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

def featurize():
    # bag of words feature
    descriptions = [rating[1] for rating in ratings]
    X = vectorizer.fit_transform(descriptions)
    dump(vectorizer, 'model/cv')
    features.append(X)

def featurize_keras():
    # bag of words feature
    descriptions = [rating[1] for rating in ratings]
    # fit the tokenizer on the documents
    t.fit_on_texts(descriptions)
    # integer encode documents
    encoded_docs = t.texts_to_matrix(descriptions, mode='count')
    features.append(encoded_docs)

def train_multioutput_svm(model_output):

    bag_of_words = features[0]
    # more complicated when we have more than 1 feature
    x = bag_of_words
    
    # we wish 2 predict 2 things, diagnostic code and minor code
    y = np.array([[rating[0], rating[3]] for rating in ratings])

    svm_classifier = LinearSVC(random_state=0, tol=1e-5, max_iter=100000)
    classifier = MultiOutputClassifier(svm_classifier, n_jobs=-1)
    classifier.fit(x, y)
    
    dump(classifier, model_output) 

def train_multioutput_lstm(model_output):    
    bag_of_words = features[0]
    # more complicated when we have more than 1 feature
    X_train = bag_of_words
    
    X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
       
    # we wish 2 predict 2 things, diagnostic code and minor code
    #one hot encode both columns then combine
    encoded_1st = to_categorical(np.array([[rating[0] for rating in ratings]]))[0]
    encoded_2nd = to_categorical(np.array([[rating[len(rating) - 1] for rating in ratings]]))[0]
    Y_train = np.concatenate((encoded_1st,encoded_2nd), axis=1)
    
    model = Sequential()
    # should not hard code this one probably, the 17347
    model.add(LSTM(17347, dropout=0.2, recurrent_dropout=0.2))
    #model.add(Dense(2, activation='softmax'))
    #model.add(TimeDistributed(Dense(3))) 
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    epochs = 5
    batch_size = 64

    model.fit(x=X_train, y=Y_train, epochs=epochs, batch_size=batch_size,validation_split=0.1,callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])

    model.save(model_output)

def evaluate_term(term, model_output):
    # reload all applicable models
    vectorizer = load('model/cv')
    classifier = load(model_output) 
    
    test_data_x = vectorizer.transform([term])
    
    print(classifier.predict(test_data_x))

def evaluate_lstm_term(term, model_output):
    # reload all applicable models
    model = load_model(model_output)
    
    # fit the tokenizer on the term
    t.fit_on_texts([term])
    # integer encode documents
    test_data_x = t.texts_to_matrix([term], mode='count')
    test_data_x = np.reshape(test_data_x, (test_data_x.shape[0], 1, test_data_x.shape[1]))
    
    print(model.predict(test_data_x))

def evaluate_folder(test_dir, model_output):
    test_files = []
    onlyfiles = [f for f in listdir(test_dir) if isfile(join(test_dir, f))]
    for onlyfile in onlyfiles:
        with open(join(test_dir, onlyfile), 'r', encoding='utf-8') as fs:
            test_files.append(' '.join(fs.readlines()))

    # reload all applicable models
    vectorizer = load('model/cv')
    classifier = load(model_output) 
            
    test_data_x = vectorizer.transform(test_files)
    print(classifier.predict(test_data_x))            

def evaluate_lstm_folder(test_dir, model_output):
    test_files = []
    onlyfiles = [f for f in listdir(test_dir) if isfile(join(test_dir, f))]
    for onlyfile in onlyfiles:
        with open(join(test_dir, onlyfile), 'r', encoding='utf-8') as fs:
            test_files.append(' '.join(fs.readlines()))
    
    # reload all applicable models
    model = load_model(model_output)
    
    # integer encode documents
    test_data_x = t.texts_to_matrix(test_files, mode='count')
    test_data_x = np.reshape(test_data_x, (test_data_x.shape[0], 1, test_data_x.shape[1]))
    
    print(model.predict(test_data_x))

def main(ratings_file, svm_model, lstm_model, test_folder):
    read_ratings_information(ratings_file)
    #featurize()
    featurize_keras()
    #train_multioutput_svm(svm_model)
    #train_multioutput_lstm(lstm_model)
    #evaluate_term('', svm_model)
    #evaluate_folder(test_folder, svm_model)
    #evaluate_lstm_term('heart attack', lstm_model)
    evaluate_lstm_folder(test_folder, lstm_model)

if __name__ == '__main__':
    ratings_file = 'data/all_ratings_good.txt'
    svm_model = 'model/svm_model'
    lstm_model = 'model/lstm_model'
    test_folder = '/Users/gregory/eclipse-workspace/CliNER_Experiments/data/sample_texts'
    main(ratings_file, svm_model, lstm_model, test_folder)