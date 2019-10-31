'''
Created on Oct 9, 2019

@author: gregory
'''

from gensim.summarization.summarizer import summarize
import heapq
import nltk
from os import listdir
from os.path import isfile, join
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
from tika import parser

text_prefix = '§4.'
sections_prefix = '§§'

nltk.download('stopwords')
nltk.download('punkt')

nlp = spacy.load("en_core_web_sm")

vect = TfidfVectorizer(min_df=1, stop_words="english") 

def gensim_summary(text, output_directory, output_file):
    try:
        with open(join(output_directory, output_file), 'w', encoding='utf-8') as fs:
            fs.write(summarize(text))
    except:
        pass

def nltk_summary(text, output_directory, output_file):
    stopwords = nltk.corpus.stopwords.words('english')    
    
    word_frequencies = {}
    for word in nltk.word_tokenize(text):
        if word not in stopwords:
            if word not in word_frequencies.keys():
                word_frequencies[word] = 1
            else:
                word_frequencies[word] += 1

    try:
        maximum_frequncy = max(word_frequencies.values())

        for word in word_frequencies.keys():
            word_frequencies[word] = (word_frequencies[word] / maximum_frequncy)
    except:
        pass

    sentence_list = nltk.sent_tokenize(text)

    sentence_scores = {}
    for sent in sentence_list:
        for word in nltk.word_tokenize(sent.lower()):
            if word in word_frequencies.keys():
                if len(sent.split(' ')) < 30:
                    if sent not in sentence_scores.keys():
                        sentence_scores[sent] = word_frequencies[word]
                    else:
                        sentence_scores[sent] += word_frequencies[word]

    summary_sentences = heapq.nlargest(7, sentence_scores, key=sentence_scores.get)

    summary = ' '.join(summary_sentences)
    with open(join(output_directory, output_file), 'w', encoding='utf-8') as fs:
        fs.write(summary)

def sklearn_similarity(text1, text2, file1, file2, output_directory, output_file):
    # could do a single similarity process at once for all documents if we wanted
    corpus = [text1, text2]
    try:
        tfidf = vect.fit_transform(corpus)
        pairwise_similarity = tfidf * tfidf.T
        with open(join(output_directory, output_file), 'a') as fs:
            fs.write(file1 + '\t' + file2 + '\t' + str(pairwise_similarity[0,1]) + '\n')
    except:
        with open(join(output_directory, output_file), 'a') as fs:
            fs.write(file1 + '\t' + file2 + '\t' + str(-1.0) + '\n')

def spacy_ner(text, output_directory, output_file):
    doc = nlp(text)
    with open(join(output_directory, output_file), 'w', encoding='utf-8') as fs:
        for ent in doc.ents:
            fs.write(ent.text + '\t' + ent.label_ + '\n')

def process_sections(sections_dir, operation, output_directory):
    onlyfiles = [f for f in listdir(sections_dir) if isfile(join(sections_dir, f))]
    
    # slightly different path for this one since we are comparing documents
    if 'sklearn_similarity' == operation:
        for i in range(0, len(onlyfiles)):
            for j in range(i + 1, len(onlyfiles)):
                file1 = onlyfiles[i]
                file2 = onlyfiles[j]
                with open(join(sections_dir, file1), 'r', encoding='utf-8') as f:
                    text1 = ''.join(f.readlines())
                with open(join(sections_dir, file2), 'r', encoding='utf-8') as f:
                    text2 = ''.join(f.readlines())
                sklearn_similarity(text1, text2, file1, file2, output_directory, 'similarity_matrix.txt')
        return
        
    for onlyfile in onlyfiles:
        with open(join(sections_dir, onlyfile), 'r', encoding='utf-8') as f:
            lines = f.readlines()
            if 'gensim_summary' == operation:
                gensim_summary(''.join(lines), output_directory, onlyfile)
            elif 'nltk_summary' == operation:
                nltk_summary(''.join(lines), output_directory, onlyfile)
            elif 'spacy_ner' == operation:
                spacy_ner(''.join(lines), output_directory, onlyfile)

def process_section_for_rating(section_text):    
    ratings_list = []
    body = []
    for line in section_text:
        if len(body) == 0:
            m_starts = re.search(r'^\d{4}', line)
            if m_starts is not None:
                body.append(line.strip())
                m_ends = re.search(r'\d+$', line)
                if m_ends is not None:
                    ratings_list.append(' '.join(body))
                    body[0:1]
        else:
            m_starts = re.search(r'^\d{4}', line)
            if m_starts is not None:
                body = []
            body.append(line.strip())            
            m_ends = re.search(r'\d+$', line)
            if m_ends is not None:
                ratings_list.append(' '.join(body))
                body = body[0:1]
            
    return ratings_list

def extract_ratings(schedule_of_ratings_dir, rating_dir):
    onlyfiles = [f for f in listdir(schedule_of_ratings_dir) if isfile(join(schedule_of_ratings_dir, f))]
    for onlyfile in onlyfiles:
        with open(join(schedule_of_ratings_dir, onlyfile)) as fs:
            lines = fs.readlines()
        print(onlyfile)
        ratings = process_section_for_rating(lines)
        with open(join(rating_dir, onlyfile), 'w', encoding='utf-8') as fs:
            for rating in ratings:
                fs.write(str(rating) + '\n')

def combine_ratings(ratings_dir, single_ratings_file):
    global_list = []
    
    onlyfiles = [f for f in listdir(ratings_dir) if isfile(join(ratings_dir, f))]
    for onlyfile in onlyfiles:
        with open(join(ratings_dir, onlyfile), 'r', encoding='utf-8') as fs:        
            lines = fs.readlines()
        for line in lines:
            global_list.append(line.strip())
    global_list.sort()
    
    with open(single_ratings_file, 'w', encoding='utf-8') as fs:
        for item in global_list:
            fs.write(item + '\n')

def process_text_file_into_sections(input_file, sections_dir, schedule_of_ratings_dir):    
    with open(input_file, 'r', encoding = 'utf-8') as f:
        lines = f.readlines()

    header_prefix = 0
    state = -1
    header = ''
    body = ''
    is_schedule_of_ratings = False
    
    for line in lines:
        if line.startswith(sections_prefix):
            line = line[1:]
        if line.startswith(text_prefix):
            header_prefix += 1
            if state != -1:
                with open(join(sections_dir, header) + '.txt', 'w', encoding='utf-8') as fs:
                    fs.write(body)
                if is_schedule_of_ratings:
                    with open(join(schedule_of_ratings_dir, header) + '.txt', 'w', encoding='utf-8') as fs:
                        fs.write(body)  
                body = ''  
            header = line.split(' ')[0]
            is_schedule_of_ratings = 'Schedule of ratings' in line
            state = 1
        else:
            state = 0
            stripped = line.strip(' \n\r')
            if len(stripped) > 0:
                body = body + line

    # last time through
    with open(join(sections_dir, header) + '.txt', 'w', encoding='utf-8') as fs:
        fs.write(body)
    body = ''  

        
def process_pdf_file(input_file, output_file):
    raw = parser.from_file(input_file)
    with open(debug_file, 'w', encoding='utf-8') as f:
        f.write(raw['content'])    

def main(input_file, sections_dir, schedule_of_ratings_dir, ratings_dir, single_ratings_file, debug_file, mode, output_directory):
    #process_pdf_file(input_file, debug_file)
    #process_text_file_into_sections(input_file, sections_dir, schedule_of_ratings_dir)
    #process_sections(sections_dir, mode, output_directory)
    #extract_ratings(schedule_of_ratings_dir, ratings_dir)
    combine_ratings(ratings_dir, single_ratings_file)

if __name__ == '__main__':
    input_file = 'data/processed_38_CFR_4.txt'
    sections_dir = 'data/sections'
    schedule_of_ratings_dir = 'data/schedule_of_ratings'
    ratings_dir = 'data/ratings'
    single_ratings_file = 'data/all_ratings.txt'
    debug_file = 'processed_pdf.txt'
    mode = 'sklearn_similarity'
    output_directory = 'output/sklearn_similarity'
    main(input_file, sections_dir, schedule_of_ratings_dir, ratings_dir, single_ratings_file, debug_file, mode, output_directory)