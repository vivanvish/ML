import pickle
import re
import string
from pathlib import Path

import pandas as pd
import spacy
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sner import Ner

jar_path = r'C:\Users\vivan\PycharmProjects\Q_Classification\lib\stanford-ner.jar'
#Get Coarse labels
def get_coarse_labels(qlist):
    labels = extract_coarse_fine_label(qlist)
    return labels['Coarse']

#Get fine labels
def get_fine_labels(qlist):
    labels = extract_coarse_fine_label(qlist)
    return labels['Fine']

#Extract Granular and fine labels from data
def extract_coarse_fine_label(qlist):
    labels = list()
    [labels.append(str.lstrip().split()[0].lower().split(sep=':')) for str in qlist]
    return pd.DataFrame(labels, columns=['Coarse','Fine'])

def get_ques(qlist):
    ques = [val.lstrip().split(' ', 1)[1] for val in qlist]
    return ques

def clean_data(qlist):
    regex = re.compile('[%s]' % re.escape(string.punctuation))
    wnl = WordNetLemmatizer()
    ques = [word_tokenize(val) for val in qlist]
    ques_cl = []
    for q in ques:
        new_ques = []
        for token in q:
            new_token = regex.sub(u'', token)
            if not new_token == u'':
                new_ques.append(wnl.lemmatize(new_token))
        ques_cl.append(new_ques)

    return ques_cl

def get_WH(qlist,train=True):
    WH_TAGS = ['WRB','WP$','WP','WDT']
    if train:
        whfile = Path('wh_tags.pkl')
        if whfile.exists():
            with open('wh_tags.pkl', 'rb') as fp:
                whlist = pickle.load(fp)
        else:
            whlist = [' '.join([ tags[0] for tags in pos_tag(s) if tags[1] in WH_TAGS]).lstrip() for s in qlist]
            whlist = ['Other' if not w.lstrip() else w for w in whlist]
            with open('wh_tags.pkl', 'wb') as fp:
                pickle.dump(whlist, fp)
    else:
        whlist = [' '.join([tags[0] for tags in pos_tag(s) if tags[1] in WH_TAGS]).lstrip() for s in qlist]
        whlist = ['Other' if not w.lstrip() else w for w in whlist]
    return whlist

def get_NER(qlist,train=True):
    if train:
        nerfile = Path('ner_tags.pkl')
        if nerfile.exists():
            with open('ner_tags.pkl','rb') as fp:
                nerlist = pickle.load(fp)
        else:
            st = Ner(host='localhost', port=9199)
            nerlist = [' '.join([tags[1] for tags in st.get_entities(' '.join(s).lstrip())]).lstrip() for s in qlist]
            with open('ner_tags.pkl','wb') as fp:
                pickle.dump(nerlist,fp)
    else:
        st = Ner(host='localhost', port=9199)
        nerlist = [' '.join([tags[1] for tags in st.get_entities(' '.join(s).lstrip())]).lstrip() for s in qlist]
    return nerlist

def generate_POS_tag(qlist,train=True):
    if train:
        posfile = Path('pos_tags.pkl')
        if posfile.exists():
            with open('pos_tags.pkl','rb') as fp:
                poslist = pickle.load(fp)
        else:
            poslist = [' '.join([tags[1] for tags in pos_tag(s)]).lstrip() for s in qlist]
            with open('pos_tags.pkl','wb') as fp:
                pickle.dump(poslist,fp)
    else:
        poslist = [' '.join([tags[1] for tags in pos_tag(s)]).lstrip() for s in qlist]
    return poslist

def generate_chunks(qlist):
    chunks= list()
    proc = spacy.load('en')
    for s in qlist:
        s = ' '.join(s).lstrip()
        sent = proc(u''+s)
        chunks.append(' '.join([s.dep_ for s in sent]))
    return chunks

if __name__ == '__main__':
    str = ''
    with open('data/train.txt', 'r') as f:
        data = f.read().splitlines()
    #targets = get_coarse_labels(data)
    questions = get_ques(data)
    questions = clean_data(questions)
    # chunks=generate_chunks(questions)

    print(questions)
