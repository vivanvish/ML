import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import CountVectorizer
import data_prep as prep

from textblob.classifiers import NaiveBayesClassifier as NBC
import pickle
from sklearn.svm import LinearSVC

vect_wh = CountVectorizer(min_df=1)
vect_chunk = CountVectorizer(min_df=1)
vect_ner = CountVectorizer(min_df=1)
vect_pos = CountVectorizer(min_df=1)
root_pos = list()

def vectorize_wh(qlist,train=True):
    x_wh= prep.get_WH(qlist,train)
    if train:
        x_wh = vect_wh.fit_transform(x_wh)
    else:
        x_wh = vect_wh.transform(x_wh)
    return x_wh

def vectorize_ner(qlist,train=True):
    x_ner= prep.get_NER(qlist,train)
    if train:
        x_ner = vect_ner.fit_transform(x_ner)
    else:
        x_ner = vect_ner.transform(x_ner)
    return x_ner

def vectorize_pos(qlist,train=True):
    x_pos= prep.generate_POS_tag(qlist,train)
    if train:
         x_pos =  vect_pos.fit_transform(x_pos)
    else:
        x_pos = vect_pos.transform(x_pos)
    return x_pos

def vectorize_chunk(qlist, train=True):
    x_chunk= prep.generate_chunks(qlist)
    if train:
        x_chunk = vect_chunk.fit_transform(x_chunk)
    else:
        x_chunk = vect_chunk.transform(x_chunk)
    return x_chunk


def process_test(qtest):
    test_cl = prep.clean_data(qtest)
    xt_wh = vectorize_wh(test_cl,train=False)
    xt_ner = vectorize_ner(test_cl,train=False)
    xt_pos = vectorize_pos(test_cl,train=False)
    xt_chunk = vectorize_chunk(test_cl,train=False)
    xtest = hstack([xt_wh, xt_ner, xt_pos, xt_chunk])
    return xtest

if __name__ == '__main__':
    with open('data/train.txt', 'r') as f:
        data = f.read().splitlines()
    questionlist = prep.clean_data(prep.get_ques(data))
    questionlist = [' '.join(s) for s in questionlist]
    # X_wh = vectorize_wh(questionlist)
    # X_ner = vectorize_ner(questionlist)
    # X_pos = vectorize_pos(questionlist)
    # X_chunk = vectorize_chunk(questionlist)
    #
    # Xtrain = hstack([X_wh,X_ner,X_pos,X_chunk])
    ytrain = prep.get_coarse_labels(data)

    train_corpus = list(zip(questionlist,ytrain.ravel()))

    model = NBC(train_corpus)

    # clf = LinearSVC()
    # clf.fit(Xtrain,ytrain.ravel())
    #
    with open('data/test.txt', 'r') as f:
        test = f.read().splitlines()
    X_test = prep.clean_data(prep.get_ques(test))
    X_test = [' '.join(s) for s in X_test]
    ytest = prep.get_coarse_labels(test)

    test_corpus = list(zip(X_test,ytest))
    print(model.accuracy(test_corpus))
    # ypred = clf.predict(X_test)
    # with open('coarse_pred.pkl','wb') as fp:
    #     pickle.dump(ypred,fp)
    # score = clf.score(X_test,ytest.ravel())
    # print(score)
