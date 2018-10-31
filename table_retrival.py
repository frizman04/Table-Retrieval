#!/usr/bin/python3

import os
import json
import pickle
import argparse

import numpy as np
import tensorflow_hub as hub
import tensorflow as tf

from tqdm import tqdm

import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize

from sklearn.metrics.pairwise import cosine_similarity


def vectorize_tables(table_path,elma_path,table_vect_save_path = r'tmp/table_vect_dict.p') :
    print('Start table vectorizations')

    #load data
    with open(table_path,'r') as f:
        data = json.load(f)

    print("Load Elmo")
    elmo = hub.Module(elma_path, trainable=False)


    init = tf.global_variables_initializer()
    table_vect_dict = dict()

    for table_indx,table in tqdm(enumerate(data)) :
        print("Statrt vectorize tabel_%s " % table_indx)

        tokens_input = []
        for row_indx, row in enumerate(table['rows']) :
            for col_indx,col_name in enumerate(table['columns']) :

                cell = " ".join(row)
                title = table['title']
                headerPathes = " ".join(table['headerPath'])

                # row_name = ''
                # if table['name_entity_col'] != None :
                #     row_name = table['rows'][row_indx][table['name_entity_col']]

                cell_full = " ".join([col_name, cell, title, headerPathes]) #row_name

                tokens = word_tokenize(cell_full.lower())
                tokens_input.append(tokens)

        tokens_length = [len(tok) for tok in tokens_input]

        #заполним до максимума
        tokens_length_max = max(tokens_length)
        tmp = []
        for tok in tokens_input :
            tmp.append(tok + ["" for _ in range(tokens_length_max - len(tok))] )
        tokens_input = tmp
        
        #get embed from elma
        feed_dict = {"tokens": tokens_input,"sequence_len": tokens_length}
        embeddings = elmo(inputs=feed_dict,signature="tokens", as_dict=True)["elmo"]
        print('Calc embed : %s' % embeddings)

        #calc embeddings
        with tf.Session() as sess :
            sess.run(init)
            table_vect = embeddings.eval(session=sess)
        print('Shape of emed : %s ' % str(table_vect.shape))

        #reshape table vector
        table_vect = table_vect.mean(axis=1)
        rows_len = len(table['rows'])
        col_len = len(table['columns'])
        table_vect = table_vect.reshape((rows_len,col_len,table_vect.shape[1]))

        #save result

        table_vect_dict[table_indx] = table_vect
    pickle.dump(table_vect_dict,open(table_vect_save_path,'wb'))


def _search_cell_index(question_vect,table_vect) :
    """
        Return more relevan cell indexes and there metrix
    """ 

    cosin_sim = []
    for row_indx in range(len(table_vect)) :
        cosin_sim.append(cosine_similarity(question_vect,table_vect[row_indx])[0])
    cosin_sim = np.array(cosin_sim)

    # index = np.unravel_index(cosin_sim.argmax(), cosin_sim.shape)
    indexes = np.dstack(np.unravel_index(np.argsort(cosin_sim.ravel())[::-1], cosin_sim.shape))
    cosin_sim_sorted = np.sort(cosin_sim.ravel())[::-1]
    
    top = 10
    return indexes[:,:top,:],cosin_sim_sorted[:top]


def _vectorize_question(question,elma_path) :
    tf.reset_default_graph()

    elmo = hub.Module(elma_path, trainable=False)
    tokens_input = [word_tokenize(question.lower())]
    tokens_length = [len(tok) for tok in tokens_input]

    feed_dict = {"tokens": tokens_input,"sequence_len": tokens_length}
    embeddings = elmo(inputs=feed_dict,signature="tokens", as_dict=True)["elmo"]

    
    init = tf.global_variables_initializer()

    with tf.Session() as sess :
        sess.run(init)
        question_vect = embeddings.eval(session=sess)
    print("Question was vectorized \n ")

    question_vect = question_vect.mean(axis=1)
    return question_vect


def retrive_in_table(table_path,table_vect_path,elma_path,question,table_index) :

    with open(table_vect_path,'rb') as f:
        table_dict = pickle.load(f)

    question_vect = _vectorize_question(question,elma_path)
    cell_indxes,cosin_sim_ravel = _search_cell_index(question_vect,table_dict[table_index])

    with open(table_path,'r') as f:
        data = json.load(f)
    
    answeres = []
    for cell_indx in cell_indxes[0] :
        answer = data[table_index]['rows'][cell_indx[0]][cell_indx[1]]
        answeres.append(answer)

    return answeres,cell_indxes,cosin_sim_ravel


def task_handler(args) :
    if args.vectorize_tables : vectorize_tables(args.table_path,args.elma)
    elif args.question : 
        answeres,cell_indxes,cosin_sim = retrive_in_table(args.table_path,args.table_vect_path,args.elma,args.question,args.table_index)
        
        with open('out.csv') as f :
            for answer in answeres :
                row_csv = ",".join([args.table_index,args.question,answer])
                f.write(row_csv)    

        print("%s %s" % (answeres,cosin_sim))




def _get_elma_path() :
    if os.path.exists("model") and os.path.isdir("model") :
        if os.path.exists(r'model/oil-gas_epoches_n_13') :
            elma = r'model/oil-gas_epoches_n_13'
        else :
            elma = "http://files.deeppavlov.ai/deeppavlov_data/oil-gas_epoches_n_13.tar.gz"
    return elma


if __name__ == "__main__" :


    parser = argparse.ArgumentParser(description = "Table retriv")

    parser.add_argument("--vectorize_tables",action="store_true" ,help = "Vectorize tables to search in them")

    parser.add_argument("--table_path",help="Path to tables dataset",default=r"corrected_tables_v2.json")
    parser.add_argument("--table_vect_path",help="Path to pretrain table",default= r"tmp/table_vect_dict_row_withtitleheader.p")
    parser.add_argument("--question")
    parser.add_argument("--table_index")
    parser.add_argument("--elma", help="Path to elma", default = _get_elma_path())

    args = parser.parse_args()

    task_handler(args)