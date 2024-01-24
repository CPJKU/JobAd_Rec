import pickle
import pandas as pd

with open( '/share/hel/datasets/jobiqo/talent.com/UK_IR_subset/anonymous_new_bios.pkl', 'rb') as file:
    dicts = pickle.load(file)
bios = pd.DataFrame(dicts).reset_index()
bios = bios.drop('index', axis=1)


with open( '/share/hel/datasets/jobiqo/talent.com/UK_IR_subset/test.pkl', 'rb') as file:
    dicts = pickle.load(file)
uk_jobs_test = pd.DataFrame(dicts)

test_hits = pd.read_csv('/home/deepak/RecSys2023/Dataset/BM25/original_gender/test_hits.txt', sep=' ', header=None, names=['query_id', 'Q0', 'doc_id', 'rank', 'score','Anserini'])
test_hits = test_hits.drop(['Q0','rank','score','Anserini'], axis='columns')
test_hits['doc_id'] = test_hits['doc_id'].replace('doc','',regex=True).astype(int)

from torch.utils.data import DataLoader
from sentence_transformers import LoggingHandler, util
from sentence_transformers.cross_encoder import CrossEncoder
import torch
from tqdm.autonotebook import tqdm
import numpy as np


base_path = '/home/deepak/RecSys2023/Dataset/BM25/original_gender/results/'


model_name_list = ["CEUNK-distilroberta-base-2023-11-06_12-00-00"]


for model_name in model_name_list:

    model = CrossEncoder(base_path+model_name, num_labels=1, max_length=512, device=torch.device("cuda:6"))

    #model = CrossEncoder(base_path+model_name, num_labels=1, max_length=512, adv_dropout=0.1, device=torch.device("cuda:2"))



    result = []

    for id in tqdm(range(len(uk_jobs_test))):
        new_bios = bios.loc[test_hits[test_hits['query_id']==id]['doc_id']-1]
        query = uk_jobs_test['description'][id]
        sentence_combinations = [[query,hit] for hit in new_bios['bio']]
        cross_scores = model.predict(sentence_combinations)
        result.append({'corpus_id': new_bios.index[np.flip(np.argsort(cross_scores))],'scores':sorted(cross_scores, reverse=True)})

    import pickle

    with open('/home/deepak/RecSys2023/Dataset/BM25/original_gender/results/'+ model_name +'.pkl', 'wb') as f:
        pickle.dump(result, f)