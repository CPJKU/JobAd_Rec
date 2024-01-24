import pickle
import pandas as pd
import argparse

from sentence_transformers.cross_encoder import CrossEncoder
import torch
from tqdm.autonotebook import tqdm
import numpy as np
import re

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_id", type=int, default=0, help="gpu no.")
    parser.add_argument("--seed", type=int, default=0, help="torch random seed")
    parser.add_argument("--model", type=str, default='BERT', help="model BERT or")
    parser.add_argument("--mask", type=int, default=1, help="number of words to be masked")
    parser.add_argument("--candidate", type=int, default=1, help="number of candidates to be used for masked")
    base_args, _ = parser.parse_known_args()

    n = base_args.candidate
    m = base_args.mask
    torch.manual_seed(base_args.seed)
    print(f"torch.manual_seed({base_args.seed})")
    print(m)
    print(n)
    device = torch.device(f"cuda:{int(base_args.gpu_id)}")

    with open( '/home/deepak/RecSys2023/Dataset/new_bios.pkl', 'rb') as file:
        dicts = pickle.load(file)
    bios = pd.DataFrame(dicts).reset_index()
    bios = bios.drop('index', axis=1)


    with open( '/home/deepak/RecSys2023/Dataset/test.pkl', 'rb') as file:
        dicts = pickle.load(file)
    uk_jobs_test = pd.DataFrame(dicts)

    test_hits = pd.read_csv('/home/deepak/RecSys2023/Dataset/BM25/original_gender/test_hits.txt', sep=' ', header=None, names=['query_id', 'Q0', 'doc_id', 'rank', 'score','Anserini'])
    test_hits = test_hits.drop(['Q0','rank','score','Anserini'], axis='columns')
    test_hits['doc_id'] = test_hits['doc_id'].replace('doc','',regex=True).astype(int)



    base_path = '/home/deepak/RecSys2023/Dataset/BM25/original_gender/results/'
    dictionary_words = {}


    if base_args.model == 'BERT':
        model_name = 'cross-encoder-bert-base-uncased-2023-05-06_14-22-10-latest'
        for job in uk_jobs_test['title'].unique():
            with open( '/home/deepak/RecSys2023/shap_bert/'+job.replace(' ','_')+'.pkl', 'rb') as file:
                dictionary_words.update(pickle.load(file))
    elif base_args.model == 'roberta':
        model_name = 'cross-encoder-distilroberta-base-2023-05-06_14-30-40-latest'
        for job in uk_jobs_test['title'].unique():
            with open( '/home/deepak/RecSys2023/shap/'+job.replace(' ','_')+'.pkl', 'rb') as file:
                dictionary_words.update(pickle.load(file))

    model = CrossEncoder(base_path+model_name, num_labels=1, max_length=512, device=device)


    result = []

    for idx in tqdm(range(len(uk_jobs_test))):
        new_bios = bios.loc[test_hits[test_hits['query_id']==idx]['doc_id']]
        #list_dictionary_words = []
        list_dictionary_words = dictionary_words[(idx,n-1)]
        #for i in range(n):
        #    list_dictionary_words = list_dictionary_words + dictionary_words[(idx,i)]
        replace_words = re.compile("|".join(map(re.escape,[i for i in list_dictionary_words[0][m*(-1):] if len(i)>2])), re.IGNORECASE)
        query = replace_words.sub('[MASK]',uk_jobs_test['description'][idx])
        sentence_combinations = [[query,hit] for hit in new_bios['bio']]
        cross_scores = model.predict(sentence_combinations)
        result.append({'corpus_id': new_bios.index[np.flip(np.argsort(cross_scores))],'scores':sorted(cross_scores, reverse=True)})

    with open(base_path+model_name+'/test_mask_'+ str(n)+'_'+str(m)+'.pkl', 'wb') as f:
        pickle.dump(result, f)

if __name__ == "__main__":

    main()