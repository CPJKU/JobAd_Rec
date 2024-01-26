import pickle
import pandas as pd
from sentence_transformers.cross_encoder import CrossEncoder
import re
from tqdm.autonotebook import tqdm
import numpy as np

def testing(path,device,pth, masked=False):

    with open(f'{pth}share/hel/datasets/jobiqo/talent.com/JobRec/uk_jobs.pkl', 'rb') as file:
        dicts = pickle.load(file)
    uk_jobs = pd.DataFrame(dicts).reset_index()
    uk_jobs = uk_jobs.drop('index', axis=1)

    if masked:
        with open(f'{pth}share/hel/datasets/jobiqo/talent.com/JobRec/dictionary_jobs.pkl', 'rb') as file:
            dictionary_jobs = pickle.load(file)
        list_dictionary_words = list(set([k for v in dictionary_jobs.values() for k in v.keys()]))
        replace_words = re.compile("|".join(map(re.escape,list_dictionary_words)), re.IGNORECASE)
        if 'roberta' in path:
            mask_token='<mask>'
        else:
            mask_token='[MASK]'
        uk_jobs['description'] = uk_jobs['description'].apply(lambda x:  replace_words.sub(mask_token,x))

    with open( f'{pth}share/hel/datasets/jobiqo/talent.com/JobRec/test.pkl', 'rb') as file:
        dicts = pickle.load(file)
    bios_test = pd.DataFrame(dicts)

    test_hits = pd.read_csv(f'{pth}share/hel/datasets/jobiqo/talent.com/JobRec/BM25/test_hits.txt', sep=' ', header=None, names=['query_id', 'Q0', 'doc_id', 'rank', 'score','Anserini'])
    test_hits = test_hits.drop(['Q0','rank','score','Anserini'], axis='columns')
    test_hits['doc_id'] = test_hits['doc_id'].replace('doc','',regex=True).astype(int)

    model = CrossEncoder(path, num_labels=1, device=device)

    result = []

    for id in tqdm(range(len(bios_test))):
        new_uk_jobs = uk_jobs.loc[test_hits[test_hits['query_id']==id]['doc_id']-1]
        query = bios_test['bio'][id]
        sentence_combinations = [[query,hit] for hit in new_uk_jobs['description']]
        cross_scores = model.predict(sentence_combinations)
        result.append({'corpus_id': new_uk_jobs.index[np.flip(np.argsort(cross_scores))],'scores':sorted(cross_scores, reverse=True)})

    if masked:
        with open(path+'mask_result.pkl', 'wb') as f:
            pickle.dump(result, f)    
    else:
        with open(path+'shahed_result.pkl', 'wb') as f:
            pickle.dump(result, f)