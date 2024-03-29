import pickle
import torch
import pandas as pd
from sentence_transformers.cross_encoder import CrossEncoder
import re
from tqdm.autonotebook import tqdm
import numpy as np
#from sklearn.metrics import ndcg_score
from metrics import get_ndcg,SDR

def testing(path,
            gpu,
            pth= '/',
            masked=False,
            counterfactual=False,
            wandb_logger=None):
    prename = 'cf' if counterfactual else ''
    device = torch.device(f"cuda:{int(gpu)}")
    
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

    with open( f'{pth}share/hel/datasets/jobiqo/talent.com/JobRec/unbalanced_test.pkl', 'rb') as file:
        dicts = pickle.load(file)
    bios_test = pd.DataFrame(dicts)

    if counterfactual:
        bios_test['bio']=bios_test['counter_bio']

    test_hits = pd.read_csv(f'{pth}share/hel/datasets/jobiqo/talent.com/JobRec/BM25/unbalanced_test_hits.txt', sep=' ', header=None, names=['query_id', 'Q0', 'doc_id', 'rank', 'score','Anserini'])
    test_hits = test_hits.drop(['Q0','rank','score','Anserini'], axis='columns')
    test_hits['doc_id'] = test_hits['doc_id'].replace('doc','',regex=True).astype(int)

    model = CrossEncoder(path, num_labels=1, device=device)

    result = []
    ndcg_list = []
    male_ndcg_list = []
    female_ndcg_list = []
    neutrality_score = []
    for id in tqdm(range(len(bios_test))):
        new_uk_jobs = uk_jobs.loc[test_hits[test_hits['query_id']==id]['doc_id']-1]
        query = bios_test['bio'][id]
        sentence_combinations = [[query,hit] for hit in new_uk_jobs['description']]

        cross_scores = model.predict(sentence_combinations)
        dicts = {'corpus_id': new_uk_jobs.index[np.flip(np.argsort(cross_scores))],
                       'scores':sorted(cross_scores, reverse=True)}
        ndcg_ = get_ndcg(uk_jobs,bios_test,id, dicts)
        if bios_test.iloc[id]["gender"] == 'M':
            male_ndcg_list.append(ndcg_)
        elif bios_test.iloc[id]["gender"] == 'F':
            female_ndcg_list.append(ndcg_)
        ndcg_list.append(ndcg_)
        avg_male_ndcg  = sum(male_ndcg_list)/len(male_ndcg_list) if len(male_ndcg_list) > 0  else 0
        avg_female_ndcg = sum(female_ndcg_list) / len(female_ndcg_list) if len(female_ndcg_list) > 0  else 0
        gap = np.abs(avg_male_ndcg - avg_female_ndcg)
        neutrality = uk_jobs.iloc[dicts['corpus_id'][:10]]['neutrality'].mean()
        neutrality_score.append(neutrality)

        if wandb_logger is not None:
            wandb_logger.log({f"{prename} test_data_percentage":id/(len(bios_test)-1)*100,
                              f"{prename} test ndcg10_step":ndcg_,
                              f"{prename} test NDCG@10": sum(ndcg_list)/len(ndcg_list),
                              f"{prename} test Male_NDCG@10": avg_male_ndcg,
                              f"{prename} test Female_NDCG@10": avg_female_ndcg,
                              f"{prename} test NDCG@10 Gap": gap,
                              f"{prename} test neutrality@10": np.mean(neutrality_score)
                              },)

        result.append(dicts)
    
    sdr = SDR(bios_test,result)

    #if wandb_logger is not None:
    #    wandb_logger.log({"Final test NDCG10": sum(ndcg_list)/len(ndcg_list),
    #                    "Final test male NDCG10": avg_male_ndcg,
    #                    "Final test female NDCG10": avg_female_ndcg,
    #                    "Final test GAP": gap,
    #                    "Final test neutrality@10":sum(neutrality_score)/len(neutrality_score)})

    if wandb_logger is not None:
        wandb_logger.log({f"{prename} Final test NDCG10": sum(ndcg_list)/len(ndcg_list),
                        f"{prename} Final test male NDCG10": avg_male_ndcg,
                        f"{prename} Final test female NDCG10": avg_female_ndcg,
                        f"{prename} Final test GAP": gap,
                        f"{prename} Final test SDR": sdr,
                        f"{prename} Final test neutrality@10":sum(neutrality_score)/len(neutrality_score)})

    if masked:
        with open(path+'mask_result.pkl', 'wb') as f:
            pickle.dump(result, f)
    else:
        with open(path+f'{"shahed" if not counterfactual else "counter"}_result.pkl', 'wb') as f:
            pickle.dump(result, f)


#def get_ndcg(uk_jobs, bios_test, id, dicts):
#    ndcg = ndcg_score(np.asarray(
#        [uk_jobs.iloc[dicts['corpus_id']]['title'].apply(lambda x: 1 if x == bios_test['raw_title'][id] else 0)]),
#        [dicts['scores']], k=10)
#    return ndcg


