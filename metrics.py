from sklearn.metrics import ndcg_score
import numpy as np
import pandas as pd



def get_ndcg(uk_jobs, bios_test, id, dicts):
    ndcg = ndcg_score(np.asarray(
        [uk_jobs.iloc[dicts['corpus_id']]['title'].apply(lambda x: 1 if x == bios_test['raw_title'][id] else 0)]),
        [dicts['scores']], k=10)
    return ndcg


def get_gap(uk_jobs, bios_test,dicts):
    ndcg = []
    for i in range(len(bios_test)):
        ndcg.append(ndcg_score(np.asarray([uk_jobs.iloc[dicts[i]['corpus_id']]['title'].apply(lambda x: 1 if x==bios_test['raw_title'][i] else 0)]),[dicts[i]['scores']],k=10))
    

    male_ndcg = [ndcg[i] for i in bios_test[bios_test.gender =='M'].index]
    female_ndcg = [ndcg[i] for i in bios_test[bios_test.gender =='F'].index]
    
    if len(male_ndcg)>0:
        male_ndcg =sum(male_ndcg)/len(male_ndcg)
    else:
        male_ndcg = 0
    if len(female_ndcg)>0:
        female_ndcg =sum(female_ndcg)/len(female_ndcg)
    else:
        female_ndcg = 0

    return abs(np.mean(np.array(male_ndcg))-np.mean(np.array(female_ndcg)))


def get_counterfactual_gap(uk_jobs,bios_test,dicts,dicts_counter):
    ndcg = []
    ndcg_counter = []
    for i in range(len(bios_test)):
        ndcg.append(ndcg_score(np.asarray([uk_jobs.iloc[dicts[i]['corpus_id']]['title'].apply(lambda x: 1 if x==bios_test['raw_title'][i] else 0)]),[dicts[i]['scores']],k=10))
        ndcg_counter.append(ndcg_score(np.asarray([uk_jobs.iloc[dicts_counter[i]['corpus_id']]['title'].apply(lambda x: 1 if x==bios_test['raw_title'][i] else 0)]),[dicts_counter[i]['scores']],k=10))

    ndcg_separation = [abs(ndcg[i]-ndcg_counter[i]) for i in range(len(bios_test))]
        
    if len(ndcg_separation)>0:
        ndcg_separation =sum(ndcg_separation)/len(ndcg_separation)
    else:
        ndcg_separation = 0 
    return ndcg_separation

def SDR(bios_test,dicts):
    male_item_ids=[]
    female_item_ids=[]
    for male in [dicts[i] for i in bios_test[bios_test.gender =='M'].index]:
        male_df = pd.DataFrame(male)
        male_item_ids = male_item_ids+list(male_df[male_df['scores']>sorted(male_df['scores'],reverse=True)[10]]['corpus_id'])
    
    for female in [dicts[i] for i in bios_test[bios_test.gender =='F'].index]:
        female_df = pd.DataFrame(female)
        female_item_ids = female_item_ids+list(female_df[female_df['scores']>sorted(female_df['scores'],reverse=True)[10]]['corpus_id'])

    output = (len(male_item_ids)+len(female_item_ids)-2*len(set(male_item_ids).intersection(female_item_ids)))/(len(male_item_ids)+len(female_item_ids)-len(set(male_item_ids).intersection(female_item_ids)))
    #output = len(set(male_item_ids).intersection(female_item_ids))
    return output

def LDR(pth,dicts,dicts_counter):
    with open( f'{pth}share/hel/datasets/jobiqo/talent.com/JobRec/unbalanced_test.pkl', 'rb') as file:
        dicts = pickle.load(file)
    bios_test = pd.DataFrame(dicts)
    output = []
    
    for i in range(len(bios_test)):
        output.append((sum(((dicts[i]['corpus_id']!=dicts_counter[i]['corpus_id'])*1)[:10]))/10)
        #print(((dicts[i]['corpus_id']!=dicts_counter[i]['corpus_id'])*1)[:10])
        #break
           
    if len(output)>0:
        ldr =sum(output)/len(output)
    else:
        ldr = 0
     
    return ldr
