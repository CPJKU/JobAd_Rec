#Load Data
import argparse

import pickle
import pandas as pd

from torch.utils.data import DataLoader
from sentence_transformers import LoggingHandler
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CERerankingEvaluator

import logging
from datetime import datetime
import torch
import regex as re
from tqdm.autonotebook import tqdm

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_id", type=int, default=0, help="gpu no.")
    parser.add_argument("--seed", type=int, default=0, help="torch random seed")
    parser.add_argument("--model", type=str, default="bert--uncased", help="model name")
    parser.add_argument("--balance", type=str, default="original_gender", help="model name")
    parser.add_argument("--anonymous", default=False, action='store_true', help="remove gender from text input")
    base_args, _ = parser.parse_known_args()

    model_name = base_args.model
    balance = base_args.balance
    if base_args.anonymous:
        condition = "UNK"
    else:
        condition = "KNO"  
    #train_batch_size = 64
    num_epochs = 4
    model_save_path = '/home/deepak/RecSys2023/Dataset/BM25/'+balance+'/results/CE'+condition+'-'+model_name+'-'+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    #pos_neg_ration = 4
    torch.manual_seed(base_args.seed)
    print(f"torch.manual_seed({base_args.seed})")

    device = torch.device(f"cuda:{int(base_args.gpu_id)}")

    if balance =='original_gender':
        with open( '/share/hel/datasets/jobiqo/talent.com/UK_IR_subset/anonymous_new_bios.pkl', 'rb') as file:
            dicts = pickle.load(file)
    elif balance == 'female':
        with open( '/home/deepak/RecSys2023/Dataset/new_bios_female.pkl', 'rb') as file:
            dicts = pickle.load(file)
    elif balance == 'male':
        with open( '/home/deepak/RecSys2023/Dataset/new_bios_male.pkl', 'rb') as file:
            dicts = pickle.load(file)
    else:
        print('No candidate dataset')
    bios = pd.DataFrame(dicts).reset_index()
    bios = bios.drop('index', axis=1)

    gender_words = ['mr','his','he','him','himself','mrs','hers','she','her','herself','Alice', 'Bob']
    if base_args.anonymous:
        anonymous_bios = []
        for i in range(len(bios['bio'])):
            bio_string = bios['bio'][i]
            for i in range(len(gender_words)):
                bio_string = re.sub(gender_words[i], '_',bio_string,flags=re.I)
            anonymous_bios.append(bio_string)

        bios['bio'] == anonymous_bios



    with open( '/share/hel/datasets/jobiqo/talent.com/UK_IR_subset/train.pkl', 'rb') as file:
        dicts = pickle.load(file)
    uk_jobs_train = pd.DataFrame(dicts)

    with open( '/share/hel/datasets/jobiqo/talent.com/UK_IR_subset/val.pkl', 'rb') as file:
        dicts = pickle.load(file)
    uk_jobs_val = pd.DataFrame(dicts)

    val_hits = pd.read_csv('/home/deepak/RecSys2023/Dataset/BM25/'+balance+'/val_hits.txt', sep=' ', header=None, names=['query_id', 'Q0', 'doc_id', 'rank', 'score','Anserini'])
    train_hits = pd.read_csv('/home/deepak/RecSys2023/Dataset/BM25/'+balance+'/train_hits.txt', sep=' ', header=None, names=['query_id', 'Q0', 'doc_id', 'rank', 'score','Anserini'])
    val_hits = val_hits.drop(['Q0','rank','score','Anserini'], axis='columns')
    val_hits['doc_id'] = val_hits['doc_id'].replace('doc','',regex=True).astype(int)
    train_hits = train_hits.drop(['Q0','rank','score','Anserini'], axis='columns')
    train_hits['doc_id'] = train_hits['doc_id'].replace('doc','',regex=True).astype(int)

    model = CrossEncoder(model_name, num_labels=1, max_length=512, device=device)

    #In Train and Test for every datapoint with label 1 we add 4 negative samples
    import random
    from sentence_transformers import InputExample
    from tqdm.autonotebook import tqdm

    train_samples = []

    for id in tqdm(range(len(uk_jobs_train))):

        new_bios = bios.loc[train_hits[train_hits['query_id']==id]['doc_id']-1]

        query = uk_jobs_train['description'][id]
        job = uk_jobs_train['title'][id]
        
        pos_passage = random.choice(list(new_bios.loc[(bios.raw_title == job)&(bios.gender == 'M')]['bio']))
        train_samples.append(InputExample(texts=[query, pos_passage], label=1))
        pos_passage = random.choice(list(new_bios.loc[(bios.raw_title == job)&(bios.gender == 'F')]['bio']))
        train_samples.append(InputExample(texts=[query, pos_passage], label=1))
        
        for looper in range(4):
            neg_passage = random.choice(list(new_bios.loc[bios.raw_title != job]['bio']))
            train_samples.append(InputExample(texts=[query, neg_passage],label=0))


    dev_samples = {}

    for id in tqdm(range(len(uk_jobs_val))):

        new_bios = bios.loc[val_hits[val_hits['query_id']==id]['doc_id']-1]

        query = uk_jobs_val['description'][id]
        job = uk_jobs_val['title'][id]

        dev_samples[id] = {'query': query, 'positive': set(), 'negative': set()}

        pos_passage = random.choice(list(new_bios.loc[bios.raw_title == job]['bio']))

        neg_passage = random.choice(list(new_bios.loc[bios.raw_title != job]['bio']))

        dev_samples[id]['positive'].add(pos_passage)
        dev_samples[id]['negative'].add(neg_passage)


    train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=16)

    # It performs a classification task and measures scores like F1 (finding relevant passages) and Average Precision
    evaluator = CERerankingEvaluator(dev_samples, name='train-eval')

    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO,
                        handlers=[LoggingHandler()])


    warmup_steps = 500
    logging.info("Warmup-steps: {}".format(warmup_steps))


    # Train the model
    model.fit(train_dataloader=train_dataloader,
            evaluator=evaluator,
            epochs=num_epochs,
            evaluation_steps=5904,
            warmup_steps=warmup_steps,
            output_path=model_save_path,
            save_best_model= True,
            debias=False,
            use_amp=True)

    #Save latest model
    #model.save(model_save_path+'-latest')

if __name__ == "__main__":

    main()
