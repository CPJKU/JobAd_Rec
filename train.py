#Load Data
import argparse

import pickle
import pandas as pd

from torch.utils.data import DataLoader
from sentence_transformers import LoggingHandler
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CERerankingEvaluator
from sentence_transformers import InputExample
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
    #parser.add_argument("--balance", type=str, default="balanced", help="the gender distribution in training data is skewed towards")
    #parser.add_argument("--anonymous", default=False, action='store_true', help="remove gender from candidate text")
    base_args, _ = parser.parse_known_args()

    model_name = base_args.model
    #balance = base_args.balance
    #if base_args.anonymous:
    #    condition = "UNK"
    #else:
    #    condition = "KNO"  
    #train_batch_size = 64
    num_epochs = 4
    model_save_path = '/home/deepak/JobAd_Rec/Models/'+model_name+'_'+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    #pos_neg_ration = 4
    print(f"manual_seed({base_args.seed})")
    torch.manual_seed(base_args.seed)

    device = torch.device(f"cuda:{int(base_args.gpu_id)}")
    torch.cuda.manual_seed(base_args.seed)
    torch.backends.cudnn.deterministic = True


    try:
        with open( '/share/hel/datasets/jobiqo/talent.com/JobRec/uk_jobs.pkl', 'rb') as file:
            dicts = pickle.load(file)
    except:
        print('No Job Ad dataset')
    uk_jobs = pd.DataFrame(dicts).reset_index()
    uk_jobs = uk_jobs.drop('index', axis=1)


    with open( '/share/hel/datasets/jobiqo/talent.com/JobRec/train.pkl', 'rb') as file:
        train_samples = pickle.load(file)

    with open( '/share/hel/datasets/jobiqo/talent.com/JobRec/val.pkl', 'rb') as file:
        dev_samples = pickle.load(file)

    #gender_words = ['mr','his','he','him','himself','mrs','hers','she','her','herself','Alice', 'Bob']
    #if base_args.anonymous:
    #    anonymous_bios = []
    #    for i in range(len(bios_train['bio'])):
    #        bio_string = bios_train['bio'][i]
    #        for i in range(len(gender_words)):
    #            bio_string = re.sub(gender_words[i], '_',bio_string,flags=re.I)
    #        anonymous_bios.append(bio_string)

    #    bios_train['bio'] == anonymous_bios
    #    anonymous_bios = []
    #    for i in range(len(bios_val['bio'])):
    #       bio_string = bios_val['bio'][i]
    #        for i in range(len(gender_words)):
    #            bio_string = re.sub(gender_words[i], '_',bio_string,flags=re.I)
    #        anonymous_bios.append(bio_string)

    #    bios_val['bio'] == anonymous_bios

    model = CrossEncoder(model_name, num_labels=1, device=device)

    train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=16)

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
            warmup_steps=warmup_steps,
            output_path=model_save_path,
            save_best_model= True,
            debias=False,
            use_amp=True)

    #Save latest model
    #model.save(model_save_path+'-latest')

if __name__ == "__main__":

    main()
