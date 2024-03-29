#Load Data
import argparse

import pickle
# import pandas as pd

from torch.utils.data import DataLoader
from sentence_transformers import LoggingHandler
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CERerankingEvalUpdated
# from sentence_transformers import InputExample
import logging
from datetime import datetime
import torch
# import regex as re
# from tqdm.autonotebook import tqdm
from testing import testing
from metrics import get_counterfactual_gap,LDR

import wandb
import os


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_id", type=int, default=0, help="gpu no.")
    parser.add_argument("--seed", type=int, default=0, help="torch random seed")
    parser.add_argument("--model", type=str, default='distilroberta-base', help="model name")
    parser.add_argument("--local", default=False, action='store_true', help="changes path suitable to run locally")

    parser.add_argument("--num_epochs", type=int, default=15, help="training Epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="batch size")
    parser.add_argument("--warmup_steps", type=int, default=500, help="batch size")

    parser.add_argument("--debias", type=str, default=None, help="select from debias methods "
                                                                 "'reg' for regularization  "
                                                                 "'adv' for adversarial")
    parser.add_argument("--lmbda", type=float, default=0., help="Regularization Strength")

    parser.add_argument("--wandb", default=False, action='store_true', help="Logs on Wandb")
    parser.add_argument("--project_name", type=str, default="unbalanced_sigmoid_JAR", help="Wandb project name")
    parser.add_argument("--exp_name", type=str, default=None, help="Experiment Name")
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
    pth = "/home/shahed/" if base_args.local==True else "/"
    model_str = str(base_args.seed)+'_'+str(base_args.debias)+'_'+str(base_args.lmbda)+'_'+model_name+'_'+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_save_path = f'./Models/'+model_str

    config = vars(base_args)
    config["model_save_path"] = model_save_path
    config["train_path"] = f'{pth}share/hel/datasets/jobiqo/talent.com/JobRec/unbalanced_train_samples.pkl'
    config["dev_path"] = f'{pth}share/hel/datasets/jobiqo/talent.com/JobRec/unbalanced_dev_samples.pkl'

    #pos_neg_ration = 4
    print(f"manual_seed({base_args.seed})")
    torch.manual_seed(base_args.seed)

    device = torch.device(f"cuda:{int(base_args.gpu_id)}")
    torch.cuda.manual_seed(base_args.seed)
    torch.backends.cudnn.deterministic = True


    with open( f'{pth}share/hel/datasets/jobiqo/talent.com/JobRec/unbalanced_train_samples.pkl', 'rb') as file:
        train_samples = pickle.load(file)

    with open( f'{pth}share/hel/datasets/jobiqo/talent.com/JobRec/unbalanced_dev_samples.pkl', 'rb') as file:
        dev_samples = pickle.load(file)


    model = CrossEncoder(model_name, num_labels=1, device=device)
    config["model_config"] = model.config

    if base_args.wandb:
        if not os.path.exists(model_save_path):
            os.mkdir(model_save_path)
        wandb_logger = wandb.init(dir=model_save_path,
                                  project=base_args.project_name,
                                  name=f"{base_args.exp_name if base_args.exp_name is not None else model_str}",
                                  config=config)
    else:
        wandb_logger = None
    train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=base_args.batch_size)

    evaluator = CERerankingEvalUpdated(dev_samples, name='train-eval', lmbda=base_args.lmbda)

    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO,
                        handlers=[LoggingHandler()])


    warmup_steps = base_args.warmup_steps
    logging.info("Warmup-steps: {}".format(warmup_steps))

    # Train the model
    model.fit(train_dataloader=train_dataloader,
            evaluator=evaluator,
            epochs=base_args.num_epochs,
            warmup_steps=base_args.warmup_steps,
            output_path=model_save_path,
            save_best_model= "loss",
            optimizer_params={'lr': 1e-5},
            debias=base_args.debias, #remove this if using orininal sentence-transformer libaray
            lmbda=base_args.lmbda,
            use_amp=True,
            wandb_logger= wandb_logger
              )
    model.save(model_save_path)
    #Test latest model
    testing(path=model_save_path,
            gpu=base_args.gpu_id,
            pth=pth,
            wandb_logger=wandb_logger)
    #Test latest model on counterfactuals
    testing(path=model_save_path,
            gpu=base_args.gpu_id,
            pth=pth,
            counterfactual=True,
            wandb_logger=wandb_logger)

    if wandb_logger is not None:
        with open( model_save_path+'shahed_result.pkl','rb') as file1:
            with open( model_save_path+'counter_result.pkl','rb') as file2:
                dicts=pickle.load(file1)
                dicts_counter = pickle.load(file2)
                wandb_logger.log({"Final test LDR10": LDR(pth,dicts,dicts_counter),
                                  "Final test counterfactual GAP": get_counterfactual_gap(pth,dicts,dicts_counter)})
        
if __name__ == "__main__":

    main()
