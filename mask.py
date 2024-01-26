from testing import testing



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
from testing import testing


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_id", type=int, default=0, help="gpu no.")
    parser.add_argument("--seed", type=int, default=0, help="torch random seed")
    parser.add_argument("--model_path", type=str, default="", help="model path")
    parser.add_argument("--local", default=False, action='store_true', help="changes path suitable to run locally")
    
    base_args, _ = parser.parse_known_args()


    pth = "/home/shahed/" if base_args.local==True else "/"
    model_path = base_args.model_path

    print(f"manual_seed({base_args.seed})")
    torch.manual_seed(base_args.seed)

    device = torch.device(f"cuda:{int(base_args.gpu_id)}")
    torch.cuda.manual_seed(base_args.seed)
    torch.backends.cudnn.deterministic = True

    #Test latest model
    testing(path=model_path,device=device, pth=pth, masked=True)

if __name__ == "__main__":

    main()
