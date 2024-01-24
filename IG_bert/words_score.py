import pickle
import pandas as pd
from sentence_transformers.cross_encoder import CrossEncoder
import torch
from transformers import BertTokenizerFast
from sklearn.preprocessing import LabelEncoder
from tqdm.autonotebook import tqdm
import argparse
import numpy as np
from torch.utils.data import DataLoader
from captum.attr import LayerIntegratedGradients
from torch import nn



def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_id", type=int, default=1, help="")
    parser.add_argument("--job", type=str, default="physician", help="Can be jobs like software engineer, dentist, teacher, accountant etc.")
    base_args, optional = parser.parse_known_args()

    base_args.job
    device = torch.device("cuda:"+str(base_args.gpu_id))


    with open( '/home/deepak/RecSys2023/Dataset/new_bios.pkl', 'rb') as file:
        dicts = pickle.load(file)
    bios = pd.DataFrame(dicts).reset_index()
    bios = bios.drop('index', axis=1)


    with open( '/home/deepak/RecSys2023/Dataset/test.pkl', 'rb') as file:
        dicts = pickle.load(file)
    uk_jobs_test = pd.DataFrame(dicts)

    uk_jobs_test = uk_jobs_test[uk_jobs_test['title']==base_args.job]


    model_name = '/home/deepak/RecSys2023/Dataset/BM25/original_gender/results/cross-encoder-bert-base-uncased-2023-06-14_20-34-03-latest'
    model = CrossEncoder(model_name, num_labels=1, max_length=512, device=device)
    model.model.to(device)

    tokenizer = BertTokenizerFast.from_pretrained(model_name)

    with open( '/home/deepak/RecSys2023/Dataset/BM25/original_gender/results/cross-encoder-bert-base-uncased-2023-06-14_20-34-03-latest.pkl', 'rb') as file:
        dicts = pickle.load(file)
    
    def smart_batching_collate_text_only(batch):
        texts = [[] for _ in range(len(batch[0]))]


        for example in batch:
            for idx, text in enumerate(example):
                texts[idx].append(text.strip())

        tokenized = tokenizer(*texts, padding=True, truncation='longest_first', return_tensors="pt")

        for name in tokenized:
            tokenized[name] = tokenized[name].to(device)

        return tokenized

    def create_data(sentences):
        if isinstance(sentences[0], str):
            sentences = [sentences]      
        return DataLoader(sentences, collate_fn=smart_batching_collate_text_only, shuffle=False)

    def construct_baseline( tok_len, counter=False):
        if counter==False:
            baseline = tokenizer( ' '.join(['[MASK]']*(tok_len-2)), padding=True, return_tensors="pt")
        else:
            baseline = tokenizer( ' '.join(['[MASK]']*(tok_len-2)), padding=True, return_tensors="pt")
        return baseline['input_ids'].to(device)


    def model_output(text):
        model.model.eval()
        logits = activation_default(model.model(text).logits) 
        return logits

    def summarize_attributions(attributions):
        attributions = attributions.sum(dim=-1).squeeze(0)
        attributions = attributions / torch.norm(attributions)
        return attributions

    model_input = model.model.bert.embeddings
    activation_default =  nn.Sigmoid()

    lig = LayerIntegratedGradients(model_output, model_input)



    word_score_dict = {}

    for idx in tqdm(uk_jobs_test.index):
        query_word_dict = {}

        for rank in range(10):
            dataload = create_data([[uk_jobs_test['description'][idx],bios['bio'][dicts[idx]['corpus_id'][rank]]],[uk_jobs_test['description'][idx],bios['counter_bio'][dicts[idx]['corpus_id'][rank]]]])
            attributions, _ = lig.attribute(inputs= [id['input_ids'] for id in dataload][0],
                                    baselines= construct_baseline(len(tokenizer.convert_ids_to_tokens([i['input_ids'][0] for i in dataload][0])), counter=False),
                                    return_convergence_delta=True,
                                    internal_batch_size=1
                                    )
            attributions_counter, _ = lig.attribute(inputs= [id['input_ids'] for id in dataload][1],
                                    baselines= construct_baseline(len(tokenizer.convert_ids_to_tokens([i['input_ids'][0] for i in dataload][1])), counter=False),
                                    return_convergence_delta=True,
                                    internal_batch_size=1
                                    )
            attributions_sum = summarize_attributions(attributions)
            attributions_counter_sum = summarize_attributions(attributions_counter)

            separation = tokenizer.convert_ids_to_tokens([i['input_ids'][0] for i in dataload][0]).index('[SEP]')

            new_bias_score = abs((attributions_sum-attributions_counter_sum)[:separation])

            prob = torch.softmax(new_bias_score,dim=0)
            prob[prob< (1/len(prob))] = 0

            for word_position in range(len(prob)):
                if prob[word_position]!=0:
                    try:
                        query_word_dict[tokenizer.decode([id['input_ids'] for id in dataload][0][0][word_position])] = query_word_dict[tokenizer.decode([id['input_ids'] for id in dataload][0][0][word_position])] + prob[word_position]*(1/np.log(rank+2))
                    except KeyError:
                        query_word_dict[tokenizer.decode([id['input_ids'] for id in dataload][0][0][word_position])] = prob[word_position]*(1/np.log(rank+2))
                    
        word_score_dict[idx]=query_word_dict



    with open("/home/deepak/RecSys2023/IG_bert/"+base_args.job+".pkl", 'wb') as f:
        pickle.dump(word_score_dict, f)

if __name__ == "__main__":

    main()