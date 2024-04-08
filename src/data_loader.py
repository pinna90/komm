import random
import numpy as np
import pickle

from tqdm import tqdm_notebook
from collections import defaultdict

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Dataset
from transformers import *

from create_dataset import MOSI, MOSEI, UR_FUNNY, PAD, UNK

bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

class MSADataset(Dataset):
    def __init__(self, config): # config 데이터 directory 탐색 시 사용

        ## Fetch dataset
        if "mosi" in str(config.data_dir).lower(): # (Check - 2022.12.4)
            #with open('./total_data.pickle', 'rb') as f:
                #dataset = pickle.load(f)    
            dataset = MOSI(config) # (Check - 2022.12.4)
        else: # (Check - 2022.12.4)
            print("Dataset not defined correctly") # (Check - 2022.12.4)
            exit() # (Check - 2022.12.4)
        
        self.data, self.word2id, self.pretrained_emb = dataset.get_data(config.mode) # ToDo
       
        with open('./total_data_text.pickle', 'rb') as f:
            self.data = pickle.load(f)   
    
        #with open('./../dataset/total_data.pickle', 'rb') as f:
        #    self.data = pickle.load(f) 
        
        #with open('./total_data_.pickle', 'rb') as f:
        #    self.data = pickle.load(f) 
        
        # (Check - 2024.1.29)
        if config.mode == 'train':
            with open('./sample_train_text.pickle', 'rb') as f:
            #with open('./../dataset/train_.pickle', 'rb') as f:
                self.data = pickle.load(f)
        elif config.mode == 'dev':
            with open('./sample_dev_text.pickle', 'rb') as f:
            #with open('./../dataset/dev_.pickle', 'rb') as f:
                self.data = pickle.load(f)
        #elif config.mode == 'test':
        #    with open('./../dataset/test_.pickle', 'rb') as f:
        #        self.data = pickle.load(f)
        
        self.len = len(self.data) #length (Check - 2022.12.4)
               
        config.visual_size = self.data[0][0][1].shape[1] # (5, 47) # Visual Feature (Check - 2022.12.4)
        config.acoustic_size = self.data[0][0][2].shape[1] # (5, 74) # Acoustic Feature (Check - 2022.12.4)

        config.word2id = self.word2id # (Check - 2022.12.4)
        config.pretrained_emb = self.pretrained_emb # (Check - 2022.12.4)

    def __getitem__(self, index): # (Check - 2022.12.4)
        return self.data[index] # (Check - 2022.12.4)

    def __len__(self): # (Check - 2022.12.4)
        return self.len # (Check - 2022.12.4)


def get_loader(config, shuffle=True):
    """Load DataLoader of given DialogDataset"""
    
    dataset = MSADataset(config) # MSADataset object (Check - 2022.12.4)
    
    # load
    #with open('./total_data.pickle', 'rb') as f:
    #    dataset = pickle.load(f)
    
    # print(type(dataset.data[0][0][0]))
    
    # 데이터 변경 필요!
    # 하나하나 차근차근!
    
    # (ToDO - 2022.12.4) 데이터 구조 파악해야함!
    
    print(dataset.data[0][1]) # Label (Check - 2022.12.40
    print(dataset.data[0][2]) # File name
    print(type(dataset.data[0][1]))
    # print(dataset.data[0][0][0]) # sentence (id로 표현됨) embedding
    # print(dataset.word2id)
    # print(dataset.data[0][0][1]) # Visual Feature # Numpy # 5 X 47
    # print(dataset.data[0][0][2]) # Acoustic Feature # Numpy # 5 X 74
    
    # type(dataset.data) : list, length 1283
    # type(dataset.data[i]) : tuple, length 3, index 0~1282
    # type(dataset.data[0][i]) : tuple, length 4, index 0~2(index0 : feature, index1 : label, index2 : 파일명...? (서로 다른 문장) 
    # print(dataset.data[0][1]) # e.g. [[2.4]]
    
    # type(dataset.data[0][0][0]) : type numpy, 
    # type(dataset.data[0][0][1] : type numpy, 5 X 47
    # type(type(dataset.data[0][0][2]) : type numpy, 5 X 74
    # type(dataset.data[0][0][3]) : type list, # Sentence 전체가 나눠져 들어감 ex. ['anyhow', 'it', 'was', 'really', 'good']
      
    #quit()
    
    config.data_len = len(dataset) #(Check - 2022.12.4)
    
    def collate_fn(batch): 
        '''
        Collate functions assume batch = [Dataset[i] for i in index_set]
        '''
        # for later use we sort the batch in descending order of length
        batch = sorted(batch, key=lambda x: x[0][0].shape[0], reverse=True)
        
        # get the data out of the batch - use pad sequence util functions from PyTorch to pad things
        
        #print(type(sample[1])) # test - 2023.1.16
                    
        labels = torch.cat([torch.from_numpy(sample[1]) for sample in batch], dim=0)
        #print(labels.shape)
        #files = torch.cat([np.array([ord(char) for char in sample[2]]) for sample in batch]) # 2023.10.13
        #print('파일명 테스트') # 2023.11.29
        #print(batch[0][2]) # acriil_ang_00001272.png 이런 형태의 파일명이 들어있음
        #print('batch 길이', len(batch)) # 32개
        files = torch.cat([torch.tensor([ord(char) for char in sample[2]]) for sample in batch]) # 736개(32개 * 23) 너무 잘됨
        #print(files)
        #print(files.shape)
        
        sentences = pad_sequence([torch.LongTensor(sample[0][0]) for sample in batch], padding_value=PAD)
        #print(sentences.shape)
        visual = pad_sequence([torch.FloatTensor(sample[0][1]) for sample in batch])
        acoustic = pad_sequence([torch.FloatTensor(sample[0][2]) for sample in batch])

        ## BERT-based features input prep

        SENT_LEN = sentences.size(0)
        # Create bert indices using tokenizer

        bert_details = []
        for sample in batch:
            text = " ".join(sample[0][3])
            encoded_bert_sent = bert_tokenizer.encode_plus(
                text, max_length=SENT_LEN+2, add_special_tokens=True, pad_to_max_length=True)
            bert_details.append(encoded_bert_sent)


        # Bert things are batch_first
        bert_sentences = torch.LongTensor([sample["input_ids"] for sample in bert_details])
        bert_sentence_types = torch.LongTensor([sample["token_type_ids"] for sample in bert_details])
        bert_sentence_att_mask = torch.LongTensor([sample["attention_mask"] for sample in bert_details])


        # lengths are useful later in using RNNs
        lengths = torch.LongTensor([sample[0][0].shape[0] for sample in batch])
        #print('test result')
        return sentences, visual, acoustic, labels, files, lengths, bert_sentences, bert_sentence_types, bert_sentence_att_mask

    data_loader = DataLoader( # Pytorch 
        dataset=dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn)

    return data_loader
