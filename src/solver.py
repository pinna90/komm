import os
import math
from math import isnan
import re
import pickle
import gensim
import numpy as np
from tqdm import tqdm
from tqdm import tqdm_notebook
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from scipy.special import expit

import torch
import torch.nn as nn
import pandas as pd
#from openpyxl.workbook import Workbook

from torch.nn import functional as F
torch.manual_seed(123)
torch.cuda.manual_seed_all(123)

from utils import to_gpu, time_desc_decorator, DiffLoss, MSE, SIMSE, CMD
import models


class Solver(object):
    def __init__(self, train_config, dev_config, test_config, train_data_loader, dev_data_loader, test_data_loader, is_train=True, model=None):

        self.train_config = train_config
        self.epoch_i = 0
        self.train_data_loader = train_data_loader
        self.dev_data_loader = dev_data_loader
        self.test_data_loader = test_data_loader
        self.is_train = is_train
        self.model = model

    
    @time_desc_decorator('Build Graph')
    def build(self, cuda=True):

        if self.model is None:
            self.model = getattr(models, self.train_config.model)(self.train_config)
        
        # Final list
        for name, param in self.model.named_parameters():

            # Bert freezing customizations 
            if self.train_config.data == "mosei":
                if "bertmodel.encoder.layer" in name:
                    layer_num = int(name.split("encoder.layer.")[-1].split(".")[0])
                    if layer_num <= (8):
                        param.requires_grad = False
            elif self.train_config.data == "ur_funny":
                if "bert" in name:
                    param.requires_grad = False
            
            if 'weight_hh' in name:
                nn.init.orthogonal_(param)
            print('\t' + name, param.requires_grad)

        # Initialize weight of Embedding matrix with Glove embeddings
        if not self.train_config.use_bert:
            if self.train_config.pretrained_emb is not None:
                self.model.embed.weight.data = self.train_config.pretrained_emb
            self.model.embed.requires_grad = False
        
        if torch.cuda.is_available() and cuda:
            self.model.cuda()

        if self.is_train:
            self.optimizer = self.train_config.optimizer(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=self.train_config.learning_rate)
    
            

    @time_desc_decorator('Training Start!')
    def train(self):
        curr_patience = patience = self.train_config.patience
        num_trials = 1

        # self.criterion = criterion = nn.L1Loss(reduction="mean")
        if self.train_config.data == "ur_funny": # 2022.1.30
            self.criterion = criterion = nn.CrossEntropyLoss(reduction="mean")
        else: # mosi and mosei are regression datasets -> mosi
            self.criterion = criterion = nn.MSELoss(reduction="mean")

        self.domain_loss_criterion = nn.CrossEntropyLoss(reduction="mean")
        self.sp_loss_criterion = nn.CrossEntropyLoss(reduction="mean")
        self.loss_diff = DiffLoss()
        self.loss_recon = MSE()
        self.loss_cmd = CMD()
        
        best_valid_loss = float('inf')
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.5)
        
        train_losses = []
        valid_losses = []
        
        #file 
        ft = open("./loss.txt", 'w')
        count = 0 # loss 개수
        for e in range(self.train_config.n_epoch):
            count += 1 # loss 개수
            self.model.train()

            train_loss_cls, train_loss_sim, train_loss_diff = [], [], []
            train_loss_recon = []
            train_loss_sp = []
            train_loss = []
            
            
            for batch in self.train_data_loader:
                
                
                self.model.zero_grad()
                t, v, a, y, f, l, bert_sent, bert_sent_type, bert_sent_mask = batch # 2023.10.13

                batch_size = t.size(0)
                t = to_gpu(t)
                v = to_gpu(v)
                a = to_gpu(a)
                y = to_gpu(y)
                l = to_gpu(l)
                bert_sent = to_gpu(bert_sent)
                bert_sent_type = to_gpu(bert_sent_type)
                bert_sent_mask = to_gpu(bert_sent_mask)

                y_tilde = self.model(t, v, a, l, bert_sent, bert_sent_type, bert_sent_mask)
                
                ########################################
                encoder_layer = self.model.transformer_encoder.layers[0]

                '''
                # 모델의 가중치 중에서 특정 레이어의 가중치 접근
                for name, param in encoder_layer.named_parameters():
                    print('test')
                    print(name)
                    print(param.shape)
                    print(param)
                    # 첫 번째 인코더 레이어의 self-attention 가중치에 접근
                    if "layers.0.self_attn" in name:
                        print(f"{name}: {param.size()}")
                '''
                #########################################
                
              
                
                
                #########################################
                
                
                #if self.train_config.data == "ur_funny":
                if self.train_config.data == "ur_funny": # 수정함 2022.1.30
                    #print('test')
                    y = y.squeeze()
                
                #print('y_tilde', y_tilde) # 비정상.!
                #print('y', y) # 정상적으로 들어옴
                cls_loss = criterion(y_tilde, y) # 수정 필요 2022.1.30
                diff_loss = self.get_diff_loss()
                domain_loss = self.get_domain_loss()
                recon_loss = self.get_recon_loss()
                cmd_loss = self.get_cmd_loss()
                
                #print('train loss test')
                #print(cls_loss)
                #print(diff_loss)
                #print(domain_loss)
                #print(recon_loss)
                #print(cmd_loss)
                
                #ft.write("======================" + str(count) + "======================\n")
                #ft.write("cls_loss"+ str(cls_loss) + "\n")
                #ft.write("diff_loss" + str(diff_loss) + "\n")
                #ft.write("domain_loss" + str(domain_loss) + "\n")
                #ft.write("recon_loss" + str(recon_loss) + "\n")
                #ft.write("cmd_loss" + str(cmd_loss) + "\n")
                #ft.write("\n")
                    
                
                if self.train_config.use_cmd_sim:
                    similarity_loss = cmd_loss
                else:
                    similarity_loss = domain_loss
                
                loss = cls_loss + \
                    self.train_config.diff_weight * diff_loss + \
                    self.train_config.sim_weight * similarity_loss + \
                    self.train_config.recon_weight * recon_loss

                loss.backward()
                
                torch.nn.utils.clip_grad_value_([param for param in self.model.parameters() if param.requires_grad], self.train_config.clip)
                self.optimizer.step()

                train_loss_cls.append(cls_loss.item())
                train_loss_diff.append(diff_loss.item())
                train_loss_recon.append(recon_loss.item())
                train_loss.append(loss.item())
                train_loss_sim.append(cmd_loss.item())
                
                
            ft.write("======================" + str(count) + "======================\n")
            train_losses.append(train_loss)
            print(f"Training loss: {round(np.mean(train_loss), 4)}")
            
            ft.write(str(round(np.mean(train_loss_sim), 4)))
            ft.write("\n")
            
            ft.write(str(round(np.mean(train_loss_diff), 4)))
            ft.write("\n")
            
            ft.write(str(round(np.mean(train_loss_recon), 4)))
            ft.write("\n")
            
            
            
            valid_loss, valid_acc = self.eval(mode="dev")
            
            print(f"Current patience: {curr_patience}, current trial: {num_trials}.")
            if valid_loss <= best_valid_loss:
                best_valid_loss = valid_loss
                print("Found new best model on dev set!")
                if not os.path.exists('checkpoints'): os.makedirs('checkpoints')
                torch.save(self.model.state_dict(), f'checkpoints/model_{self.train_config.name}.std')
                torch.save(self.optimizer.state_dict(), f'checkpoints/optim_{self.train_config.name}.std')
                curr_patience = patience
            else:
                curr_patience -= 1
                if curr_patience <= -1:
                    print("Running out of patience, loading previous best model.")
                    num_trials -= 1
                    curr_patience = patience
                    self.model.load_state_dict(torch.load(f'checkpoints/model_{self.train_config.name}.std'))
                    self.optimizer.load_state_dict(torch.load(f'checkpoints/optim_{self.train_config.name}.std'))
                    lr_scheduler.step()
                    print(f"Current learning rate: {self.optimizer.state_dict()['param_groups'][0]['lr']}")
            
            if num_trials <= 0:
                print("Running out of patience, early stopping.")
                break
        ft.close() # txt 파일 닫기
        self.eval(mode="test", to_print=True)

        

    
    def eval(self,mode=None, to_print=False):
        assert(mode is not None)
        self.model.eval()

        y_true, y_pred = [], []
        file_code = [] # 2023.11.29
        eval_loss, eval_loss_diff = [], []
        
        #file 
        fd = open("./dev_loss.txt", 'w')# 2024.02.25
                          
        if mode == "dev":
            dataloader = self.dev_data_loader
        elif mode == "test":
            dataloader = self.test_data_loader

            if to_print:
                self.model.load_state_dict(torch.load(
                    f'checkpoints/model_{self.train_config.name}.std'))
            

        with torch.no_grad():
            count = 0 # loss 개수 # 2024.02.25
            dev_loss_cls, dev_loss_sim, dev_loss_diff = [], [], []
            dev_loss_recon = []
            dev_loss_sp = []
            dev_loss = []
            
            for batch in dataloader:
                count += 1 # loss 개수 # 2024.02.25                   

                #print('misclassification test')          
                #print('batch') # id는 안들어 있나? 2023. 10. 13
                #print(batch[0][2])
                self.model.zero_grad()
                t, v, a, y, f, l, bert_sent, bert_sent_type, bert_sent_mask = batch # 2023.10.13

                t = to_gpu(t)
                v = to_gpu(v)
                a = to_gpu(a)
                y = to_gpu(y)
                l = to_gpu(l)
                bert_sent = to_gpu(bert_sent)
                bert_sent_type = to_gpu(bert_sent_type)
                bert_sent_mask = to_gpu(bert_sent_mask)

                y_tilde = self.model(t, v, a, l, bert_sent, bert_sent_type, bert_sent_mask)

                if self.train_config.data == "ur_funny":
                    y = y.squeeze()
                
                cls_loss = self.criterion(y_tilde, y)
                loss = cls_loss
                        
                diff_loss = self.get_diff_loss() # 2024.02.25
                domain_loss = self.get_domain_loss() # 2024.02.25
                recon_loss = self.get_recon_loss() # 2024.02.25
                cmd_loss = self.get_cmd_loss()      # 2024.02.25  
                
                fd.write("======================" + str(count) + "======================\n")
                fd.write("cls_loss"+ str(cls_loss) + "\n")
                fd.write("diff_loss" + str(diff_loss) + "\n")
                fd.write("domain_loss" + str(domain_loss) + "\n")
                fd.write("recon_loss" + str(recon_loss) + "\n")
                fd.write("cmd_loss" + str(cmd_loss) + "\n")
                fd.write("\n")      
                          
                dev_loss_cls.append(cls_loss.item())
                dev_loss_diff.append(diff_loss.item())
                dev_loss_recon.append(recon_loss.item())
                dev_loss.append(loss.item())
                dev_loss_sim.append(cmd_loss.item())          
                          
                          
                eval_loss.append(loss.item())
                #print(type(eval_loss))         
                #ft.write(' '.join(map(str, eval_loss)))
                #ft.write("\n")
                y_pred.append(y_tilde.detach().cpu().numpy())
                y_true.append(y.detach().cpu().numpy())
                file_code.append(f.detach().cpu().numpy())
                # 여기에 file을 append 해줘야함
            
            #fd.write(str(round(np.mean(dev_loss_sim), 4)))
            #fd.write("\n")

            #fd.write(str(round(np.mean(dev_loss_diff), 4)))
            #fd.write("\n")

            #fd.write(str(round(np.mean(dev_loss_recon), 4)))
            #fd.write("\n")
                          
        fd.close() # txt 파일 닫기      # 2024.02.25                
        eval_loss = np.mean(eval_loss)
        y_true = np.concatenate(y_true, axis=0).squeeze()
        y_pred = np.concatenate(y_pred, axis=0).squeeze()
        file_code = np.concatenate(file_code)

        accuracy = self.calc_metrics(y_true, y_pred, file_code, mode, to_print)
        
        return eval_loss, accuracy

    def multiclass_acc(self, preds, truths):
        """
        Compute the multiclass accuracy w.r.t. groundtruth
        :param preds: Float array representing the predictions, dimension (N,)
        :param truths: Float/int array representing the groundtruth classes, dimension (N,)
        :return: Classification accuracy
        """
        return np.sum(np.round(preds) == np.round(truths)) / float(len(truths))

    # 실험시 중요                      
    def calc_metrics(self, y_true, y_pred, files, mode=None, to_print=False):
        """
        Metric scheme adapted from:
        https://github.com/yaohungt/Multimodal-Transformer/blob/master/src/eval_metrics.py
        """

        #print(self.train_config.data)
        #if self.train_config.data == "ur_funny":
        if self.train_config.data == "ur_funny": # 2023.01.30 # 매번 try 할 때 마다 돌아감
            #print(y_true) # 왜 0만 있는건지...? (수정 완료)
            #print(y_pred) # 700개
            
            test_preds = np.argmax(y_pred, 1)
            test_truth = y_true

            if to_print:
                print("Confusion Matrix (pos/neg) :")
                print(confusion_matrix(test_truth, test_preds))
                print("Classification Report (pos/neg) :")
                print(classification_report(test_truth, test_preds, digits=5))
                print("Accuracy (pos/neg) ", accuracy_score(test_truth, test_preds))
            
            
            return accuracy_score(y_test, y_pred)
            #acc = 0.75
            #return acc

        else:
          
  
            test_preds = y_pred
            test_truth = y_true
            
            if to_print:
                print('y_pred len', len(y_pred))    
                print('y_pred type', type(y_pred))
                print(y_pred)
                print('y_true', len(y_true)) # Label
                print('y_true type', type(y_true))
                print(y_true)
                
                 # Label 표출 코드
                print("Test: True Label, Pred Label")
                print("file code", type(files))
                print(files.shape)
                print(files)

                # 파일명으로 변환하는 함수
                def convert_to_filename(tensor):
                    return ''.join([chr(value) for value in tensor])

                # 파일명, y_true, y_pred를 포함하는 DataFrame 생성
                data = {
                    'Filename': [convert_to_filename(files[i:i+23]) for i in range(0, len(files), 23)],
                    'True Label': y_true,
                    'Predicted Label': y_pred
                }
                df = pd.DataFrame(data)

                # Excel 파일로 저장
                # CSV 파일로 저장
                csv_filename = 'output.csv'
                df.to_csv(csv_filename, index=False)         
                          
            non_zeros = np.array([i for i, e in enumerate(test_truth) if e != 0])

            test_preds_a7 = np.clip(test_preds, a_min=-3., a_max=3.)
            test_truth_a7 = np.clip(test_truth, a_min=-3., a_max=3.)
            test_preds_a5 = np.clip(test_preds, a_min=-2., a_max=2.)
            test_truth_a5 = np.clip(test_truth, a_min=-2., a_max=2.)

            mae = np.mean(np.absolute(test_preds - test_truth))   # Average L1 distance between preds and truths
            corr = np.corrcoef(test_preds, test_truth)[0][1]
            mult_a7 = self.multiclass_acc(test_preds_a7, test_truth_a7)
            mult_a5 = self.multiclass_acc(test_preds_a5, test_truth_a5)
            
            f_score = f1_score((test_preds[non_zeros] > 0), (test_truth[non_zeros] > 0), average='weighted')
            
            # pos - neg
            binary_truth = (test_truth[non_zeros] > 0)
            binary_preds = (test_preds[non_zeros] > 0)

           # 2023.11.29 Binary_truth와 Binary_preds를 찍어봐야함               
                          
                          
            if to_print:
                print("mae: ", mae)
                print("corr: ", corr)
                print("mult_acc: ", mult_a7)
                print("Classification Report (pos/neg) :")
                print(classification_report(binary_truth, binary_preds, digits=5))
                print("Accuracy (pos/neg) ", accuracy_score(binary_truth, binary_preds))
            
            # non-neg - neg
            binary_truth = (test_truth >= 0)
            binary_preds = (test_preds >= 0)

            if to_print:
                print("Classification Report (non-neg/neg) :")
                print(classification_report(binary_truth, binary_preds, digits=5))
                print("Accuracy (non-neg/neg) ", accuracy_score(binary_truth, binary_preds))
            
            return accuracy_score(binary_truth, binary_preds)


    def get_domain_loss(self,):

        if self.train_config.use_cmd_sim:
            return 0.0
        
        # Predicted domain labels
        domain_pred_t = self.model.domain_label_t
        domain_pred_v = self.model.domain_label_v
        domain_pred_a = self.model.domain_label_a

        # True domain labels
        domain_true_t = to_gpu(torch.LongTensor([0]*domain_pred_t.size(0)))
        domain_true_v = to_gpu(torch.LongTensor([1]*domain_pred_v.size(0)))
        domain_true_a = to_gpu(torch.LongTensor([2]*domain_pred_a.size(0)))

        # Stack up predictions and true labels
        domain_pred = torch.cat((domain_pred_t, domain_pred_v, domain_pred_a), dim=0)
        domain_true = torch.cat((domain_true_t, domain_true_v, domain_true_a), dim=0)

        return self.domain_loss_criterion(domain_pred, domain_true)

    def get_cmd_loss(self,):

        if not self.train_config.use_cmd_sim:
            return 0.0

        # losses between shared states
        loss = self.loss_cmd(self.model.utt_shared_t, self.model.utt_shared_v, 5)
        loss += self.loss_cmd(self.model.utt_shared_t, self.model.utt_shared_a, 5)
        loss += self.loss_cmd(self.model.utt_shared_a, self.model.utt_shared_v, 5)
        loss = loss/3.0

        return loss

    def get_diff_loss(self):

        shared_t = self.model.utt_shared_t
        shared_v = self.model.utt_shared_v
        shared_a = self.model.utt_shared_a
        private_t = self.model.utt_private_t
        private_v = self.model.utt_private_v
        private_a = self.model.utt_private_a

        # Between private and shared
        loss = self.loss_diff(private_t, shared_t)
        loss += self.loss_diff(private_v, shared_v)
        loss += self.loss_diff(private_a, shared_a)

        # Across privates
        loss += self.loss_diff(private_a, private_t)
        loss += self.loss_diff(private_a, private_v)
        loss += self.loss_diff(private_t, private_v)

        return loss
    
    def get_recon_loss(self, ):

        loss = self.loss_recon(self.model.utt_t_recon, self.model.utt_t_orig)
        loss += self.loss_recon(self.model.utt_v_recon, self.model.utt_v_orig)
        loss += self.loss_recon(self.model.utt_a_recon, self.model.utt_a_orig)
        loss = loss/3.0
        return loss





