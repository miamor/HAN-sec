import json
import time
import numpy as np
import torch
import dgl
from dgl import load_graphs
from torch.utils.data import DataLoader

from utils.early_stopping import EarlyStopping
from utils.io import load_checkpoint
from utils.utils import label_encode_onehot, indices_to_one_hot

from utils.constants import *
# from models.model import Model
# from __save_results.gat_nw__8379__1111__cuckoo_ADung__noiapi__vocablower_noiapi_full__tfidf.model import Model

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import os
import sys
from utils.utils import load_pickle, save_pickle, save_txt

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"



def collate(samples):
    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    return batched_graph, torch.tensor(labels)


class App:

    dir_data_pickle = 'data/pickle/TuTu_sm'

    model_src_path = None

    g_train = [] #? train graphs
    l_train = [] #? train labels
    g_test = [] #? test graphs
    l_test = [] #? test labels
    n_rels = 0 #? total relations
    n_entities = 0 #? total nodes
    nentities_train = [] #? number of nodes within each graph in train set
    nentities_test = [] #? number of nodes within each graph in test set
    nrels_train = [] #? number of edges within each graph in train set
    nrels_test = [] #? number of edges within each graph in test set

    #? model config
    model_config = None
    pretrained_weight = None
    learning_config = None
    gpu = -1 #? no gpu
    early_stopping = None


    def __init__(self, config_filepath='config.model.json') -> None:
        """ Load config """
        if not os.path.isfile(config_filepath):
            print(f'[x] config file not exist')
            exit()
        
        self.__config__ = json.load(open(config_filepath))


        """ Input dir for loading graphs """
        self.dir_data_pickle = self.__config__['dir_data_pickle']
        if not os.path.exists(self.dir_data_pickle):
            print('[!] `dir_data_pickle` not exist.', self.dir_data_pickle)
            exit()


        """ Mapping labels (classes) """
        self.mapping_labels = self.__config__['mapping_labels']


        # """ Get train/list names """
        # self.train_list_file = self.__config__['train_list_file']
        # self.test_list_file = self.__config__['test_list_file']
        # if not os.path.exists(self.train_list_file):
        #     print('[!] `train_list_file` not exist.', self.train_list_file)
        #     exit()
        # if not os.path.exists(self.test_list_file):
        #     print('[!] `test_list_file` not exist.', self.test_list_file)
        #     exit()
        # self.train_list_name = [line.strip() for line in open(self.train_list_file).read().split('\n')]
        # self.test_list_name = [line.strip() for line in open(self.test_list_file).read().split('\n')]
        

        """ Load model config """
        self.model_config = self.__config__['model_config']
        if 'pretrained_weight' in self.__config__:
            if not os.path.isfile(self.__config__['pretrained_weight']):
                print('[x] `pretrained_weight` not exists.', self.__config__['pretrained_weight'])
                exit()
            self.pretrained_weight = self.__config__['pretrained_weight']
        self.learning_config = self.__config__['learning_config']
        self.gpu = self.learning_config['gpu']
        self.is_cuda = True if self.gpu >= 0 else False
        if self.gpu >= 0:
            torch.cuda.set_device(self.gpu)
        if self.model_config is None or self.learning_config is None:
            print('[x] `model_config` and `learning_config` cannot be none.')
            exit()


        """ Use early stopping """
        patience = self.__config__['early_stopping']['patience']
        self.early_stopping = EarlyStopping(patience=patience, verbose=True)


        """ Import Model builder """
        if 'model_src_path' in self.__config__:
            self.model_src_path = self.__config__['model_src_path']
            if not os.path.exists(self.model_src_path):
                print(f'[!] `model_src_path` not exists.', self.model_src_path)
        if self.model_src_path is not None:
            sys.path.insert(0, self.model_src_path)
            print('*** [app][__init__] model_src_path', self.model_src_path)
            from model_edgnn_o import Model
        else:
            from models.model_edgnn_o import Model

        self.Model = Model

        pass



    def load_data(self) -> list:
        # self.g_train = load_pickle(os.path.join(self.dir_data_pickle, 'g_train'))
        # self.g_test = load_pickle(os.path.join(self.dir_data_pickle, 'g_test'))
        # self.l_train = torch.load(os.path.join(self.dir_data_pickle, 'l_train'))
        # self.l_test = torch.load(os.path.join(self.dir_data_pickle, 'l_test'))
        self.g_train, ldict_train = load_graphs(os.path.join(self.dir_data_pickle, 'data_train.bin'))
        self.g_test, ldict_test = load_graphs(os.path.join(self.dir_data_pickle, 'data_test.bin'))
        self.l_train = ldict_train['labels']
        self.l_test = ldict_test['labels']
        # self.n_entities = load_pickle(os.path.join(self.dir_data_pickle, 'n_entities'))
        # self.n_rels = load_pickle(os.path.join(self.dir_data_pickle, 'n_rels'))
        self.nentities_train = load_pickle(os.path.join(self.dir_data_pickle, 'nentities_train'))
        self.nentities_test = load_pickle(os.path.join(self.dir_data_pickle, 'nentities_test'))
        self.nrels_train = load_pickle(os.path.join(self.dir_data_pickle, 'nrels_train'))
        self.nrels_test = load_pickle(os.path.join(self.dir_data_pickle, 'nrels_test'))
        self.n_entities = self.g_train[0]

        return



    def train(self, save_dir='', k_fold=10):
        """
        Train
        """

        print('[ ][app][train] CALLED')
        
        if len(save_dir) == 0:
            print('[x] `save_dir` cannot be empty.')
            exit()
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)


        """ Load train/test graphs """
        print('\n[ ] Load data')
        self.load_data()
        print('    len g_train', len(self.g_train))
        print('    len g_test', len(self.g_test))


        """ Build model """
        print('\n[ ] Build model')
        self.model = self.Model(g=self.g_train[0],
                           config_params=self.model_config,
                           n_classes=len(self.mapping_labels),
                           n_rels=self.nrels_train[0],
                           n_entities=self.nentities_train[0],
                           is_cuda=self.is_cuda,
                           batch_size=1,
                           model_src_path=self.model_src_path)

        if self.pretrained_weight is not None:
            print('[ ] Load pretrained weight')
            self.model = load_checkpoint(self.model, self.pretrained_weight, self.is_cuda)


        #? print model layers
        print('\n*** Model layers:')
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                print('    - ', name, param.data.type())
        print('    - self.model.fc.weight.type', self.model.fc.weight.type())


        """ Train by fold """
        K = k_fold
        loss_fcn = torch.nn.CrossEntropyLoss()
        self.accuracies = np.zeros(k_fold)
        for k in range(K):
            start = int(len(self.g_train)/K) * k
            end = int(len(self.g_train)/K) * (k+1)
            print(f'\n--------------\n[ ] Process new k = {k}/{K}  ({start} - {end})')

            # self.model = self.ModelObj(g=self.data_graph_train[0],
            #                 config_params=self.model_config,
            #                 n_classes=self.data_nclasses,
            #                 n_rels=self.data_nrels,
            #                 n_entities=self.data_nentities,
            #                 is_cuda=self.is_cuda,
            #                 batch_size=1,
            #                 model_src_path=self.model_src_path)


            optimizer = torch.optim.Adam(self.model.parameters(),
                                         lr=self.learning_config['lr'],
                                         weight_decay=self.learning_config['weight_decay'])

            """ Divide train set to train/val """
            #? train batch
            train_batch_graphs = self.g_train[:start] + self.g_train[end:]
            train_batch_labels = self.l_train[list(range(0, start)) + list(range(end+1, len(self.g_train)))]
            train_batch_samples = list(map(list, zip(train_batch_graphs, train_batch_labels)))
            train_batch = DataLoader(train_batch_samples,
                                          batch_size=self.learning_config['batch_size'],
                                          shuffle=True,
                                          collate_fn=collate)

            #? validation batch
            val_batch_graphs = self.g_train[start:end]
            val_batch_labels = self.l_train[start:end]
            val_batch = dgl.batch(val_batch_graphs)

            # print('    train_batch', train_batch)
            # print('    val_batch', val_batch)
            # print('    len train_batch: ', len(train_batch))
            # print('    len val_batch: ', len(val_batch))
            print('    len train_batch_graphs: ', len(train_batch_graphs))
            print('    len val_batch_graphs', len(val_batch_graphs))

            print('    val_batch_graphs[0].number_of_nodes()', val_batch_graphs[0].number_of_nodes())
            print('    val_batch_graphs[-1].number_of_nodes()', val_batch_graphs[-1].number_of_nodes())
            
            dur = []
            for epoch in range(self.learning_config['epochs']):
                self.model.train()
                if epoch >= 3:
                    t0 = time.time()
                losses = []
                training_accuracies = []
                for iter_idx, (batch_graph, label) in enumerate(train_batch):
                    # print('~~~ batch_graph', batch_graph)
                    logits = self.model(batch_graph)
                    
                    if self.is_cuda:
                        label = label.cuda()
                    
                    loss = loss_fcn(logits, label)
                    losses.append(loss.item())
                    _, indices = torch.max(logits, dim=1)

                    # print('~~~~ logits', logits)
                    # print('------------------')
                    # print('\t indices', indices)
                    # print('\t label', label)
                    correct = torch.sum(indices == label)
                    training_accuracies.append(correct.item() * 1.0 / len(label))

                    optimizer.zero_grad()
                    loss.backward(retain_graph=True)
                    # loss.backward()
                    optimizer.step()

                if epoch >= 3:
                    dur.append(time.time() - t0)

                val_acc, val_loss, _ = self.model.eval_graph_classification(val_batch_labels, val_batch)

                print('[+] Epoch {:05d} | Time(s) {:.4f} | train_acc {:.4f} | train_loss {:.4f} | val_acc {:.4f} | val_loss {:.4f}'.format(epoch, np.mean(dur) if dur else 0, np.mean(training_accuracies), np.mean(losses), val_acc, val_loss))

                is_better = self.early_stopping(val_loss, self.model, save_dir)
                if is_better:
                    self.accuracies[k] = val_acc


                if self.early_stopping.early_stop:
                    #? Print model's state_dict
                    print('    *** Model state_dict:')
                    for param_tensor in self.model.state_dict():
                        print(param_tensor, '\t', self.model.state_dict()[param_tensor].size())

                    #? Print optimizer's state_dict
                    print('    *** Optimizer state_dict:')
                    for var_name in optimizer.state_dict():
                        print(var_name, '\t', optimizer.state_dict()[var_name])

                    #? Save state dict
                    torch.save(self.model.state_dict(), save_dir+'/model_state.pt')

                    #? Save model
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_loss': val_loss,
                    }, save_dir+'/saved')


                    print('    [+] Early stopping')
                    break

            self.early_stopping.reset()


if __name__ == '__main__':
    app = App()
    app.train('output/TuTu_sm')