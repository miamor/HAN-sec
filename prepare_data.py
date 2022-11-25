"""
prep_data__doc2vec_edge_node
"""
from re import S
import sys
sys.path.insert(1, '../..')
from utils.utils import indices_to_one_hot, label_encode_onehot, sample_mask, preprocess_adj, preprocess_features, save_pickle, load_pickle, save_txt, care_APIs
import shutil
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from utils.word_embedding import TFIDF, Doc2Vec_
from utils.constants import *
import torch
from dgl import DGLGraph
from dgl import save_graphs, load_graphs
import networkx as nx
from graphviz import Source

#? for python < 3.9
from typing import Tuple as tuple


class PrepareData():

    save_json = True #? save processed output to disk json file
    do_draw = True #? save created graph to dot for visualization
    render_svg = True #? output svg for graphviz graphs ?
    # prepend_vocab = False #? assume I have a vocab for node & edge generated on another dataset, when I use this vocab to encode node & edge for a new dataset, I can encounter new words (node name / edge (API) name). When I meet new words that cannot be found in vocab, do I want to extend the vocab by prepending this new word to the vocab
    create_vocab = True #? create vocab when do encode
    train_embedder = True #? train node/edge embedding
    process_from = 'report' #? (report | json | graph) process from cuckoo reports (report) or after processing behavior json file (json) or from graph files (if from graph files then run `pack_graphs` only)
    split_train_test = True
    train_ratio = 70

    #? under each dir_data_..., data must be grouped by label (eg. each data dir contains 2 folder: benign & malware, under which contains files)
    dir_data_report = 'data/reports/TuTu_sm' #? directory that contains cuckoo reports by labels
    dir_data_json = 'data/json/TuTu_sm'
    dir_data_pickle = 'data/pickle/TuTu_sm'
    dir_data_embedding = 'data/embeddings/TuTu_sm'
    dir_data_graph = 'data/graphs/TuTu_sm' #? store encoded graph for training / testing
    dir_data_graphviz = 'data/graphviz/TuTu_sm' #? store dot file for visualization
    dir_data_networkx = 'data/nx/TuTu_sm' #? store visualization of networkx graph (img)
    
    mapping_labels = {'benign': 0, 'malware': 1}


    #? 7 types of node
    n_types = {
        'process': 0, #? process_handle
        'file': 1, #? file_handle
        'registry': 2, #? registry key_handle

        'process_api': 3, #? API category process (API that interacts with process)
        'file_api': 4, #? API category file (API that interacts with file)
        'registry_api': 5, #? API category registry (API that interacts with registry)

        'other_api': 4,
    }
    #? 6 types for edge
    e_types = {
        'process__process_api': 0, #? connects a process with a process API (a pid calls to a process_api or a process_api interacts with a process handle)
        'process__registry_api': 1, #? connects a process with a registry API (a pid calls to a registry_api)
        'process__file_api': 2, #? connects a process with a file API (a pid calls to a file_api)
        'file__file_api': 3, #? connects a file handle with a file API (a file_api interacts with a file handle)
        'registry__registry_api': 4, #? connects a registry handle with a registry API (a registry_api interacts with a registry handle)
        'process__other_api': 5, #? connects a process with an API out of interstt (a pid calls to an api)
    }

    #? for visualization
    node_color = ['red', 'orange', 'blue', 'pink', 'yellow', 'cyan']
    map_ntype2style = {
        'process': {
            'shape': 'octagon',
            'style': 'filled',
            'color': 'black',
            'fillcolor': '#3c00783d',
            'fontcolor': '#3c0078', #? purple
        },
        'file': {
            'shape': 'oval',
            'style': 'filled',
            'color': 'black',
            'fillcolor': '#1a28eb3d',
            'fontcolor': '#1a28eb', #? blue
        },
        'registry': {
            'shape': 'oval',
            'style': 'filled',
            'color': 'black',
            'fillcolor': '#db21653d',
            'fontcolor': '#db2165', #? pink
        },
        'process_api': {
            'shape': 'box',
            'style': '',
            'color': 'gray',
            'fillcolor': '#3c00783d', 
            'fontcolor': '#3c0078',
        },
        'file_api': {
            'shape': 'box',
            'style': '',
            'color': 'gray',
            'fillcolor': '#1a28eb3d', 
            'fontcolor': '#1a28eb',
        },
        'registry_api': {
            'shape': 'box',
            'style': '',
            'color': 'gray',
            'fillcolor': '#db21653d', 
            'fontcolor': '#db2165',
        },
        'other_api': {
            'shape': 'box',
            'style': '',
            'color': 'gray',
            'fillcolor': '#0fa8753d', 
            'fontcolor': '#0fa875', #? light green
        },
    }
    map_etype2style = {
        'process__process_api': '#3c007888', #? dark purple
        'process_api__process': '#3c007878', #? dark purple

        'process__registry_api': '#a3074088', #? dark pink
        'registry_api__process': '#a3074078', #? dark pink
        
        'process__file_api': '#080e6188', #? dark blue
        'file_api__process': '#080e6178', #? dark blue
        
        'file__file_api': '#1a28eb88', #? blue
        'file_api__file': '#1a28eb78', #? blue
        
        'registry__registry_api': '#db216588', #? pink
        'registry_api__registry': '#db216578', #? pink
        
        'process__other_api': '#1d013888', #? dark purple
        'other_api__process': '#1d013878', #? dark purple
    }

    #? = True to take into account only "interesting" apis (a predefined list)
    use_interesting_apis = False
    interesting_apis = care_APIs() + list(n_types.keys()) + ['other'] #? used when `use_interesting_apis` = True



    def __init__(self, config_filepath='config.prepare.json') -> None:
        """ Load config """
        if not os.path.isfile(config_filepath):
            print(f'[x] config file not exist')
            exit()
        
        self.__config__ = json.load(open(config_filepath))
        print(self.__config__)

        """ Overwrite """
        # self.dir_vocab = self.__config__['dir_vocab']
        self.dir_data_embedding = self.__config__['dir_data_embedding']
        
        
        """ Process from cuckoo reports (report) 
            or after processing behavior json file (json) 
            or from graph files (if from graph files then run `pack_graphs` only) """
        self.process_from = self.__config__['process_from'] #? report | json | graph

        
        """ Split dataset into train/test """
        self.split_train_test = self.__config__['split_train_test']
        self.train_ratio =  self.__config__['train_ratio']


        """ Mapping labels (classes) """
        self.mapping_labels = self.__config__['mapping_labels']


        """ Visualize """
        self.do_draw = self.__config__['do_draw']


        """ Load train/test data """
        self.train_list_file = self.__config__['train_list_file']
        self.test_list_file = self.__config__['test_list_file']
        if self.split_train_test is False:
            if not os.path.exists(self.train_list_file):
                print('[!] `train_list_file` not exist.', self.train_list_file)
                exit()
            if not os.path.exists(self.test_list_file):
                print('[!] `test_list_file` not exist.', self.test_list_file)
                exit()
            self.train_list_name = [line.strip() for line in open(self.train_list_file).read().split('\n')]
            self.test_list_name = [line.strip() for line in open(self.test_list_file).read().split('\n')]
        else:
            if self.train_ratio <= 0:
                print('[!] `train_ratio` cannot be <= 0 if `split_train_test` = false')


        """ Node/edge embedder """
        max_ft = self.__config__['max_ft']
        top_k = self.__config__['top_k']
        vector_size = self.__config__['vector_size']
        dm = self.__config__['dm']
        self.emb_trained_path__node_name = os.path.join(self.dir_data_embedding, 'node_csv')
        self.emb_trained_path__edge_args = os.path.join(self.dir_data_embedding, 'edge_csv')
        self.emb_corpus_path__node_name = os.path.join(self.dir_data_pickle, 'node_csv')
        self.emb_corpus_path__edge_args = os.path.join(self.dir_data_pickle, 'edge_csv')
        #? create dir if not exist
        for dir in [self.emb_trained_path__node_name, self.emb_trained_path__edge_args, self.emb_corpus_path__node_name, self.emb_corpus_path__edge_args]:
            if not os.path.isdir(dir):
                os.makedirs(dir)
        #? what embedder for node
        if self.__config__['node_embedder'] == 'doc2vec':
            self.node_embedder = Doc2Vec_(self.emb_trained_path__node_name, self.emb_corpus_path__node_name, self.mapping_labels, vector_size, dm)
        elif self.__config__['node_embedder'] == 'tfidf':
            self.node_embedder = TFIDF(self.emb_trained_path__node_name, self.emb_corpus_path__node_name, self.mapping_labels, max_ft, top_k)
        else:
            print('[x] `node_embedder` must be either `doc2vec` or `tfidf`')
            exit()
        #? what embedder for edge
        if self.__config__['edge_embedder'] == 'doc2vec':
            self.edge_embedder = Doc2Vec_(self.emb_trained_path__edge_args, self.emb_corpus_path__edge_args, self.mapping_labels, vector_size, dm)
        elif self.__config__['edge_embedder'] == 'tfidf':
            self.edge_embedder = TFIDF(self.emb_trained_path__edge_args, self.emb_corpus_path__edge_args, self.mapping_labels, max_ft, top_k)
        else:
            print('[x] `edge_embedder` must be either `doc2vec` or `tfidf`')
            exit()

        self.hard_reset()

        return


    def from_set(self) -> None:
        """
        Process a set of reports.
        Folder of a set must divide into n subfolders (n = num classes)
        """
        self.dir_data_report = self.__config__['dir_data_report']
        self.dir_data_json = self.__config__['dir_data_json']
        self.dir_data_pickle = self.__config__['dir_data_pickle']
        self.dir_data_graph = self.__config__['dir_data_graph']
        self.dir_data_graphviz = self.__config__['dir_data_graphviz']
        self.dir_data_networkx = self.__config__['dir_data_networkx']


        """ Create data dir if not exist """
        for dir in [self.dir_data_embedding, self.dir_data_report, self.dir_data_json, self.dir_data_pickle, self.dir_data_graph, self.dir_data_graphviz, self.dir_data_networkx]:
            print(f'[ ] Checking dir {dir}')
            if not os.path.isdir(dir):
                print(f'[!] Not exist. Creating {dir}')
                os.makedirs(dir)
        #? these data dir has sub folders which represent classes (labels)
        for dir in [self.dir_data_report, self.dir_data_json, self.dir_data_graph, self.dir_data_graphviz, self.dir_data_networkx]:
            for lbl in self.mapping_labels.keys():
                dir_data_smt_by_lbl = os.path.join(dir, lbl)
                if not os.path.isdir(dir_data_smt_by_lbl):
                    os.makedirs(dir_data_smt_by_lbl)


        """ Encode / create graphs """
        if self.process_from == 'report':
            self.encode_from_reports_set()
        elif self.process_from == 'json':
            self.encode_from_processed_behaviors()


        """ Pack all things into one pickle file to load when training/testing"""
        self.pack_graphs()
        
        return
    

    def from_files(self, report_paths) -> None:
        """
        Process reports files.
        """
        if self.train_embedder is True:
            print('[x] encode one single report does not allow `train_embedder` = True')
            return
        
        self.encode_from_reports_files(report_paths)
        return



    def pack_graphs(self) -> None:
        """
        Pack all graphs into one pickle file to be loaded on training / testing
        """
        print('\n----------------------------\n[pack_graphs] Pack graphs and all necessary stuff for training')

        g_train = [] #? train graphs
        l_train = [] #? train labels
        g_test = [] #? test graphs
        l_test = [] #? test labels
        nentities_train = [] #? number of nodes within each graph in train set
        nentities_test = [] #? number of nodes within each graph in test set
        nrels_train = [] #? number of edges within each graph in train set
        nrels_test = [] #? number of edges within each graph in test set
        
        #? pack train
        for filename in self.train_list_name:
            lbl = filename.split('__')[0]
            graph_path = os.path.join(self.dir_data_graph, lbl, filename.replace('.json.json', '.json').replace('.json', '.bin'))
            # graph = load_pickle(graph_path)
            #? this graph_path contains only 1 graph
            graphs, label_dict = load_graphs(graph_path)
            graph = graphs[0]
            # label = label_dict['labels'][0]
            g_train.append(graph)
            l_train.append(self.mapping_labels[lbl])
            nentities_train.append(graph.number_of_nodes())
            nrels_train.append(graph.number_of_edges())
        l_train = torch.LongTensor(l_train)
        
        #? pack test
        for filename in self.test_list_name:
            lbl = filename.split('__')[0]
            graph_path = os.path.join(self.dir_data_graph, lbl, filename.replace('.json.json', '.json').replace('.json', '.bin'))
            # graph = load_pickle(graph_path)
            #? this graph_path contains only 1 graph
            graphs, label_dict = load_graphs(graph_path)
            graph = graphs[0]
            # label = label_dict['labels'][0]
            g_test.append(graph)
            l_test.append(self.mapping_labels[lbl])
            nentities_test.append(graph.number_of_nodes())
            nrels_test.append(graph.number_of_edges())
        l_test = torch.LongTensor(l_test)

        #? save
        # torch.save(l_train, os.path.join(self.dir_data_pickle, 'l_train'))
        # torch.save(l_test, os.path.join(self.dir_data_pickle, 'l_test'))
        # save_pickle(g_train, os.path.join(self.dir_data_pickle, 'g_train'))
        # save_pickle(g_test, os.path.join(self.dir_data_pickle, 'g_test'))
        save_graphs(os.path.join(self.dir_data_pickle, 'data_train.bin'), g_train, {'labels': l_train})
        save_graphs(os.path.join(self.dir_data_pickle, 'data_test.bin'), g_test, {'labels': l_test})
        # save_pickle(len(self.map_nodestr_node), os.path.join(self.dir_data_pickle, 'n_entities'))
        # save_pickle(len(self.map_edgestr_edge), os.path.join(self.dir_data_pickle, 'n_rels'))
        save_pickle(nentities_train, os.path.join(self.dir_data_pickle, 'nentities_train'))
        save_pickle(nentities_test, os.path.join(self.dir_data_pickle, 'nentities_test'))
        save_pickle(nrels_train, os.path.join(self.dir_data_pickle, 'nrels_train'))
        save_pickle(nrels_test, os.path.join(self.dir_data_pickle, 'nrels_test'))
        shutil.copy(self.train_list_file, self.dir_data_pickle)
        shutil.copy(self.test_list_file, self.dir_data_pickle)
        json.dump(self.mapping_labels, open(os.path.join(self.dir_data_pickle, 'mapping_labels.json'), 'w'))
        
        del g_train, g_test, l_train, l_test, nentities_train, nentities_test, nrels_train, nrels_test #? cleanup

        return


    def hard_reset(self) -> None:
        """
        Hard reset all variables
        """
        self.embedding_data__node_names_by_graph = {}
        self.embedding_data__edge_args_by_graph = {}
        self.embedding_data__node_names_csv = []
        self.embedding_data__edge_args_csv = []
        self.flags_keys = []
        self.total_nodes = 0
        self.total_edges = 0

        self.reset()
        return


    def reset(self) -> None:
        """
        These params are used when generating graph.
        This function resets all these params.
        Should be called every time constructing a new graph
        """

        #? keep track of current working report / current graph (use graph_name (= {label}__{report_name}) to set value for self.current_graph)
        self.current_graph = ''
        self.current_graph_label = ''

        #? keep track of what nodes / edges have been added to the working graph
        #? map a node identifier string (eg. {type}__{name}) to its node idx
        self.map_nodestr_node = {}
        #? map an edge identifier string ({s_node_idx}--{d_node_idx}) to its edge idx
        self.map_edgestr_edge = {}

        if self.do_draw:
            #? networkx for visualization
            self.Gx = nx.DiGraph()
            self.Gx_nodes = {}
            self.Gx_edges_by_type = {}
            for e_type in self.map_etype2style.keys(): #? a bit different to visualize
                self.Gx_edges_by_type[e_type] = []
            self.Gx_nodes_lbl_by_type = {}
            for n_type in self.n_types:
                self.Gx_nodes_lbl_by_type[n_type] = {}

            #? graphviz for visualization
            self.g_codes = {
                'nodes': {},
                'edges': {}
            }

        return


    def encode_from_reports_set(self) -> None:
        """
        Create graphs from a set of cuckoo reports (from `dir_data_reports`)
        """
        print('\n----------------------------\n[encode_from_reports_set] Process data from reports folder to json files and encode nodes & edges')

        """ Read & process all reports
            Process a set of reports.
            Folder of a set must divide into n subfolders (n = num classes) """
        for label in self.mapping_labels.keys():
            #? report dir for this label
            report_dir_lbl = os.path.join(self.dir_data_report, label)

            #? loop through all reports in this folder and process each report
            for filename in os.listdir(report_dir_lbl):
                # #! DEBUG
                # if '01c1096bf695dc00aa03698ba9024a4e5af96021ccf3dc3a05ca233693773ebd__4912' not in filename:
                #     continue
                self.process_report_file(filename, os.path.join(self.dir_data_report, label, filename), label)


        """ Prepare stuff for training node/edge embedder """
        print('\n----------------------------\n[encode_from_reports_set] Generating corpus for node/edge embedding')
        self.gen_embedding_data__edge_args()
        self.gen_embedding_data__node_name()
        self.node_embedder.save_corpus(self.embedding_data__node_names_csv)
        self.edge_embedder.save_corpus(self.embedding_data__edge_args_csv)
        json.dump(self.flags_keys, open(os.path.join(self.dir_data_embedding, 'flags_keys.json'), 'w'))


        """ Double check train/test filenames """
        if self.split_train_test is False:
            """ If we have list of train / test files, we might recheck the list.
                Some reports may have empty behaviors. 
                Therefore, after processing behaviors, some may not have graph constructed. 
                We need to remove these items from train / test list """
            print('\n----------------------------\n[encode_from_reports_files] Checking train/test list')
            #? modify train set
            for i,filename in enumerate(self.train_list_name):
                if not os.path.exists(os.path.join(self.dir_data_json, filename.split('__')[0], f'{filename}.json')): #? not able to create graph for this file
                    del self.train_list_name[i]
            #? modify test set
            for i,filename in enumerate(self.test_list_name):
                if not os.path.exists(os.path.join(self.dir_data_json, filename.split('__')[0], f'{filename}.json')): #? not able to create graph for this file
                    del self.test_list_name[i]
            #? overwrite train/test list
            open(self.train_list_file, 'w').write('\n'.join(self.train_list_name))
            open(self.test_list_file, 'w').write('\n'.join(self.test_list_name))
            print(f'[+] Saved {len(self.train_list_name)} files to {self.train_list_file}')
            print(f'[+] Saved {len(self.test_list_name)} files to {self.test_list_file}')

        else:
            """ Or if we need to create a train/test list """
            print('\n----------------------------\n[encode_from_reports_files] Creating train/test list')
            self.train_list_name = []
            self.test_list_name = []
            for label in self.mapping_labels.keys():
                dir_json_this_lbl = os.path.join(self.dir_data_json, label)
                tot_files_this_lbl = len([name for name in os.listdir(dir_json_this_lbl)])
                print('[ ] dir_json_this_lbl', dir_json_this_lbl)
                print('    tot_files_this_lbl', tot_files_this_lbl)
                print('    self.train_ratio', self.train_ratio)
                train_end_idx = tot_files_this_lbl*self.train_ratio
                print(f'   {label} train_end_idx : {train_end_idx}')
                n = 0
                for filename in os.listdir(dir_json_this_lbl):
                    n += 1
                    if n <= train_end_idx:
                        self.train_list_name.append(filename)
                    else:
                        self.test_list_name.append(filename)
            open(self.train_list_file, 'w').write('\n'.join(self.train_list_name))
            open(self.test_list_file, 'w').write('\n'.join(self.test_list_name))
            print(f'[+] Saved {len(self.train_list_name)} files to {self.train_list_file}')
            print(f'[+] Saved {len(self.test_list_name)} files to {self.test_list_file}')



        """ Train & load node/edge embedding """
        if self.train_embedder:
            print('\n----------------------------\n[encode_from_reports_set] Train node/edge embedding')
            self.encode_from_processed_behaviors(call_load_node_edge_embedding=False)
        else:
            self.encode_from_processed_behaviors()

        return


    def encode_from_reports_files(self, report_paths=None) -> None:
        """
        Create graphs from cuckoo reports (from `report_paths`)
        """
        print('\n----------------------------\n[encode_from_reports_files] Process data from reports files to json files and encode nodes & edges')

        """ Read & process all reports """
        if report_paths is None or len(report_paths) == 0:
            print(f'[!] Process from files require `report_paths` to be a list and not empty')

        """ Process a list of reports.
            Simply loop through whole list """
        #? loop through all reports in this folder and process each report
        for filepath in os.listdir(report_paths):
            self.process_report_file(os.path.basename(filepath), filepath)

        if self.train_embedder:
            print('\n[!] Creating graph from reports files does not allow retrain embedder')
        
        self.encode_from_processed_behaviors()

        return
    

    def encode_from_processed_behaviors(self, call_load_node_edge_embedding=True) -> None:
        """
        Create graphs from processed behaviors json file (from `dir_data_json`)
        """
        print('\n----------------------------\n[encode_from_processed_behaviors] Encode nodes & edges')

        if self.process_from == 'json' and self.train_embedder:
            print('\n[!] Creating graph from reports files does not allow retrain embedder')
        

        """ Load node/edge embedding """
        if call_load_node_edge_embedding:
            print('\n[ ] Load node/edge embedding')
            self.load_node_edge_embedding()


        """ Encode node edge for graph. Create graph """
        for label in self.mapping_labels.keys():
            #? json dir for this label
            json_dir_lbl = os.path.join(self.dir_data_json, label)

            #? loop through all json files in this folder and encode each json
            # self.total_nodes = 0
            # self.total_edges = 0
            for filename in os.listdir(json_dir_lbl):
                filepath = os.path.join(json_dir_lbl, filename)
                print(f'\n---------\n[ ] Encoding {filepath}')
                n_nodes, n_edges = self.create_dglgraph(filepath)

        return



    def process_report_file(self, filename, filepath, graph_label='none') -> None:
        print(f'\n---------\n[ ] Processing {filepath}')
        filename = os.path.basename(filepath)

        #? read behaviors from this report
        report_data = json.load(open(filepath))
        if 'behavior' not in report_data.keys():
            print('[!][read_report] No behavior tag found. Skip')
            return
        behavior = report_data['behavior']
                
        print('[ ] Processing behavior data from report')
        self.process_report_behavior(behavior, filename, graph_label)
        del behavior #? cleanup

        """ Save processed json """
        if len(self.map_nodestr_node) > 0 and len(self.map_edgestr_edge) > 0:
            print(f'[+] Done. {len(self.map_nodestr_node)} nodes & {len(self.map_edgestr_edge)} edges. Saving to json file...')
            json_data = {
                'nodes': self.map_nodestr_node,
                'edges': self.map_edgestr_edge,
            }
            json_path = os.path.join(self.dir_data_json, self.current_graph_label, f'{self.current_graph}.json')
            json.dump(json_data, open(json_path, 'w'))
            del json_data #? cleanup
        else:
            print(f'[!] Done. {len(self.map_nodestr_node)} nodes & {len(self.map_edgestr_edge)} edges. Skip.')
            return


        """ Save graphviz / networkx graphs (for visualization) """
        if self.do_draw:
            self.visualize_graphviz()
            del self.g_codes #? cleanup
            self.visualize_networkx()
            del self.Gx, self.Gx_nodes, self.Gx_nodes_lbl_by_type, self.Gx_edges_by_type #? cleanup
        
        return


    def train_node_edge_embedding(self) -> None:
        """
        Train node/edge embedding
        """
        #? prepare
        self.node_embedder.prepare(self.embedding_data__node_names_csv, self.train_list_name, self.test_list_name)
        self.edge_embedder.prepare(self.embedding_data__edge_args_csv, self.train_list_name, self.test_list_name, self.flags_keys)
        #? train
        self.word_to_ix_edge, word_dict_edge = self.edge_embedder.train()
        self.word_to_ix_node, word_dict_node = self.node_embedder.train()
        # #? append to vocab
        # for w in word_dict_node:
        #     if w not in self.word_dict_node:
        #         self.word_dict_node.append(w)
        # for w in word_dict_edge:
        #     if w not in self.word_dict_edge:
        #         self.word_dict_edge.append(w)
        # open(self.vocab_node_filepath, 'w').write(' '.join(self.word_dict_node))
        # open(self.vocab_edge_filepath, 'w').write(' '.join(self.word_dict_edge))
        return

    def load_node_edge_embedding(self) -> None:
        """
        Load pretrained node/edge embedding
        """
        self.word_to_ix_edge, word_dict_edge = self.edge_embedder.load()
        self.word_to_ix_node, word_dict_node = self.node_embedder.load()
        return



    def process_report_behavior(self, behavior, report_name, report_label) -> None:
        """
        Process the data extracted from the report and save to dir_data_json.
        Use node/edge information to prepare for the node/edge embedder.
        Generate graphviz / networkx graph, for visualization only. For train/test, we need to use DGLGraph which requires running other stuff (encode node, edge, etc.)

        Arguments:
            - behavior: `behavior` tag from report
            - report_name: graph name (graph id)
            - report_label: class of report (graph label)
        """

        self.reset() #? reset graph data (constructing new graph)


        graph_name = f'{report_label}__{report_name}'
        self.current_graph = graph_name
        self.current_graph_label = report_label
        

        """ Get all processes """
        procs = behavior['processes']

        #? loop through each process
        for proc in procs:
            #? get the process' api calls
            calls = proc['calls']
            pid = proc['pid']

            proc_name = proc['process_path']
            # proc_info = '{}|{}'.format(graph_name, proc_name)

            if len(calls) == 0: #? no api call
                print(f'[!] proc {pid} has no api call. Skip')
                continue
            
            print(f'[*] proc {pid} has {len(calls)} api calls. Process...')
            
            #? loop through api calls 
            for api_call in calls:
                """ 
                For each API, we will:
                    - Check if the node that represents the process which calls this API exists in the graph. If not, create a node to represent the process that calls this API.
                    - Create a node for this API. Type of this node is one of `process_api`, `file_api`, `registry_api` (all api types defined in `n_types`)
                    - Draw a link (edge) from  [ the process that calls this api ] -> [ this API ]
                    - Create a node that represents the handle which this API performs actions on (eg. file_api will perform action on a file handle,...). Type of this node will be one of `process`, `file`, `registry` (all handle types defined in `n_types`)
                    - Draw a link (edge) between this API and the handle that this API interacts with. The direction is based on the functionality of the API, for example, 
                        if the API is to read, then the direction is   API -> handle
                        if the API is to write, then the direction is  API <- handle
                """
                cat = api_call['category']
                api_name = api_call['api'].lower()
                api_args = api_call['arguments']
                del api_call['api']
                api_flags = None
                if 'flags' in api_call and api_call['flags'] is not None and len(api_call['flags']) > 0:
                    api_flags = api_call['flags']

                if cat not in ['file', 'process', 'registry']:
                    continue

                #? if `use_interesting_apis` = True, then consider only intersting apis
                if self.use_interesting_apis and api_name not in self.interesting_apis:
                    continue

                #? if there is information in flag field, then embed edge using these arguments
                # if len(api_call['flags']) > 0:
                #     self.add_edge_args(api_call['flags'])
                
                """ Create a node to represent the process that calls this API if not exist """
                #? add proc node
                node_proc__data = {
                    'type': 'process',
                    'name': 'proc_{}'.format(proc['pid']),
                    'proc_name': proc['process_name'],
                    'pid': proc['pid'],
                }
                node_proc__idx, node_proc__data = self.insert_node(node_proc__data)
                if node_proc__idx is None:
                    continue


                """ Create a node for this API. Check if the node that represents this API exists in the graph. 
                    If not, create a node to represent this API. 
                    Type of this node is one of `process_api`, `file_api`, `registry_api` (all api types defined in `n_types`) """
                #! now, only name is used as the only characteristic
                #? add api node
                node_api_type = f'{cat}_api' if cat in ['process', 'file', 'registry'] else 'other_api'
                node_api__data = {
                    'type': node_api_type, #? file_api | process_api | registry_api | other_api
                    'name': api_name,                        
                }
                node_api__idx, node_api__data = self.insert_node(node_api__data)
                if node_api__idx is None:
                    continue


                """ Draw a link (edge) from  [ the process that calls this api ] -> [ this API ] """
                # self.edge(node_api__data, node_proc__data, args={'edge_type': 'proc__file_api'})
                self.insert_edge(node_proc__idx, node_proc__data, node_api__idx, node_api__data, args=api_flags)


                """ Create a node that represents the handle which this API performs actions on (ie. a file handle). 
                    Type of this node will be `file` (defined in `n_types`) """
                #? get handle address
                handle_adr = api_args['file_handle'] if 'file_handle' in api_args else api_args['key_handle'] if 'key_handle' in api_args else api_args['handle'] if 'handle' in api_args else None

                if handle_adr is None:
                    #! should we delete the api node if no handle is attached to it ?
                    continue

                #? add handle node
                node_handle__data = {
                    'type': cat, #? file | process | registry
                    'name': f'{cat}__{handle_adr}', #'file{'+handle_adr+'}',
                }
                node_handle__idx, node_handle__data = self.insert_node(node_handle__data)
                if node_handle__idx is None:
                    continue


                """ Draw a link (edge) between this API and the handle that this API interacts with. 
                    The direction is based on the functionality of the API, for example, 
                        if the API is to read, then the direction is   API -> handle
                        if the API is to write, then the direction is  API <- handle """
                buffer_length = 0
                if 'buffer' in api_args and 'length' in api_args:
                    buffer_length = api_args['length']
                
                #? create edge from the api node to the handle node
                etype = '{}__{}'.format(node_handle__data['type'], node_api__data['type'])
                if 'open' in api_name or 'set' in api_name or 'write' in api_name or 'create' in api_name: #? api -> handle
                    self.insert_edge(node_api__idx, node_api__data, node_handle__idx, node_handle__data, args=api_flags, buffer_size=buffer_length, e_type=etype)
                else: #? handle -> api
                    self.insert_edge(node_handle__idx, node_handle__data, node_api__idx, node_api__data, args=api_flags, buffer_size=buffer_length, e_type=etype)


        return



    def add_embedding_data__node_name(self, name) -> None:
        """
        Add node name to data dict used for embedding
        """

        if len(name) == 0:
            return
        
        if self.current_graph not in self.embedding_data__node_names_by_graph:
            self.embedding_data__node_names_by_graph[self.current_graph] = [name]
        else:
            self.embedding_data__node_names_by_graph[self.current_graph] += [name]
        
        return


    def gen_embedding_data__node_name(self) -> None:
        """
        Generate csv file to use for embedding node
        """

        print('[ ][gen_embedding_data__node_name] self.node_names_by_graph', len(self.embedding_data__node_names_by_graph))
        # print('[ ][gen_embedding_data__node_name] self.node_names_by_graph', self.embedding_data__node_names_by_graph)
        
        for graph_name in self.embedding_data__node_names_by_graph:
            label = graph_name.split('__')[0]
            nodes_names = ' '.join(self.embedding_data__node_names_by_graph[graph_name])
            if len(label) > 0:
                self.embedding_data__node_names_csv.append({'class': self.mapping_labels[label], 'data': nodes_names, 'file': graph_name})
            else:
                self.embedding_data__node_names_csv.append({'class': -1, 'data': nodes_names, 'file': graph_name})

        return


    def add_edge_args_embedding_data(self, flags) -> None:
        """
        Add edge arguments to data dict used for embedding
        """

        if flags is None or len(flags) == 0:
            return

        flags_data = []
        # print('flags', flags)
        for flag_key in sorted(flags):
            if len(flags[flag_key]) == 0:
                continue

            if flag_key not in self.flags_keys:
                self.flags_keys.append(flag_key)

            if isinstance(flags[flag_key], list):
                # print('flags', flags, 'flag_key', flag_key, 'flags[flag_key]', flags[flag_key])
                    
                flag_data = flag_key
                flag_has_value = False
                for v in flags[flag_key]:
                    # print('\t v', v)
                    if v is not None:
                        flag_data += ' ' + v.replace('|', ' ').lower()
                        flag_has_value = True
                if flag_has_value is True:
                    flags_data.append(flag_data)
            else:
                # flag_data = flags[flag_key].replace('|', ' ').lower()
                flag_data = flag_key + ' ' + flags[flag_key].replace('|', ' ').lower()
                flags_data.append(flag_data)
        
        if len(flags_data) > 0:
            if self.current_graph not in self.embedding_data__edge_args_by_graph:
                self.embedding_data__edge_args_by_graph[self.current_graph] = flags_data
            else:
                self.embedding_data__edge_args_by_graph[self.current_graph] += flags_data
        
        return


    def gen_embedding_data__edge_args(self) -> None:
        """
        Generate csv file to use for embedding edge
        """

        print('[ ][gen_embedding_data__edge_args] self.edge_args_by_graph', len(self.embedding_data__edge_args_by_graph))
        # print('[ ][gen_embedding_data__edge_args] self.edge_args_by_graph', self.embedding_data__edge_args_by_graph)

        for graph_name in self.embedding_data__edge_args_by_graph:
            label = graph_name.split('__')[0]
            flags_data_txt = ' '.join(self.embedding_data__edge_args_by_graph[graph_name])
            if len(label) > 0:
                self.embedding_data__edge_args_csv.append({'class': self.mapping_labels[label], 'data': flags_data_txt, 'file': graph_name})
            else:
                self.embedding_data__edge_args_csv.append({'class': -1, 'data': flags_data_txt, 'file': graph_name})

        return



    def insert_node(self, node, skip_duplicate=True):
        """
        Insert node with specific type.

        Arguments:
            - node: a dict of node attribute
                {
                    (mandatory) name: str,
                    (mandatory) type: str,
                    (optional)...
                }
            - skip_duplicate: 
                + If set to True, the function first checks if this node exists in G then insert if this node not exists yet.
                + If set to False, insert anyway
        Returns:
            node_idx

        --------------------------------------

        One note is that sometimes process_handle would be 0xffffffff, or 0x00000000, which obviously makes no sense.
        So let's check if is all ffff, if it is, do NOT insert it as a node
        """

        n_type = node['type']
        n_name = node['name']
        node_identifier_str = f'{n_type}__{n_name}'


        """ Sometimes node will not be inserted to the graph """
        n_handle_adr = None
        if '0x' in n_name:
            n_handle_adr = n_name[2:]
            la = len(n_handle_adr)
            if n_name == '0' or (n_handle_adr is not None and (n_handle_adr == '0'*la or n_handle_adr == 'f'*la)):
                return None, None


        """ Only now insert node to graph """
        if skip_duplicate is False or node_identifier_str not in self.map_nodestr_node.keys():
            #? get node idx
            node_idx = len(self.map_nodestr_node)
            self.map_nodestr_node[node_identifier_str] = node

            node['id'] = node_idx
            node['graph'] = self.current_graph
            node['graph_label'] = self.current_graph_label
            # self.json_data['nodes'][node_idx] = node
            self.add_embedding_data__node_name(n_name)


            """ For visualization """
            if self.do_draw:
                n_shape = self.map_ntype2style[n_type]['shape']
                n_style = self.map_ntype2style[n_type]['style']
                n_color = self.map_ntype2style[n_type]['color']
                n_fillcolor = self.map_ntype2style[n_type]['fillcolor']
                n_fontcolor = self.map_ntype2style[n_type]['fontcolor']
                #? Networkx graph
                self.Gx.add_node(node_idx)
                self.Gx_nodes[node_idx] = node
                self.Gx_nodes[node_idx]['color'] = n_fillcolor
                self.Gx_nodes[node_idx]['shape'] = n_shape
                self.Gx_nodes_lbl_by_type[n_type][node_idx] = node_idx if n_type == 'api' else n_name
                #? Graphviz graph
                n_txt = f'{node_idx} {node_identifier_str}'
                self.g_codes['nodes'][node_idx] = f'node [shape="{n_shape}" style="{n_style}" color="{n_color}" fontcolor="{n_fontcolor}" fillcolor="{n_fillcolor}"] {node_idx} [label="{n_txt}"]'

            return node_idx, node
        
        return self.map_nodestr_node[node_identifier_str]['id'], self.map_nodestr_node[node_identifier_str]



    def insert_edge(self, s_node_idx, source_node, d_node_idx, dest_node, args=None, buffer_size=None, e_type=None) -> None:
        """
        Insert edge from source_node to dest_node, with specific type.

        Arguments:
            - s_node_idx: source node idx
            - d_node_idx: dest node idx
            - source_node: a dict of node attribute
            - dest_node: a dict of node attribute
            - e_type: edge type. 
                If not defined (None), use default value = {source['type']}__{dest['type']}.
                If not None, use e_type
        
        --------------------------------------

        Check if source_node and dest_node exists in G.
        Check if edge(s, d) exists in G. 
            If yes, skip
            If not, insert
        """

        #? do not allow self loop. uncomment to allow
        if s_node_idx == d_node_idx:
            return

        if e_type is None:
            e_type = '{}__{}'.format(source_node['type'], dest_node['type'])

        if source_node['type'] not in self.n_types or dest_node['type'] not in self.n_types:
            return
        
        if s_node_idx is not None and d_node_idx is not None:
            edge_identifier_str = f'{s_node_idx}--{d_node_idx}'

            if edge_identifier_str not in self.map_edgestr_edge:
                edge_idx = len(self.map_edgestr_edge)
                edge = {
                    'type': e_type,
                    'id': edge_idx,
                    'args': args if args is not None and len(args) > 0 else {},
                    'from': s_node_idx,
                    'to': d_node_idx,
                    'buffer_size': buffer_size if buffer_size is not None else -1,
                    'graph': self.current_graph,
                    'graph_label': self.current_graph_label
                }

                self.map_edgestr_edge[edge_identifier_str] = edge
                self.add_edge_args_embedding_data(args)

                """ For visualization """
                if self.do_draw:
                    #? Networkx graph 
                    self.Gx.add_edge(s_node_idx, d_node_idx)
                    self.Gx_edges_by_type[e_type].append((s_node_idx, d_node_idx))
                    #? Graphviz graph
                    # self.g_codes.append(f'{s_node_idx} -> {d_node_idx} [color="{e_color}" label="{source_node_str}->{dest_node_str}"]')
                    self.g_codes['edges'][edge_identifier_str] = f'{s_node_idx} -> {d_node_idx} [color="{self.map_etype2style[e_type]}"]'

        return



    def create_dglgraph(self, json_path) -> tuple[int, int]:
        """
        Encode nodes & edges from processed json file
        Output: a dgl graph for a report. Save to dir_data_graph
        """

        json_data = json.load(open(json_path))
        self.current_graph = os.path.basename(json_path).split('.json')[0]
        self.current_graph_label = os.path.basename(os.path.dirname(json_path))

        """ Construct graph """
        self.current_dglgraph = DGLGraph(multigraph=True)

        n_nodes = len(json_data['nodes'])
        for node_str in json_data['nodes']:
            node = json_data['nodes'][node_str]
            self.encode_node(node)
            # print(f'[+] Finished encoding node {node_str}')
        
        n_edges = len(json_data['edges'])
        for edge_str in json_data['edges']:
            edge = json_data['edges'][edge_str]
            self.encode_edge(edge)
            # print(f'[+] Finished encoding edge {edge_str}')

        """ Save generated graph """
        dgl_path = os.path.join(self.dir_data_graph, self.current_graph_label, f'{self.current_graph}.bin')
        print('[+] self.current_dglgraph', self.current_dglgraph)
        # save_pickle(self.current_dglgraph, dgl_path)
        save_graphs(dgl_path, [self.current_dglgraph], {'labels': torch.tensor([self.mapping_labels[self.current_graph_label]])})
        print(f'[+] Graph saved to {dgl_path}')
        
        del json_data #? cleanup
        
        return n_nodes, n_edges
    


    def encode_node(self, node) -> None:
        """
        Encode node information to node attribute
        ----------------------------
            Calculate node attributes (init features)
            All nodes must have same features space.
        """

        ndata = {}

        """ GNN_NODE_TYPES_KEY """
        node_type_encoded = indices_to_one_hot(self.n_types[node['type']], out_vec_size=len(self.n_types))
        nte_torch = torch.from_numpy(np.array([node_type_encoded])).type(torch.FloatTensor)
        ndata[GNN_NODE_TYPES_KEY] = nte_torch

        """ GNN_NODE_LABELS_KEY """
        name_transformed = self.node_embedder.transform(self.nodename_to_str(node['name']))
        cbow_node = torch.tensor(name_transformed).type(torch.FloatTensor)
        ndata[GNN_NODE_LABELS_KEY] = cbow_node.view(1, -1)

        """ dgl require field nid """
        ndata['nid'] = torch.tensor([node['id']])

        """ add node with data to graph """
        self.current_dglgraph.add_nodes(1, data=ndata)


        return


    def encode_edge(self, edge) -> None:
        """
        Encode edge information to node attribute
        """

        edata = {}

        """ GNN_EDGE_TYPES_KEY """
        edge_type_encoded = indices_to_one_hot(self.e_types[edge['type']], out_vec_size=len(self.e_types))
        ete_torch = torch.from_numpy(np.array([edge_type_encoded])).type(torch.FloatTensor)
        # print('ete_torch', ete_torch)
        edata[GNN_EDGE_TYPES_KEY] = ete_torch

        """ GNN_EDGE_LABELS_KEY """
        # cbow_edge = self.cbow_encode_edge_args(self.args_to_strargs_to_str(edge['args']))
        # args_transformed, txt_chosen = self.edge_embedder.transform(self.args_to_str(edge['args']))
        # print('edge', edge, '|', edge['graph'])
        # print('\t Transform edge:', self.args_to_str(edge['args']))
        args_transformed = self.edge_embedder.transform(self.args_to_str(edge['args']))
        # print('\t args_transformed:', args_transformed)
        cbow_edge = torch.tensor(args_transformed).type(torch.FloatTensor)
        edata[GNN_EDGE_LABELS_KEY] = cbow_edge.view(1, -1)
        
        """ dgl require field eid """
        edata['from'] = torch.tensor([edge['from']]) #? source node id (s_node_idx)
        edata['to'] = torch.tensor([edge['to']]) #? dest node id (d_node_idx)
        edata['eid'] = torch.tensor([edge['id']])

        # print("\nself.args_to_str(edge['args'])", txt_chosen, '||', self.args_to_str(edge['args']))
        # print('edata[GNN_EDGE_TYPES_KEY]', edata[GNN_EDGE_TYPES_KEY])
        # print('edata[GNN_EDGE_LABELS_KEY]', edata[GNN_EDGE_LABELS_KEY])
        # print('edata[GNN_EDGE_LABELS_KEY].shape', edata[GNN_EDGE_LABELS_KEY].shape)

        """ GNN_EDGE_BUFFER_SIZE_KEY """
        edata[GNN_EDGE_BUFFER_SIZE_KEY] = torch.Tensor([[edge['buffer_size']]])
        # print('edata[GNN_EDGE_LABELS_KEY]', edata[GNN_EDGE_LABELS_KEY])
        # print('edata[GNN_EDGE_BUFFER_SIZE_KEY]', edata[GNN_EDGE_BUFFER_SIZE_KEY])
        # print(edge['buffer_size'])
        
        """ add edge with data to graph """
        self.current_dglgraph.add_edge(edge['from'], edge['to'], data=edata)

        return


    def visualize_graphviz(self) -> None:
        """ 
        Visualize graph using graphviz
        """
        dot_file = os.path.join(self.dir_data_graphviz, self.current_graph_label, f'{self.current_graph}.dot')
        print(f'[ ] Saving to {dot_file}')

        dot_temp = 'digraph G{'
        dot_temp += '\n'.join(list(self.g_codes['nodes'].values()))
        dot_temp += '\n'.join(list(self.g_codes['edges'].values()))
        dot_temp += '\n}'

        # """ Save the dot template and svg visualization for the graph of this file """
        # s = Source(dot_temp, filename=os.path.join(dot_out_dir, f'{report_name}.gv'), format='svg') #? save the graph
        # s.view()

        """ Save the dot template for the graph of this file """
        s = Source(dot_temp, filename=dot_file) 
        s.save(skip_existing=None)
        # s.render()
        del dot_temp

        if self.render_svg:
            """ Automatically generate output file names based on the input file name and the various output formats specified by the -T flags.
                $ dot -Tsvg -O ~/family.dot ~/debug.dot
                Generates ~/family.dot.svg and ~/debug.dot.svg files. """
            os.system(f'dot -Tsvg -O {dot_file}')
        
        return


    def visualize_networkx(self) -> None:
        """ 
        Visualize graph using networkx
        """
        nx_file = os.path.join(self.dir_data_networkx, self.current_graph_label, f'{self.current_graph}.svg')
        print(f'[ ] Saving to {nx_file}')

        plt.figure(self.current_graph, figsize=(20,20))

        pos = nx.spring_layout(self.Gx)
            
        #? short version of building nodes_color array
        nodes_color = [self.map_ntype2style[self.Gx_nodes[node_idx]['type']]['fillcolor'] for node_idx in self.Gx.nodes()]
        #? draw nodes
        nx.draw_networkx_nodes(self.Gx, pos, 
                                cmap=plt.get_cmap('jet'),
                                node_color=nodes_color, 
                                node_size=200
                            )
        for n_type in self.Gx_nodes_lbl_by_type:
            nx.draw_networkx_labels(self.Gx, pos, 
                                    labels=self.Gx_nodes_lbl_by_type[n_type],
                                    font_size=6,
                                    font_color=self.map_ntype2style[n_type]['fontcolor']
                                )

        #? draw edges
        # edges = self.Gx.edges()
        # print('\nself.Gx_nodes_lbl_by_type', self.Gx_nodes_lbl_by_type)
        # print('\nself.Gx_edges_by_type', self.Gx_edges_by_type)
        for e_type in self.Gx_edges_by_type:
            nx.draw_networkx_edges(self.Gx, pos, 
                                    edgelist=self.Gx_edges_by_type[e_type], 
                                    edge_color=self.map_etype2style[e_type], 
                                    arrows=True
                                )

        #? save output
        plt.tight_layout()
        plt.savefig(nx_file)
        plt.clf()
        # plt.show()

        return


    def nodename_to_str(self, txt) -> str:
        txt = txt.lower().strip()

        if self.use_interesting_apis is False:
            return txt.split('{')[0]

        if txt.split('{')[0] not in self.interesting_apis:
            print(f'[!] Not in interesting_apis. Convert from `{txt}` to `other`')
            return 'other'

        return txt.split('{')[0]

    def args_to_str(self, args_):
        # get flags values only. ignore keys
        # print('args_', args_)
        arr_val = []
        for key in sorted(args_):
            values_txt = key + ' ' + args_[key].replace('|', ' ')
            # values_txt = args_[key].replace('|', ' ')
            arr_val += values_txt.split(' ')
        arr_val = [v.strip().lower() for v in arr_val]
        # if len(arr_val) == 0:
        #     arr_val = ["null"] * 4
        str_val = ' '.join(arr_val)
        # print('str_val', str_val)
        return str_val



if __name__ == '__main__':
    config_filepath = sys.argv[1] if len(sys.argv) > 1 else ''
    preparer = PrepareData(config_filepath)
    preparer.from_set()