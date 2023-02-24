## Sample data & train output (checkpoint): 
https://drive.google.com/drive/folders/1mr8SYoCvkJEcT0YWV1UMAsZQg1h3Qqei?usp=sharing
## Full data (binaries):
https://drive.google.com/drive/folders/1_FzZUIXfC652Z7GdlT9lyYUikx9Q4MiR?usp=sharing

## Prepare data, create graphs
```bash
python prepare_data.py
```

#### Modify `config.prepare.json`
```json
{
    "dir_data_report": "data/reports/TuTu_sm",
    "dir_data_json": "data/json/TuTu_sm",
    "dir_data_embedding": "data/embeddings/TuTu_sm",
    "dir_data_graph": "data/graphs/TuTu_sm",
    "dir_data_graphviz": "data/graphviz/TuTu_sm",
    "dir_data_networkx": "data/nx/TuTu_sm",
    "dir_data_pickle": "data/pickle/TuTu_sm",

    "mapping_labels": {"benign": 0, "malware": 1},

    "do_draw": true,

    "split_train_test": false,
    "train_ratio": 0,
    "train_list_file": "data/TuTu_sm_train_list.txt",
    "test_list_file": "data/TuTu_sm_test_list.txt",

    "process_from": "report",
    
    "train_embedder": true,
    "edge_embedder": "tfidf",
    "node_embedder": "tfidf",
    "preprocess_level": "word",

    "max_ft": 100000,
    "top_k": 3,

    "vector_size": 10,
    "dm": 0
}
```
- `dir_data_report`: path to directory that contains cuckoo reports, divided into N sub-folders (N = number of classes/labels). eg, `dir_data_report` has 2 folders: `benign`, `malware`
- `dir_data_json`: path to directory that contains json file of processed behaviors for each report (some reports might have empty behaviors (no api calls), perhaps that binary needs user interactions to execute, therefore, that file did not execute when putting into cuckoo environment => fails to generate api calls)
- `dir_data_embedding`: for node/edge embedding
- `dir_data_graph`: store graph generated from each report individually (dgl graph)
- `dir_data_graphviz`: store .dot (and .svg) files for visualization (graphviz)
- `dir_data_networkx`: store .dot (and .svg) files for visualization (networkx)
- `dir_data_pickle`: path to directory to store all stuff needed for training (final output of `PrepareData`)

- `mapping_labels`: a dict that maps each label to an int64 code (label is the same as subsfolder names under each `dir_data...`, int64 code is used for training)
- `do_draw`: if `true`, generate graphviz and networkx graph for visualization and save to `dir_data_graphviz` and `dir_data_networkx`
- `split_train_test`: `true` if we want the code to split the set into `train` and `test` set. `false` if we've already had a list of files used for train and test.  
**NOTE 1:** Even if `split_train_test = false`, these two files can still be overwritten (filepaths listed in these two files which does not have api calls (cannot construct graph) will be removed from the list)  
**NOTE 2:** `test` set here is used only for testing inference. During training, the `train` set here will be further divided into `train`, `val` set (refer to `app.py`)
- `train_ratio`: used only if `split_train_test = true` (must `> 0` if used)
- `train_list_file`: path to save list of files we want to use for training. (If `split_train_test = false`, file must exist and not empty)
- `test_list_file`: path to save list of files we want to use for testing. (If `split_train_test = false`, file must exist and not empty)

- `process_from`: accept 3 values:
    - `report`: 
        1. Read inputs from `dir_data_report`. 
        2. Process behaviors and save processed behaviors to `dir_data_json`. 
        3. (optional) Save graphviz/networkx graph to `dir_data_graphviz` and `dir_data_networkx`. 
        4. Run node/edge embedding. 
        5. Create dgl graph and save to `dir_data_graph`. 
        6. Pack all necessary stuff and save to `dir_data_pickle`
    - `json`: 
        1. Read inputs from `dir_data_json`.
        2. Perform  `c -> f`  like above (`report` option)
    - `graph`: 
        1. Read inputs from `dir_data_graph`.
        2. Pack all necessary stuff and save to `dir_data_pickle`

- `train_embedder`: train node/edge embedder. Can `= true` only if `process_from = 'report'`
- `node_embedder`, `edge_embedder`: accept `tfidf` or `doc2vec`
- `max_ft`, `top_k`: params for `tfidf` embedder
- `vector_size`, `dm`: params for `doc2vec` embedder


## Train
```bash
python app.py
```

#### Modify `config.model.json`
```json
{
    "dir_data_pickle": "data/pickle/TuTu_sm",
    "mapping_labels": {"benign": 0, "malware": 1},

    "model_config": {
        "layer_type": "edGNNLayer",
        "layer_params": {
            "n_units": [
                8,
                8
            ],
            "activation": [
                "relu",
                "relu",
                "relu"
            ]
        }
    },
    "learning_config": {
        "lr": 0.001,
        "epochs": 100,
        "weight_decay": 0.001,
        "batch_size": 16,
        "gpu": -1
    },
    "early_stopping": {
        "patience": 100
    }
}
```
