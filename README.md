# ClasReg: A Classification-Regression framework

# 1. Introduction
A unique multilayer framework simultaneously capable of Link Prediction and Breakup/Rift Prediction functionalities.

Basically, ClasReg is a hybrid (Deep Learning and Heuristic methodologies) model whose framework essentially comprises the following layers, viz:
- a Preprocessing layer;
- a Representation/Feature Learning layer;
- a Classification layer;
- a Regression layer; and
- an Inference/Heuristic engine.

ClasReg, as a scalable and bifunctional framework, exploits its multilayer framework for simultaneously resolving the problems of Breakup/Rift Prediction as well as Link Prediction in social network structures.

# 2. Overview of ClasReg's (Repository) Directory Structure
| Directory/File | GitHub URL/Link | Description |
| -------------- | --------------- | ----------- |
| custom_classes | [visit URL](https://github.com/bhevencious/ClasReg/tree/master/custom_classes) | Subdirectory containing ClasReg's dependencies (or class files). |
| generic_datasets | [visit URL](https://github.com/bhevencious/ClasReg/tree/master/generic_datasets) | Subdirectory containing the datasets employed in evaluating the performance (via experiments) of ClasReg. Within this subdirectory, each dataset is housed in a directory which bears the same name as the edgelist (*.edgelist*) file of the dataset. Each edgelist file and its corresponding dataset folder MUST bear the same name. The *.edgelist* file is a prerequisite for every input dataset; all other files are generated by ClasReg during its evaluation on the dataset. |
| breakup_and_link_prediction.py | [visit URL](https://github.com/bhevencious/ClasReg/blob/master/breakup_and_link_prediction.py) | Primary source code (or implementation) of ClasReg. **entry_point()** is the *entry point* function of this source file. |
| eval_log.txt | [visit URL](https://github.com/bhevencious/ClasReg/blob/master/eval_log.txt) | Log file, which records the details and performance reports of ClasReg, with respect to Link Prediction task on each dataset. |

# 3. Baselines (or Benchmark Models)
The benchmark models (or baselines) evaluated against ClasReg herein include, viz:
| S/N | Baseline | Acronym | Description |
| --- | -------- | ------- | ----------- |
| 1. | Adamic Adar Index | Adamic | Strength-of-Ties methodology |
| 2. | Common Neighbor Index | CommNeigh | Strength-of-Ties methodology |
| 3. | Jaccard Coefficient | Jaccard | Strength-of-Ties methodology |
| 4. | Katz Index | Katz | Strength-of-Ties methodology |
| 5. | Preferential Attachment Index | PrefAttach | Strength-of-Ties methodology |
| 6. | Resource Allocation Index | ResAlloc | Strength-of-Ties methodology |
| 7. | DeepWalk | DeepWalk | Random-Walk Graph Embedding approach |
| 8. | Node2vec | Node2vec | Random-Walk Graph Embedding approach |
| 9. | Struc2vec | Struc2vec | Random-Walk Graph Embedding approach |
| 10. | Graph Factorization | GraFac | Matrix-Factorization Graph Embedding approach |
| 11. | Graph Representations | GraRep | Matrix-Factorization Graph Embedding approach |
| 12. | High-Order Proximity preserved Embedding | HOPE | Matrix-Factorization Graph Embedding approach |
| 13. | Laplacian Eigenmap | LapEigen | Matrix-Factorization Graph Embedding approach |
| 14. | Singular Value Decomposition | SVD | Matrix-Factorization Graph Embedding approach |
| 15. | Graph Auto-Encoders | GAE | Neural-Network Graph Embedding approach |
| 16. | Large-scale Information Network Embedding | LINE | Neural-Network Graph Embedding approach |
| 17. | Structural Deep Network Embedding | SDNE | Neural-Network Graph Embedding approach |

# 4. Benchmark Datasets
The benchmark datasets employed herein for experiments and analyses include, viz:
| Dataset | Nodes/Actors | Edges/Ties |
| ------- | ------------ | ---------- |
| CiteSeer | 3,312 | 4,732 |
| Cora | 2,708 | 5,429 |
| Internet-Industry Partnerships | 219 | 631 |
| PubMed-Diabetes | 19,717 | 44,338 |
| Terrorists-Relation | 851 | 8,592  |
| Zachary-Karate | 34 | 78 |

# 5. Code Installation and Usage
The implementation for the baselines (or benchmark models) based on Strength-of-Ties methodology (*Adamic, CommNeigh, Jaccard, Katz, PrefAttach, and ResAlloc*) is availablve via: [EvalNE Library](https://github.com/bhevencious/EvalNE).

The implementation for the baselines (or benchmark models) based on Graph-Embedding approach (*DeepWalk, GAE, GraFac, GraRep, HOPE, LapEigen, LINE, Node2vec, SDNE, Struc2vec, and SVD)*) is available via: [BioNEV Library](https://github.com/bhevencious/BioNEV).

The implementation for our proposed hybrid framework, **ClasReg**, is available for direct installation (from GitHub) via any of the following options, viz:
## 5.1.1. Direct Installation via pip
```
# Python 2
$ pip install git+https://github.com/bhevencious/ClasReg.git

# Python 3
$ pip3 install git+https://github.com/bhevencious/ClasReg.git
```

## 5.1.2. Installation via repository cloning
```
$ git clone https://github.com/bhevencious/ClasReg.git
$ cd ClasReg
$ pip install -e .
```

## 5.1.3. Direct download of repository
```
[Click here to initiate download](https://github.com/bhevencious/ClasReg/archive/master.zip)
```

## 5.2.1. ClasReg's run parameters/options
| Parameter | Data Type | Default  | Description |
| --------- | ------- | --------- | ----------- |
| -rp *or* --root_path | string | 'generic_datasets/' | Base/Root path for ClasReg's operations. |    
| -el or --edge_list | string | 'CiteSeer' | Name of the folder containing the input-dataset edgelist. NB: The *.edgelist* file MUST bear the same name as its host folder/directory. | 
| -rm or --run_mode | string | 'single' | This defines how ClasReg will evaluate the datasets. ClasReg can either be run per ('single') specified dataset OR cumulatively ('all') for all intrinsic datasets. |
