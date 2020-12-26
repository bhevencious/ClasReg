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

# 2. Overview of Directory Structure
| Directory/File | GitHub URL/Link | Description |
| -------------- | --------------- | ----------- |
| custom_classes | [visit URL](https://github.com/bhevencious/ClasReg/tree/master/custom_classes) | Subdirectory containing ClasReg's dependencies (or class files). |
| generic_datasets | [visit URL](https://github.com/bhevencious/ClasReg/tree/master/generic_datasets) | Subdirectory containing the datasets employed in evaluating the performance (via experiments) of ClasReg. Within this subdirectory, each dataset is housed in a directory which bears the same name as the edgelist (*.edgelist*) file of the dataset. Hence, each edgelist file and its corresponding dataset folder MUST bear the same name. |
| breakup_and_link_prediction.py | [visit URL](https://github.com/bhevencious/ClasReg/blob/master/breakup_and_link_prediction.py) | Primary source code (or implementation) of ClasReg. entry_point() is the *entry point* function of this source file. |
| eval_log.txt | [visit URL](https://github.com/bhevencious/ClasReg/blob/master/eval_log.txt) | Log file, which records the details and performance reports of ClasReg, with respect to Link Prediction task on each dataset. |

BioNEV: https://github.com/bhevencious/BioNEV
Library for the graph-embedding baselines (DeepWalk, GAE, GraFac, GraRep, HOPE,
LapEigen, LINE, Node2vec, SDNE, Struc2vec, and SVD)
6.
EvalNE: https://github.com/bhevencious/EvalNE
Library for the strength-of-ties baselines (Adamic, CommNeigh, Jaccard, Katz, PrefAttach,
and ResAlloc)
