# KAPLE: a Knowledgeable Aadapter-based model for Pre-trained Language Embedding
This repository contains the codes and data of a final year project titled "Enhance graph-based document retrieval with external knowledge". The research was conducted in the School of Electrical and Electronic Engineering at Nanyang Technological University under the supervision of [A/P Chen Li-Hui](https://dr.ntu.edu.sg/cris/rp/rp00969).

In this study, we start from a recent paper "K-Adapter: Infusing Knowledge into Pre-Trained Models with Adapters", that supports continual knowledge infusion into language models. Novel adjustments such as joint optimization, contrastive learning and knowledge distillation are combined to further improve the model's performance on sentence-pair modeling tasks (e.g. NLI). We extend our work to patent document retrieval, and show that our new adaptation of K-Adapter work well on retrieval tasks.

## Software Environment
- Python 3.9.13
- Conda 22.9.0
- Cuda 11.2
- Git 2.17.1
- GPU version: GeForce RTX 3090

## Required Packages
- info-nce-pytorch 0.1.4
- scikit-learn 0.24.2
- huggingface-hub 0.4.0
- transformers 4.18.0
- sentencepiece 0.1.97
- sqlalchemy 1.3.13
- sqlite 1.30.2
- torchvision 0.9.0+cu111
- torch 1.3.1
- pandas 1.1.5
- numpy 1.16.2
- faiss 1.7.0
- tensorflow 2.6.2

The environment can be installed via conda:
```bash
conda create -n kaple python=3.9
```
```bash
conda install -r requirements.txt
```
## Dataset Information
For patent pair modeling, two datasets are inspected in this project. We suggest using the Patent Similarity Search dataset to reproduce all experiments, as it shows noticeable changes in our evaluation metric of choice and therefore more suitable for ablation studies. The other dataset is the Patch Match dataset, which is considered a more challenging dataset and the performance might not show significant difference using our system.

- The PatentSim dataset is downloadable through [here](https://figshare.com/articles/corpus_tar_gz/7257194). For few-shot learning, refer to `util_data.py` under `data/patent-sim` to construct a more compact subset.
- The PatentMatch dataset is downloadable through [here](https://hpi.de/naumann/projects/web-science/paar-patent-analysis-and-retrieval/patentmatch.html). We select the ultra-balanced version where the number of novelty-destroying and non novelty-destroying patent pairs and balanced.

## Patent Pair Modeling
...
### 1. Joint Optimization of Multiple Loss Functions
To conduct the experiment, run `examples/kaple.py` through a bash script like `run_pretrain_fac-adapter.sh` and execute as such:
```bash
bash run_pretrain_fac-adapter.sh
```
Try changing the arguments and inspect how they result in model performance. See customisable arguments in the command line via
```bash
python examples/kaple.py -h
```
### 2. Self-Distillation of Encoders
To conduct the experiment, run `examples/self-distillation.py` through a bash script like `run_pretrain_fac-adapter.sh` and execute as such:
```bash
bash run_pretrain_fac-adapter.sh
```
Similarly, customizable arguments are available via
```bash
python examples/self-distillation.py -h
```

## Patent Prior Art Retrieval
...

**(1) run the pipeline**
```bash
bash run_finetune_cosmosqa_adapter.sh
```

**(2) result**



**(3) Data format**


