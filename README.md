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

## Data Preparation and Preprocessing
For patent pair modeling, two datasets are inspected in this project. We suggest using the Patent Similarity Search dataset () to reproduce all experiments, as it shows noticeable changes in our evaluation metric of choice and therefore more suitable for ablation studies. The other dataset is the Patch Match dataset, which is considered a more challenging dataset and the performance might not show significant difference with our system.

### 1. Process Dataset
- `./scripts/clean_T_REx.py`: clean [raw T-Rex dataset](https://hadyelsahar.github.io/t-rex/downloads/) (32G), and save the cleaned T-Rex to JSON format
- `./scripts/create_subdataset-relation-classification.ipynb`: create the dataset from T-REx for pre-training factual adapter on relation classification task. This sub-dataset can be found [here](https://drive.google.com/drive/folders/1xRGmIUXwPrtnsksQ1GY8YAE87gf7Ct6E?usp=sharing).
- `refer to this` [code](https://github.com/windweller/DisExtract/tree/master/preprocessing) to get the dependency parsing dataset : create the dataset from Book Corpus for pre-training the linguistic adapter on dependency parsing task.

### 2. Factual Adapter
To pre-train fac-adapter, run
```bash
bash run_pretrain_fac-adapter.sh
```
### 3. Linguistic Adapter
To pre-train lin-adapter, run
```bash
bash run_pretrain_lin-adapter.sh
```
The pre-trained fac-adapter and lin-adapter models can be found [here](https://drive.google.com/drive/folders/12mfLpYq4BTwdbLZnQsdwDJKubM9aOr9f?usp=sharing).

## Fine-tuning on Downstream Tasks
Adapter Structure
- The fac-adapter (lin-adapter) consists of two transformer layers (L=2, H=768, A = 12)
- The RoBERTa layers where adapters plug in: 0,11,23 or 0,11,22
- For using only single adapter
    - Use the concatenation of the last hidden feature of RoBERTa and the last hidden feature of the adapter as the input representation for the task-specific layer.
- For using combine adapter
    - For each adapter, first concat the last hidden feature of RoBERTa and the last hidden feature of every adapter and feed into a linear layer separately, then concat the representations as input for task-specific layer.

About how to load pretrained RoBERTa and pretrained adapter
- The pre-trained adapters are in `./pretrained_models/fac-adapter/pytorch_model.bin` and `./pretrained_models/lin-adapter/pytorch_model.bin`.
    For using only single adapter, for example, fac-adapter, then you can set the argument `meta_fac_adaptermodel=<the path of factual adapter model>` and set `meta_lin_adaptermodel=””`.
    For using both adapters, just set the arguments `meta_fac_adaptermodel` and `meta_lin_adaptermodel` as the path of adapters.
- The pretrained RoBERTa will be downloaded automaticly when you run the pipeline.

### 1. Entity Typing
#### 1.1 OpenEntity
One single 16G P100

**(1) run the pipeline**
```bash
bash run_finetune_openentity_adapter.sh
```
**(2) result**
- with fac-adapter
    dev: (0.7967123287671233, 0.7580813347236705, 0.7769169115682607)
    test: (0.7929708951125755, 0.7584033613445378, 0.7753020134228187)
- with lin-adapter
    dev: (0.8071672354948806, 0.7398331595411888, 0.7720348204570185)
    test:(0.8001135718341851, 0.7400210084033614, 0.7688949522510232)
- with fac-adapter + lin-adapter
    dev: (0.8001101321585903, 0.7575599582898853, 0.7782538832351366)
    test: (0.7899568034557235, 0.7627737226277372, 0.7761273209549072)

the results may vary when running on different machines, but should not differ too much.
I just search results from per_gpu_train_batch_sizeh: [4, 8] lr: [1e-5, 5e-6], warmup[0,200,500,1000,1200], maybe you can change other parameters and see the results.
For w/fac-adapter, the best performance is achieved at gpu_num=1, per_gpu_train_batch_size=4, lr=5e-6, warmup=500(it takes about 2 hours to get the best result running on singe 16G P100)
For w/lin-adapter, the best performance is achieved at gpu_num=1, per_gpu_train_batch_size=4, lr=5e-6, warmup=1000(it takes about 2 hours to get the best result running on singe 16G P100)

**(3) Data format**

Add special token "@" before and after a certain entity, then the first @ is adopted to perform classification.
9 entity categories: ['entity', 'location', 'time', 'organization', 'object', 'event', 'place', 'person', 'group'], each entity can be classified to several of them or none of them. The output is represented as [0,1,1,0,1,0,0,0,0], 0 represents the entity does not belong to the type, while 1 belongs to.

#### 1.2 FIGER
**(1) run the pipeline**
```bash
bash run_finetune_figer_adapter.sh
```
The detailed hyperparamerters are listed in the running script.

### 2. Relation Classification
4*16G P100

**(1) run the pipeline**
```bash
bash run_finetune_tacred_adapter.sh
```
**(2) result**
- with fac-adapter
    - 'dev': (0.6686945083853996, 0.7481604120676968, 0.7061989928807085)
    - 'test': (0.693900391717963, 0.7458646616541353, 0.7189447746050153)
- with lin-adapter
    - 'dev': (0.6679165308118683, 0.7536791758646063, 0.7082108902333621),
    - 'test': (0.6884615384615385, 0.7536842105263157, 0.7195979899497488)
- with fac-adapter + lin-adapter
    - 'dev': (0.6793893129770993, 0.7367549668874173, 0.7069102462271645)
    - 'test': (0.7014245014245014, 0.7404511278195489, 0.7204096561814192)

- the results may vary when running on different machines, but should not differ too much.
- I just search results from per_gpu_train_batch_sizeh: [4, 8] lr: [1e-5, 5e-6], warmup[0,200,1000,1200], maybe you can change other parameters and see the results.
- The best performance is achieved at gpu_num=4, per_gpu_train_batch_size=8, lr=1e-5, warmup=200 (it takes about 7 hours to get the best result running on 4 16G P100)
- The detailed hyperparamerters are listed in the running script.

**(3) Data format**

Add special token "@" before and after the first entity, add '#' before and after the second entity. Then the representations of  @ and # are concatenated to perform relation classification.


### 3. Question Answering
#### 3.1 CosmosQA
One single 16G P100

**(1) run the pipeline**
```bash
bash run_finetune_cosmosqa_adapter.sh
```

**(2) result**

CosmosQA dev accuracy: 80.9
CosmosQA test accuracy: 81.8

The best performance is achieved at gpu_num=1, per_gpu_train_batch_size=64, GRADIENT_ACC=32, lr=1e-5, warmup=0 (it takes about 8 hours to get the best result running on singe 16G P100)
The detailed hyperparamerters are listed in the running script.

**(3) Data format**

For each answer, the input is `<s>context</s></s>question</s></s>answer</s>`, and will get a score for this answers. After getting four scores, we will select the answer with the highest score.

#### 3.2 SearchQA and Quasar-T 
The source codes for fine-tuning on SearchQA and Quasar-T dataset are modified based on the [code](https://github.com/thunlp/OpenQA) of paper "Denoising Distantly Supervised Open-Domain Question Answering".

## Use K-Adapter just like RoBERTa 
- You can use K-Adapter (RoBERTa with adapters) just like RoBERTa, which almost have the same inputs and outputs. Specifically, we add a class `RobertawithAdapter` in `pytorch_transformers/my_modeling_roberta.py`.
- A demo code `[run_example.sh and examples/run_example.py]` about how to use “RobertawithAdapter”, do inference, save model and load model. You can leave the arguments of adapters as default.
- Now it is very easy to use Roberta with adapters. If you only want to use single adapter, for example, fac-adapter, then you can set the argument `meta_fac_adaptermodel='./pretrained_models/fac-adapter/pytorch_model.bin''` and set `meta_lin_adaptermodel=””`. If you want to use both adapters, just set the arguments `meta_fac_adaptermodel` and `meta_lin_adaptermodel` as the path of adapters.
```bash
bash run_example.sh
```
## TODO
- Remove and merge redundant codes
- Support other pre-trained models, such as BERT...

