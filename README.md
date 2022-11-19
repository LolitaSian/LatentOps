# LatentOps [WIP]
Source code of paper: ***Composable Text Controls in Latent Space with ODEs***

*https://arxiv.org/abs/2208.00638*


***Code is coming soon...***

## Preparation
### Recommended Environment
We recommend to create a new conda enviroment (named *latentops*) by:
```shell
conda create -n latentops python==3.9.1 pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
```
Then activate *latentops* and install the required packages by running:
```shell
conda activate latentops
bash build_envs.sh
```

### Prepare Datasets
Download and process the datasets by running the script:
```shell
bash download_datasets.sh
```

### Pretrained Models
Download and process the pretrained model by running the script:
```shell
bash download_pretrained_models.sh
```
    
### Prepare Classifiers
Download and process the external classifiers by running the script:
 ```shell
 bash download_classifiers.sh
 ```
## Conditional Generation
You can do conditional generation (default Yelp) by running:
```shell
cd code
bash conditional_generation.sh $1 $2
```
$1 represents operators (1 for sentiment, 4 for tense, 33 for formality).
$2 represents desired labels:
- sentiment: 0-negative, 1-positive
- tense: 0-past, 1-present, 2-future
- formality: 0-informal, 1-formal

For examples, you can run:
```shell
# for positive sentences
bash conditional_generation.sh 1 1
# for past sentences
bash conditional_generation.sh 4 0
# for positive & future sentences
bash conditional_generation.sh '1,4' '1,2'
# for positive & future & informal
bash conditional_generation.sh '1,4,33' '1,2,0'
# for positive & future & informal and negative & future & informal
bash conditional_generation.sh '1,4,33' '1,2,0;0,2,0'
```
The generated files can be found in *../ckpts/model/sample/* (default: *../ckpts/large_yelp/sample/sampling\*.txt*)
## Outputs
To facilitate comparison, we provide the output files of text editing with single attribute (text style transfer) in [*./outputs*](/outputs) folder.

## Train Your Own Latent Model (VAE)
Modify the path of data file in *code/train_vae.sh*
```shell
dataset=your_dataset_name
# e.g., dataset=yelp
TRAIN_FILE=path_to_train_data_file 
# e.g., TRAIN_FILE=../data/datasets/yelp_data/train.shuf.merge
TEST_FILE=path_to_test_data_file
# e.g., TEST_FILE=../data/datasets/yelp_data/test.merge
```
The structure of the data file: one line one sentence. See *../data/datasets/yelp_data/test.merge* for example.

Then run the script to train a VAE
```shell
cd code
bash train_vae.sh
```
The checkpoints will be saved in *../ckpts/LM/$dataset/$name* by default. You also can find the tensorboard logs in *code/runs/$dataset*


## Cite
```
@article{liu2022composable,
    title={Composable Text Control Operations in Latent Space with Ordinary Differential Equations},
    author={Liu, Guangyi and Feng, Zeyu and Gao, Yuan and Yang, Zichao and Liang, Xiaodan and Bao, Junwei and He, Xiaodong and Cui, Shuguang and Li, Zhen and Hu, Zhiting},
    journal={arXiv preprint arXiv:2208.00638},
    year={2022}
}
```

