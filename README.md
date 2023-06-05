# LatentOps [WIP]
Source code of paper: ***Composable Text Controls in Latent Space with ODEs***

*https://arxiv.org/abs/2208.00638*

## Preparation
### Recommended Environment
新建虚拟环境
```shell
conda create -n latent python==3.9.1 pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
```
激活虚拟环境，安装需要的包：
```shell
conda activate latent
bash build_envs.sh
# build_envs.sh修改过了
```

### Prepare Datasets

数据集处理：
```shell
bash download_datasets.sh
```

### Pretrained Models
下载处理预训练模型：

```shell
bash download_pretrained_models.sh
```

### Prepare Classifiers
外部分类器：
 ```shell
 bash download_classifiers.sh
 ```
## Conditional Generation
要进行条件生成（默认为Yelp），可以运行以下命令：
```shell
cd code
bash conditional_generation.sh $1 $2
```
`$1` represents operators (1 for sentiment, 4 for tense, 33 for formality).
`$2` represents desired labels:

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
生成的文件可以在 *../ckpts/model/sample/* 目录中找到（默认为 *../ckpts/large_yelp/sample/sampling.txt*）。

## Train VAE
要修改 *code/train_vae.sh* 脚本中的数据文件路径，请按照以下步骤进行操作：

```
shellCopy codedataset=你的数据集名称
# 例如，dataset=yelp
TRAIN_FILE=数据训练文件路径 
# 例如，TRAIN_FILE=../data/datasets/yelp_data/train.shuf.merge
TEST_FILE=数据测试文件路径
# 例如，TEST_FILE=../data/datasets/yelp_data/test.merge
```

数据文件的结构：每行一个句子。参考 *../data/datasets/yelp_data/test.merge* 示例文件。

然后运行脚本来训练一个 VAE：

```
shellCopy codecd code
bash train_vae.sh
```

检查点默认保存在 *../ckpts/LM/$dataset/$name* 中。你还可以在 *code/runs/$dataset* 目录中找到 TensorBoard 日志。

通过这些步骤，你可以在 *code/train_vae.sh* 脚本中修改数据文件的路径。请确保路径与你的数据文件的实际位置相匹配。

## Train GAN and Classifiers
在训练完VAE之后，你可以训练GAN和分类器来执行一些操作。
### Train GAN
你需要指定一些关键参数：

```
shellCopy codetrain_cls_gan='gan'

ckpt_path=VAE检查点的路径  # 例如，ckpt_path=../ckpts/base_yelp

TRAIN_FILE=训练GAN的数据文件路径 
# 例如，TRAIN_FILE=../data/datasets/yelp_data/train_gan.txt

TEST_FILE=测试GAN的数据文件路径 
# 例如，TEST_FILE=../data/datasets/yelp_data/test.txt
```

GAN的训练和测试数据文件应该采用以下行格式（不包括方括号 []）：[0]\t[文本]，其中 [0] 在训练中没有使用且没有意义，可以是任何其他整数。参考 *../data/datasets/yelp_data/train_gan.txt* 中的示例。

然后运行以下命令来训练GAN：

```
shellCopy codecd code
bash train_classifier_latent.sh
```

### Train Classifiers

你需要指定一些关键参数：

```
shellCopy codetrain_cls_gan='gan'

ckpt_path=VAE检查点的路径  # 例如，ckpt_path=../ckpts/base_yelp

TRAIN_FILE=训练GAN的数据文件路径 
# 例如，TRAIN_FILE=../data/datasets/yelp_data/train_gan.txt

TEST_FILE=测试GAN的数据文件路径 
# 例如，TEST_FILE=../data/datasets/yelp_data/test.txt
```

GAN的训练和测试数据文件应该采用以下行格式（不包括方括号 []）：[0]\t[文本]，其中 [0] 在训练中没有使用且没有意义，可以是任何其他整数。请参考 *../data/datasets/yelp_data/train_gan.txt* 中的示例。

然后运行以下命令来训练GAN：

```
shellCopy codecd code
bash train_classifier_latent.sh
```


## Outputs
为了方便比较，我们在 [*./outputs*](https://chat.openai.com/outputs) 文件夹中提供了进行单属性文本编辑（文本风格转换）的输出文件。

