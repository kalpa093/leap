# Is It Worth It? An Empirical Study of Data Augmentation for Sentiment Analysis in Software Engineering

This repository is an implementation of the paper "Is It Worth It? An Empirical Study of Data Augmentation for Sentiment Analysis in Software Engineering" submitted to the ICSE 2027.
We release our artifacts of Data Augmentation (Generation) and Fine-tuning including our visualizations to facilitate further research and adoption.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Data Augmentation & Model Training](#data-augmentation--model-training)
- [Visualization](#visualization)
- [Detailed Results](#detailed-results)


---

## Prerequisites

### Required Libraries

We have two environments, fine-tuning SLMs and LLMs with QLoRA.

#### SLMs
```
pip install -r requirements.txt
```

#### LLMs

```
conda create -n qlora python=3.10.18
conda activate qlora
conda install -r requirements_qlora.txt
```

### Supported Models

- **DeBERTa**: microsoft/deberta-v3-large
- **XLNet**: xlnet-large-cased
- **T5**: google/flan-t5-base
- **CodeGen2**: Salesforce/codegen2-7b
- **Phi-2**: microsoft/phi-2
- **DeepSeek**: deepseek-ai/deepseek-llm-7b-base

### Supported Datasets

- **APP**: 3-class app review classification
- **StackOverFlow**: 3-class Q&A  emotional polarity classification
- **GitHub**: 3-class issues' emotional polarity classification
- **Jira**: 3-class issues' emotional polarity classification
- **Gerrit**: Binary code review polarity classification
- **Tweets_F**: 5-class multi-label sentiment classification
- **Tweets_P**: 4-class multi-label sentiment classification
- **Tweets_n**: 4-class multi-label sentiment classification

### Supported Data Augmentation Techniques
We implemented each data augmentation approaches according to their open source codes in GitHub repositories.

These are studies we have studied.
- SSMBA (Self-Supervised Manifold Based Augmentation) - [SSMBA: Self-Supervised Manifold Based Data Augmentation for Improving Out-of-Domain Robustness](https://aclanthology.org/2020.emnlp-main.97/)
- AEDA (An Easier Data Augmentation) - [AEDA: An Easier Data Augmentation Technique for Text Classification](https://aclanthology.org/2021.findings-emnlp.234/)
- C2L (Causally Contrastive Learning) - [C2L: Causally Contrastive Learning for Robust Text Classification](https://ojs.aaai.org/index.php/AAAI/article/view/21296)
- TextSmoothing - [Text Smoothing: Enhance Various Data Augmentation Methods on Text Classification Tasks](https://aclanthology.org/2022.acl-short.97/)

---

## Data Augmentation & Model Training

```
bash run.sh
```

The ```run.sh``` is a script for all experiment including augmenation, fine-tuning, prompting, and evaluation.
```linux
# run.sh

echo "Script for SLM is Running"

datasets=("app" "so" "github" "jira" "gerrit" "tweets" " tweets_n" "tweets_p")
da=("none" "aeda" "ssmba" "c2l" "ts")
models=("deberta" "xlnet" "t5")

for dataset in "${datasets[@]}"; do
	for da in "${da[@]}"; do
		for model in "${models[@]}"; do
			echo "Running SLM with dataset=$dataset, da=$da, model=$model"
				CUDA_VISIBLE_DEVICES=0,1,2 python train.py -dataset "$dataset" -da "$da" -model "$model"
		done
	done
done

echo "Script for LLM is Running"

datasets=("app" "so" "github" "jira" "gerrit" "tweets" " tweets_n" "tweets_p")
da=("none" "aeda" "ssmba" "c2l" "ts")
models=("codegen" "phi" "deepseek")

for dataset in "${datasets[@]}"; do
        for da in "${da[@]}"; do
                for model in "${models[@]}"; do
                        echo "Running LLM with dataset=$dataset, da=$da, model=$model"
                                CUDA_VISIBLE_DEVICES=0,1,2 python llm.py -dataset "$dataset" -da "$da" -model "$model"
                done
        done
done
```

**Output:**
The augmented sets by each data augmentation are saved inside ```\temp``` folder with format, ```{dataset}_{da}.csv``` that you can check how training set is augmented.<br />
Each checkpoints of fine-tuned models are also saved inside ```\temp``` folder with format, ```train_{dataset}_{da}_{model}```.<br />



### If you want to evaluate with a specific case, follow the script below.<br />
```
# SLM
CUDA_VISIBLE_DEVICES=0,1,2 python train.py -dataset "dataset" -da "da" -model "model"

# LLM Fine-tuning
CUDA_VISIBLE_DEVICES=0,1,2 python llm.py -dataset "dataset" -da "da" -model "model"
```
For arguments, you can put as below.<br />

dataset: ```app```, ```so```, ```github```, ```jira```, ```gerrit```, ```tweets```, ```tweets_p```, and ```tweets_n```<br />
da: ```none```, ```ssmba```, ```aeda```, ```c2l``` and ```ts```<br />
model: <br />
```deberta```, ```xlnet```, and ```t5``` for SLM<br />
```codegen```, ```phi```, and ```deepseek``` for LLM<br />

---

## Visualization
If you want to check our visualizations of Figure 5, 10a, and 10b from our paper, you can confirm the figures by running these codes.


```
# Loss Landscape to identify Proximal-Support Augmentation Condition
CUDA_VISIBLE_DEVICES=0,1,2 python visualize_loss_psa.py -dataset github -da none -model xlnet

# Token Attribution by Shapley value
CUDA_VISIBLE_DEVICES=0,1,2 python visualize_shapley.py -dataset github -da none -model xlnet
CUDA_VISIBLE_DEVICES=0,1,2 python visualize_shapley.py -dataset github -da none -model phi2
```

---

## Detailed Results

A detailed results for our experiments are presented in our document pages in - [Detailed Results](docs/index.md).
