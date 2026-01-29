#!/bin/bash

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

echo "Script for LLM-FT is Running"

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
