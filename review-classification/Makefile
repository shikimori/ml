# HELP
# This will output the help for each task
# thanks to https://marmelab.com/blog/2016/02/29/auto-documented-makefile.html
.PHONY: help

mkfile_path := $(abspath $(lastword $(MAKEFILE_LIST)))
current_dir := $(patsubst %/,%,$(dir $(mkfile_path)))

include .env
export $(shell sed 's/=.*//' .env)

help: ## This help
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

.DEFAULT_GOAL := help

prepare_input: ## Read input.json, split by sentences and slice to train/test sets
	python3 "$(current_dir)/prepare_input.py" \
		--input "$(current_dir)/demo/input.json" \
		-s 43 \
		-ss 0.8 \
		-v 52 \
		--train "$(current_dir)/out/train.json" \
		--test "$(current_dir)/out/test.json" \
		--sentences "$(current_dir)/out/sentences.json" \
		--labels "$(current_dir)/out/labels.json"

train_sentences: ## Train sentences classifier
	python3 "$(current_dir)/train_sentences_classifier.py" \
		-s 43 \
		-v 52 \
		--vocab 1000000 \
		--train "$(current_dir)/out/train.json" \
		--test "$(current_dir)/out/test.json" \
		--labels "$(current_dir)/out/labels.json" \
		--model "$(current_dir)/out/model" \
		--history "$(current_dir)/out/history.bin"

predict_sentences: ## Predict sentences using classifier
	python3 "$(current_dir)/predict_sentences.py" \
		--sentences "$(current_dir)/out/sentences.json" \
		--model "$(current_dir)/out/model" \
		--labels "$(current_dir)/out/labels.json" \
		-s 43 \
		-v 52 \
		--predictions "$(current_dir)/out/predictions.json" \
		--cctrain "$(current_dir)/out/cctrain.json"

train_cc: ## Train Catboost Classifier
	python3 "$(current_dir)/train_gradient.py" \
		--train "$(current_dir)/out/cctrain.json" \
		-s 43 \
		-v 52 \
		-ss 0.8 \
		--model "$(current_dir)/out/cbmodel.bin"

predict: ## Predict prod data
	python3 "$(current_dir)/predict.py" \
		--prod "$(current_dir)/demo/prod.json" \
		--labels "$(current_dir)/out/labels.json" \
		--cbmodel "$(current_dir)/out/cbmodel.bin" \
		--model "$(current_dir)/out/model" \
		-s 43 \
		-v 52 \
		--result "$(current_dir)/demo/result.json"

pipeline: ## Run everything in sequence
	rm -rf "$(current_dir)/out"
	mkdir -p "$(current_dir)/out"
	make prepare_input
	make train_sentences
	make predict_sentences
	make train_cc
	make predict