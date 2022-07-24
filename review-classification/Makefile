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