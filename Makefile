.DEFAULT_GOAL := help
.PHONY: requirements

help: ## display this help message
	@echo "Run \`make <target>\` where <target> is one of"
	@perl -nle'print $& if m{^[\.a-zA-Z_-]+:.*?## .*$$}' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m  %-25s\033[0m %s\n", $$1, $$2}'

lint: ## run flake8
	flake8 nn.py

requirements: ## install requirements
	pip install -r requirements.txt

run: ## run tf and drop into interactive mode
	python -i nn.py
