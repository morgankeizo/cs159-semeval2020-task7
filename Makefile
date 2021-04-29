.PHONY: all clean deps

data_name := semeval-2020-task-7-dataset
data_url := https://cs.rochester.edu/u/nhossain/$(data_name).zip

all:
	@echo "usage: clean deps data"

clean:
	rm -r venv bert-cache

deps: venv
	source venv/bin/activate && \
	pip install torch pandas transformers psutil matplotlib

venv:
	python3 -m venv venv

data:
	curl -O $(data_url)
	unzip $(data_name).zip
	mv $(data_name) data
	rm $(data_name).zip
