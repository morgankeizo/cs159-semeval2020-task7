.PHONY: all clean deps

all:
	@echo: "usage: clean deps"

clean:
	rm -r venv bert-cache

deps: venv
	source venv/bin/activate && \
	pip install torch pandas transformers

venv:
	python3 -m venv venv
