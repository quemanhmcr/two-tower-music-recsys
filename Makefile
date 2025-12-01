.PHONY: setup download preprocess train clean

PYTHON = python
PIP = pip

setup:
	$(PIP) install -r requirements.txt

download:
	$(PYTHON) data/download.py

preprocess:
	$(PYTHON) data/preprocess.py

train:
	$(PYTHON) main.py

clean:
	rm -rf __pycache__
	rm -rf */__pycache__
	rm -rf */*/__pycache__
	rm -rf test_data_temp

test:
	$(PYTHON) tests/test_pipeline.py

lint:
	flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
	flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics

check: lint test
