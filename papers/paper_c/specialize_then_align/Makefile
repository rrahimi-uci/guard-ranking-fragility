PY ?= ../../../.venv/bin/python
export PYTHONPATH := $(CURDIR)/src
export PYTHONDONTWRITEBYTECODE := 1
export PYTHONPYCACHEPREFIX := $(CURDIR)/build/pycache

.PHONY: test validate grid readiness paper lock validate-lock

test:
	$(PY) -m pytest -q

validate:
	$(PY) -m paper_c_sta validate --config config/study.json

grid:
	$(PY) -m paper_c_sta grid --config config/study.json

readiness:
	$(PY) -m paper_c_sta readiness --config config/study.json

paper:
	./manuscript/build.sh

lock:
	$(PY) -m paper_c_sta lock-protocol

validate-lock:
	$(PY) -m paper_c_sta validate-lock
