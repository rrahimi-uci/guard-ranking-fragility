# Verification entry point for starting_type_adaptation_v1.
# Delegates to the command declared in studies/registry.yaml rather than
# restating it, so a package can never disagree with the registry.
#
# PY is overridable and defaults to python3: `python` is absent on a stock
# macOS PATH, and the repo venv is .venv/bin/python.
PY ?= python3

.PHONY: verify command

verify:
	@cd ../.. && $(PY) tools/study_verify.py starting_type_adaptation_v1

command:  ## print the declared command without running it
	@cd ../.. && $(PY) tools/study_verify.py starting_type_adaptation_v1 --print
