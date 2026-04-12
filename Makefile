PYTHON = uv run python3
MAIN = main.py
SRC = src
install: .venv/uv.lock

.venv/uv.lock: pyproject.toml
	@echo "Installing dependencies using uv..."
	uv sync
	@touch .venv/uv.lock

debug: install
	@echo "Debug mode: "
	$(PYTHON) -m pdb $(MAIN) $(CONFIG)


run: install
	@echo "Running $(MAIN)"
	$(PYTHON) -m $(SRC)

lint:
	uv run flake8 $(SRC) --exclude=.venv,llm_sdk
	uv run mypy $(SRC) --warn-return-any \
		--warn-unused-ignores \
		--ignore-missing-imports \
		--disallow-untyped-defs \
		--check-untyped-defs \

clean:
	@echo "Cleaning everything"
	rm -rf *_pyc_*
	rm -rf output
	rm -rf .mypy_cache
	rm -rf src/__pycache__
	rm -rf llm_sdk/*pyc*
	rm -rf uv.lock
	rm -rf llm_sdk/uv.lock

fclean: clean
	rm -rf .venv
	uv clean