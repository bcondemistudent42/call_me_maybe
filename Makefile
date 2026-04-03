PYTHON = uv run python3
MAIN = main.py
SRC = .
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
	$(PYTHON) $(MAIN)

# lint: install to finish
# 	@echo "Lint mode:"
# 	uv run flake8 $(SRC) --exclude=.venv --exclude=llm_sdk
# 	uv run mypy $(SRC)

clean:
	@echo "Cleaning everything"
	rm -rf *_pyc_*
	rm -rf output
	rm -rf .mypy_cache
	rm -rf .venv
	rm -rf llm_sdk/*pyc*
	rm -rf uv.lock
	rm -rf llm_sdk/uv.lock
	uv clean