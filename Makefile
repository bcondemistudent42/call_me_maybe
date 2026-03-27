PYTHON = uv run python3
MAIN = main.py

install: .venv/uv.lock

.venv/uv.lock: pyproject.toml
	@echo "Installing dependencies using uv..."
	uv sync
	@touch .venv/uv.lock

run: install
	@echo "Running $(MAIN)"
	$(PYTHON) $(MAIN)