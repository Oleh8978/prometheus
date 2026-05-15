.PHONY: setup run run-adk help

help:
	@echo "Targets:"
	@echo "  make setup   - uv sync + remind to copy .env"
	@echo "  make run     - one-shot traced run (MESSAGE=...)"
	@echo "  make run-adk - ADK CLI dev loop (cd agent && adk run shopping_demo)"

setup:
	uv sync
	@test -f .env || echo "Tip: copy .env.example to .env and add keys."

run:
	cd agent && uv run python main.py "$(if $(MESSAGE),$(MESSAGE),Help me find a floral summer dress and buy size M.)"

run-adk:
	cd agent && uv run adk run shopping_demo
