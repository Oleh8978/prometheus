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

batch:
	cd agent && uv run python ../scripts/batch_run.py

batch-quick:
	cd agent && uv run python ../scripts/batch_run.py --quick

evals:
	cd agent && uv run python ../evals/run_evals.py

improve:
	cd agent && uv run python improvement_agent.py $(CYCLE)

pipeline:
	cd agent && uv run python ../scripts/pipeline.py $(CYCLE)

scores:
	uv run python scripts/score_summary.py
