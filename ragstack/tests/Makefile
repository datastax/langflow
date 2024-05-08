.PHONY: format lint integ_tests

integ_tests:
	poetry run pytest integration_tests --instafail $(args)

format:
	poetry run ruff check . --fix
	poetry run ruff format .

lint:
	poetry run mypy --namespace-packages -p "integration_tests"

codespell:
	@poetry install --with spelling
	poetry run codespell --toml pyproject.toml

fix_codespell:
	@poetry install --with spelling
	poetry run codespell --toml pyproject.toml --write


