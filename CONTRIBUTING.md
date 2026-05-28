# Contributing to SHERLOC

Thanks for your interest in contributing! This project is research software, so
contributions can range from bug reports and documentation fixes to new
encoders, evaluation scripts, or experiments on additional cohorts.

## Reporting issues

Before opening an issue, please search existing issues to avoid duplicates.
When filing a new one, include:

- What you ran (command, config file).
- What you expected to happen.
- What actually happened (stack trace, log excerpt).
- Your environment: Python version, OS, `pip freeze` output if possible.

## Development setup

Clone the repository and install in editable mode with the `dev` extras:

```bash
git clone https://github.com/AaronMamannToledano/SHERLOC.git
cd SHERLOC
python -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate
pip install -e ".[dev]"
pre-commit install                  # optional: run linters on every commit
```

## Code style

- Formatting: [`black`](https://black.readthedocs.io/) (line length 100).
- Linting: [`ruff`](https://docs.astral.sh/ruff/) with the rules configured in `pyproject.toml`.
- Imports sorted by `ruff` (`I` ruleset).

Run everything before pushing:

```bash
ruff check .
black --check .
pytest
```

## Tests

Tests live under `tests/`. New model components or data utilities should ship
with at least a smoke test that verifies shapes/dtypes on a small synthetic
input — see `tests/fixtures/` for the toy dataset used by CI.

## Pull requests

1. Fork the repo and create a topic branch off `main`
   (`git checkout -b feat/short-description`).
2. Make focused commits with clear messages. We loosely follow
   [Conventional Commits](https://www.conventionalcommits.org/) (`feat:`,
   `fix:`, `docs:`, `refactor:`, `test:`, `chore:`).
3. Update the README or docstrings if your change affects user-facing behavior.
4. Make sure CI is green.
5. Open a PR describing **what** changed and **why**, and link any related
   issue.

## Reproducibility expectations

Because SHERLOC is benchmarked against published results, please:

- Avoid silently changing default hyperparameters in `config.yaml`.
- If you tweak the training loop, re-run the cross-validated evaluation and
  report the new numbers in your PR description.
- Pin any new dependency in both `requirements.txt` and `pyproject.toml`.

## License

By contributing, you agree that your contributions will be licensed under the
project's [MIT License](LICENSE).
