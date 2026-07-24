# Contributing

Keep scientific behaviour, validation level, and signal units explicit. Do not
change historical experiment outputs without a documented bug fix.

Before opening a change, run:

```powershell
python -m build
pytest --cov=ssvep_toolkit
ruff check src tests
mypy src
```

Place new supported studies under configuration control. Preserve legacy
scripts and label their validation status rather than deleting them.
