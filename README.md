# src-cascades

`src-cascades` is a Python library for simulating, analyzing, and fitting self-reinforcing cascade (SRC) models.

## Installation

Base install:

```bash
pip install .
```

Optional extras:

```bash
pip install .[inference]
pip install .[cpp]
pip install .[all]
```

The `inference` extra enables MCMC fitting and ArviZ plotting. The `cpp` extra installs the build dependency used by the optional PyBind11 backend.

## Features

- Branching-process SRC simulation in pure Python, with an optional C++ backend
- PGF-based size-distribution and supercriticality calculations
- Network-based SRC simulation on `networkx` graphs
- YAML-driven experiment runner for simulation sweeps and inference

## Development

The project uses a `src/` layout. Tests can be run with:

```bash
pytest
```
