# ADR 0001 - Python 3.12 pinned through uv

## Status

Accepted

## Context

The guide assumed Python 3.11 or newer. The machine runs 3.14.8, and a dry-run check confirmed wheels for every MVP dependency on 3.14. A 3.12 toolchain still has the longest support window plus the widest wheel coverage, and uv can install it without touching the system Python. The project needs one command that reproduces the environment on a clean laptop and later on the deploy host.

## Decision

Manage the virtual environment with uv, pinned to CPython 3.12. Pin exact versions in `requirements.txt`.

## Consequences

Installs reproduce on any laptop and on the deploy host. One extra toolchain download. If a future package drops 3.12 support, the pin moves and this ADR is superseded.
