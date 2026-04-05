# How to contribute to Modern classicalDFT

We hope researchers working in classical DFT find this library useful. Whether you fix a bug, improve performance, extend a module, or contribute a new energy model, **we'll thank you for it!**

Please follow the procedure below so that changes are properly tracked.

## Contribution procedure

1. Submit an issue describing your proposed change to the [issue tracker](https://github.com/migduroli/classicalDFT/issues). Check that it is not already being worked on.
2. One logical change per issue. Do not mix unrelated changes.
3. Coordinate with team members listed on the issue to avoid redundancy.
4. Fork the repo, develop and test your code changes.
5. Ensure your code follows [`CODING_GUIDELINES.md`](CODING_GUIDELINES.md) — naming, formatting, value semantics, test conventions, etc.
6. Run `make format` and `make lint` before committing.
7. Write Catch2 tests for all new functionality. All existing tests must continue to pass.
8. Add a doc example under `docs/` with a `README.md`, `main.cpp`, and `Makefile` (see existing examples).
9. Submit a well-documented pull request.

## Style

All coding conventions are defined in [`CODING_GUIDELINES.md`](CODING_GUIDELINES.md). Key points:

- C++23, formatted with `.clang-format` (PEP8-like, 120 col, 2-space indent)
- `struct` for data and configuration, `class` only when encapsulation serves the user
- Value semantics: functions return results, never mutate arguments
- `[[nodiscard]]` on every function that returns a value
- Catch2 v3 for testing, one test file per module