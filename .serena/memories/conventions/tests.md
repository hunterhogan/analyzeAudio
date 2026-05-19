# Test Conventions

- Pytest configuration belongs in `pyproject.toml`; fixtures/shared setup belong in `tests/conftest.py`.
- Never create fixtures outside `tests/conftest.py`.
- One test function per function/class being tested.
- All test functions use `@pytest.mark.parametrize`. Single-case parametrization is acceptable.
- Every test is assumed to use fixtures. If a test has no fixture, justify this in the response.
- Fixture names use camelCase, descriptive names; `temp` for temporary resources, `mock` for mocked dependencies.
- Share expensive operations via fixtures rather than repeating setup.
- Combine related assertions over the same result into one test function when setup is shared and properties are related.
- Test data must be deterministic. Prefer static samples from `tests/dataSamples/` when available.
- `tests/dataSamples/` should contain only data, not executable functions/classes.
- Never use random/faker data for tests.
- Synthetic values should be distinctive and non-contiguous; avoid boundary/sentinel values unless boundary behavior is explicitly under test: avoid `0`, `1`, `-1`, `''`, `' '`, `[]`, `[[]]`, `[0]`, alphabetical edge letters.
- Every assertion includes a descriptive message with feature/function under test, actual and expected values, relevant inputs, and final full stop.
- Centralize multi-parameter scenario configuration in one data structure; avoid scattered pickers or fragmented dictionaries.
- Test filenames use `test_<module_name>.py`; test functions use `test_<behavior_being_tested>`; fixtures remain camelCase.
- The user prompt says `Use the MCP test runners`; if unavailable, run the local pytest command and note the fallback.