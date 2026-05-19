# Conventions

- Project/user conventions are intentionally stricter than typical Python style; follow them when creating or modifying code.
- Read focused memories as needed:
  - `mem:conventions/code_generation` for semiotics-first functional design, Python-native boundaries, control flow, and destructive-operation rules.
  - `mem:conventions/identifiers` for naming, filesystem terminology, camelCase, and prohibited abbreviations.
  - `mem:conventions/formatting` for line breaks, indentation, comma placement, banners, quotes, and alignment.
  - `mem:conventions/syntactic_clarity` for operator visibility, semantic literals, comparison orientation, and imports.
  - `mem:conventions/types` for annotation completeness, prohibited weak types, and modern typing syntax.
  - `mem:conventions/docstrings` for NumPy-style docstring creation/repair when explicitly requested.
  - `mem:conventions/diagnostics` for exception/warning/log/status message wording.
  - `mem:conventions/tests` for pytest structure, fixtures, parametrization, deterministic test data, and assertion messages.
- Existing code predates some standards. Do not churn unrelated code solely to enforce a convention. Apply rules to touched code and explicitly scoped cleanup.
- When a specific task conflicts with general conventions, preserve behavior first and surface the conflict to the user.