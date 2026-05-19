# Syntactic Clarity Conventions

- Optimize expressions for one-pass left-to-right human parsing.
- Avoid deferred operators: if an operator appears after a function call, nested subscript, or multi-level attribute and forces mental reparsing, add parentheses or explicit structure.
- Use explicit operator functions (`operator.getitem`, `operator.invert`, `operator.neg`) only when symbolic operators are hidden inside complex expressions. Do not over-structure simple expressions.
- Replace ambiguous numeric adjustments with semantic identifiers when the adjustment is repeated or context-dependent. Look for `_semiotics.py`, `_theTypes.py`, `theTypes.py`, or exports in `__init__.py` before proposing new semantic identifiers.
- Common semantic adjustment ideas: `inclusive`, `zeroIndexed`, `decreasing`, `offsetForExclusiveUpperBound`; define centrally, not inline.
- Standard comparison orientation: prefer `<` and `<=`; rewrite `maximum > value` as `value < maximum` when semantics are identical.
- Import from highest-level public APIs. Avoid third-party private modules when a public re-export exists.
- Prefer absolute imports within local packages over relative imports for review clarity.
- Transformation workflow for clarity-only tasks: identify deferred/hidden operators, ambiguous literals, wrong comparison orientation, and import issues; locate existing semantic identifiers; transform; verify behavior unchanged.
- For syntax verification, use available static tools (Pyright/Pylance when configured, otherwise at least run targeted tests or parser checks).