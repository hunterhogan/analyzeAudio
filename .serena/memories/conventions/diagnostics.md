# Diagnostic Message Conventions

- Always assign a diagnostic/status message to `message: str` before raising, warning, logging, or printing. Do not inline f-strings in `raise`, `warnings.warn`, logger calls, or `print`.
- Use self-documenting expression syntax for values when useful: `f"I received `{dimensionLength = }`, but ..."`.
- Wrap identifiers, Python keywords, and code elements in backticks inside messages.
- First-person state observation: messages report what the function observed, not what the caller did.
  - Prefer `I received ...`, `I did not receive ...`, `I could not find ...`, `I expected ...`.
  - Avoid `You must ...`, `You cannot ...`, or causation claims about caller intent.
- Align message thesis with the triggering condition. Membership tests should produce not-found/in-container messages; type/value tests should report received and needed values.
- Include runtime values that triggered the condition. Prefer contrastive structure: `I received X, but I need Y.`
- Choose delivery by context: exceptions for precondition violations, `warnings.warn()` for recoverable anomalies, logging for operational logs, `print()`/rich output for interactive status, pytest assertions for tests.
- In pytest tests, prefer native assertions and assertion messages over hidden custom exception messages.
- ANSI color is allowed only for direct terminal output, never exceptions or logs.
- Messages should end with a full stop when they are prose.