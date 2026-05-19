# Type Annotation Conventions

- If the task is only type annotations, do not change logic, formatting, docstrings, or identifiers.
- Annotate precisely and completely: every function/method parameter, return type, class attribute, module-level binding, and non-obvious variable at first authoritative introduction.
- Never use `Any` or `object` unless unavoidable with documented third-party/dynamic boundary justification. Prefer unions, `TypeVar`, `TypedDict`, `Protocol`, or parameterized containers.
- No bare containers. Use `list[T]`, `dict[K, V]`, `tuple[...]`, `set[T]`, `Callable[[...], R]`, `Iterator[T]`, `Generator[Y, S, R]`.
- Use PEP 585 built-in generics, not `typing.List`/`Dict`/`Tuple`/`Set`.
- Use PEP 604 unions, not `Optional` or `Union`.
- For read-only function parameters prefer `collections.abc` abstractions: `Sequence`, `Mapping`, `Iterable`, etc. Return concrete containers when constructing concrete values.
- Do not create annotation-only statements. Annotate at assignment or restructure.
- Do not expand tuple unpacking solely for annotations. Improve the source function return type instead.
- Type source precedence: same module, same package (`_theTypes.py`, `_semiotics.py`, `_theSSOT.py`), parent package (`hunterMakesPy.theTypes`), third-party public type API, standard library.
- Import modern typing helpers from `collections.abc` and `typing` as appropriate. Use `TYPE_CHECKING` for annotation-only imports that are expensive or circular.
- Use `TypedDict` for known-key dictionaries, `Protocol` for structural duck typing, `@overload` when return type depends on input/flags, and descriptive `TypeVar` when preserving input/container type.
- Test modules may intentionally pass invalid types; use block-level pyright directives around such sections rather than per-line `type: ignore`.
- When encountering `typing.cast`, investigate whether it is still necessary. Remove if unnecessary; keep with documentation/comment if required; report uncertainty.