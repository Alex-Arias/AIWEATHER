# Contributing to AIWeather

Thank you for contributing to AIWeather.

## Coding Standards

- Follow PEP 8.
- Use Python type hints.
- Use Google-style docstrings.
- Keep functions focused on a single responsibility.
- Write readable, maintainable code.
- Avoid hard-coded paths.
- Keep AIWeather independent of:
  - HPC clusters
  - AI models
  - datasets
  - schedulers

## Git Workflow

Every feature should follow:

1. Design
2. Implementation
3. Testing
4. Commit

Each commit should represent one logical change.

## Testing

Before committing, verify:

- Imports
- Syntax (`compileall`)
- Unit tests (when available)
- Git status

## Documentation

Public classes and functions should include docstrings.

Configuration files should be documented.