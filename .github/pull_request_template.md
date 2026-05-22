## Description

<!-- What does this PR do? Why is it needed? -->

## Type of change

- [ ] `feat` — new feature
- [ ] `fix` — bug fix
- [ ] `refactor` — refactoring with no behavior change
- [ ] `test` — adding or fixing tests
- [ ] `docs` — documentation
- [ ] `build` — build / CI changes
- [ ] `chore` — maintenance

## Public API changed?

- [ ] No
- [ ] Yes — describe the diff to the public API in `mosaic-core` below:

```diff
// before
// after
```

> Breaking API changes require a major version bump.

## Checklist

- [ ] `./gradlew test` passes locally
- [ ] `./gradlew koverVerify` passes (coverage ≥ 80%)
- [ ] `./gradlew ktlintCheck` passes (no formatting violations)
- [ ] `./gradlew detekt` passes (no static analysis violations)
- [ ] `CHANGELOG.md` updated under `[Unreleased]`
- [ ] New public symbols have KDoc
- [ ] Everything that is not public API is marked `internal`

## Tests added / modified

<!-- List new or changed tests. -->

## Notes for the reviewer

<!-- Context, design decisions, known pitfalls. -->
