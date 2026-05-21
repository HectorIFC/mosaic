# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

#### Phase 0 — Setup & infrastructure
- Multi-module Gradle build: `mosaic-core` (library), `mosaic-cli` (application), `mosaic-samples` (examples)
- Kotlin 2.3.21, JVM toolchain 21, `explicitApi()` enabled on `mosaic-core`
- Tessera dependency wired via JitPack (`com.github.HectorIFC:tessera:tessera-core-v0.0.6`)
- GitHub Actions CI workflow: tests, coverage ≥ 80%, ktlint, detekt (parallel jobs on every PR)
- GitHub Actions release workflow: SemVer bump via conventional commits, git tag, publish to GitHub Packages
- `.github/dependabot.yml`: weekly updates for Actions + Gradle deps (grouped: kotlin, kotest, kotlinx)
- `.github/pull_request_template.md`: standardized PR checklist
- ktlint 12.1.2 + detekt 1.23.8 integrated into Gradle build
- `config/detekt/detekt.yml` with Mosaic-specific thresholds (complexity 15, magic numbers tuned for float storage)
- Kover 0.8.3 coverage plugin configured (threshold 80% on `mosaic-core`)
- `.editorconfig` for consistent formatting (IntelliJ code style, max line 120)
- `mosaic-core` smoke tests: hello world + Tessera dependency resolution check
- `mosaic-cli` skeleton with `--help` dispatcher
