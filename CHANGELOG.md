# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [v0.0.1] — 2026-05-22

### Added

#### Phase 6 — Publication & polish
- Root `README.md` rewritten end-to-end in English: badges (CI, JitPack, license, coverage, Kotlin, JVM), updated status, all 6 phases checked, benchmark summary table, related projects, mosaic etymology
- `mosaic-core/README.md` rewritten with full module reference (JitPack + GitHub Packages snippets, runtime deps table, public API at a glance, build instructions, cross-links)
- `ARCHITECTURE.md` covering module layout, flat 1D storage rationale, Float vs Double, top-K min-heap, persistence format and validation order, Tessera integration constraint, CLI design (manual parser, `runCli(args): Int`, exit codes), initializer determinism, and explicitly out-of-scope features
- KDoc coverage audited on every top-level public symbol in `mosaic-core`
- CHANGELOG cut for v0.0.1

#### Phase 5 — GitHub Pages site
- `docs/index.html` — single-page site with 10 sections (nav, hero, stats, features, how-it-works, visualized, quick start, tech stack, quality, footer)
- Orange/black/white palette (`#fb923c` / `#f97316` / `#ea580c` over `#0a0a0a`); Inter + JetBrains Mono via Google Fonts; count-up + scroll fade-in animations
- Logo asset family in `docs/logo/`: `logo.svg`, `logo-{1024,512,256,128,64}.png`, `favicon-{16,32,48,180,192,512}.png`, `favicon.ico` — programmatically generated from a 4×4 tesserae grid with diagonal tonal gradient
- `docs/lookup.mp4` (66 KB) and `docs/similar.mp4` (70 KB) — H.264 animations of the lookup pipeline and `mostSimilar` algorithm; rendered with Pillow + ffmpeg
- `docs/site.webmanifest` — PWA manifest with `theme_color: #f97316`
- Open Graph + Twitter Card meta tags
- Total `docs/` payload: 288 KB

#### Phase 4 — CLI
- `mosaic-cli` real implementation with 5 commands: `create`, `inspect`, `stats`, `similar`, `encode`
- Manual argument parser (`Args.kt`) supporting `--name value`, short aliases, and negative-number values
- Public `EmbeddingMetadata` data class promoted from `internal`
- `EmbeddingFormat.readMetadata(path|file)` and `verifyChecksum(path|file)` — read sidecar / verify integrity without full table load
- Testable `runCli(args): Int` entry point (returns exit code instead of calling `exitProcess`)
- 22 CLI integration tests (every command, --help, exit codes 0/1/2, JSON/CSV/pretty formats)
- `mosaic-cli/README.md` with full command reference
- Static guarantee: CLI uses zero `internal` symbols from `mosaic-core` (enforced by Kotlin module boundary)

#### Phase 3 — Samples, benchmarks & validation
- `mosaic-samples` populated with 5 runnable demos: `QuickStartSample`, `TesseraIntegrationSample`, `SimilaritySample`, `PersistenceSample`, `InitializationSample`
- `BenchmarkSample` runnable: measures `mostSimilar`, `save`, `load`, and memory residency across vocab sizes
- `BENCHMARKS.md` at repo root with real measurements on Apple M1 / JVM 21: `mostSimilar(topK=10)` at vocab 10k takes ~3 ms (32× under the 100 ms acceptance criterion)
- `mosaic-samples/README.md` documenting each sample and how to run it via `-PmainClass=...`

#### Phase 2 — Persistence & Tessera integration
- `EmbeddingFormat`: public constants for the on-disk format (`MAGIC`, `FORMAT_VERSION`, `HEADER_SIZE_BYTES`, `BYTES_PER_FLOAT`, `METADATA_EXTENSION`, `expectedBinarySize`)
- `EmbeddingTable.save(path|file)` and companion `load(path|file)` — binary `.bin` + JSON `.meta.json` sidecar
- 16-byte header (magic "MOSC" + version + vocabSize + embeddingDim, all little-endian) followed by raw float32 LE payload
- SHA-256 checksum verifies integrity at load time; size mismatch is detected before checksum
- `TesseraEmbeddings(tokenizer, embeddings)` public class: validates `vocabSize` match in init; `encode(text): Array<FloatArray>`, `encodeMeanPooled(text): FloatArray`
- Internal: `Persistence` (ByteBuffer LITTLE_ENDIAN + kotlinx-serialization JSON), `EmbeddingMetadata` data class
- 19 new tests (Persistence round-trip, corruption detection, version mismatch; Tessera integration train→encode→pool)

#### Phase 1 — Core library
- `EmbeddingTable` public class: `get(id)`, `get(ids)`, `set`, `update`, `mostSimilar(id, topK, includeSelf)`, `mostSimilar(query, topK)`, `companion.create(vocabSize, embeddingDim, initializer)`
- `Initializer` public fun interface with 6 factories: `uniformDefault` (PyTorch nn.Embedding default), `uniform`, `xavier`, `he`, `zeros`, `constant`
- `VectorOps` public stateless object: `dotProduct`, `norm`, `cosineSimilarity`, `normalize`, `normalizeInPlace` — all with Double accumulation for precision and zero-norm safety
- `Similarity` public data class (id, score)
- Internal: `FlatMatrix` (1D storage per), `TopKHeap` (O(N log K) for `mostSimilar`), `Validators`
- 50 unit tests covering all Phase 1 acceptance criteria (round-trip, symmetry, range, ordering, boundary errors). Final v0.0.1 line coverage on `mosaic-core` is **97.6 %** (well above the 80 % threshold); see the badge in the root README for the up-to-date figure.

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
