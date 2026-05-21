# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

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
- Internal: `FlatMatrix` (1D storage per PRD §4.6), `TopKHeap` (O(N log K) for `mostSimilar`), `Validators`
- 50 unit tests covering all Phase 1 acceptance criteria (round-trip, symmetry, range, ordering, boundary errors)
- Coverage: 99.3% lines / 95.7% methods / 95.8% branches (threshold 80%)

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
