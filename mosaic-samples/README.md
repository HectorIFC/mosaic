# mosaic-samples

Runnable examples demonstrating the `mosaic-core` library. Each sample is a top-level `main()` function — pick one and pass its main class via `-PmainClass=...`.

## Running a sample

```bash
./gradlew :mosaic-samples:run -PmainClass=dev.mosaic.samples.QuickStartSampleKt
```

Replace the class name with any of the entries below.

## Available samples

| Sample | Main class | What it shows |
|---|---|---|
| `QuickStartSample.kt` | `dev.mosaic.samples.QuickStartSampleKt` | Minimal API tour — `create`, `get`, `set`, `update`, `mostSimilar` on a 1 000 × 64 table |
| `TesseraIntegrationSample.kt` | `dev.mosaic.samples.TesseraIntegrationSampleKt` | Full text → vectors pipeline: trains a Tessera BPE tokenizer inline, wires it into `TesseraEmbeddings`, encodes 5 phrases and mean-pools one |
| `SimilaritySample.kt` | `dev.mosaic.samples.SimilaritySampleKt` | Hand-crafted compass vectors that make `mostSimilar` results easy to verify by eye |
| `PersistenceSample.kt` | `dev.mosaic.samples.PersistenceSampleKt` | Saves a table to a temp dir, prints the `.bin` size + the JSON sidecar contents, then reloads and verifies row-by-row equality |
| `InitializationSample.kt` | `dev.mosaic.samples.InitializationSampleKt` | Computes min / max / mean / stddev of values produced by every built-in `Initializer` on a 1 000 × 64 table |

## Benchmark

`BenchmarkSample.kt` measures `mostSimilar`, `save`, and `load` across `vocabSize ∈ {10 000, 50 000, 100 000}` at `dim = 128`. Numbers from a fresh run are captured in [`../BENCHMARKS.md`](../BENCHMARKS.md).

```bash
./gradlew :mosaic-samples:run -PmainClass=dev.mosaic.samples.BenchmarkSampleKt
```
