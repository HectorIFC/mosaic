# Benchmarks

Performance measurements for `mosaic-core` operations across representative vocabulary sizes.

## Environment

- **Machine:** Apple M1, 8 GB RAM, macOS (Darwin arm64)
- **JVM:** OpenJDK 21 (Temurin / SDKMAN `21-open`)
- **Mosaic version:** `0.1.0-SNAPSHOT`
- **Embedding dim:** 128 (typical for small/medium production embeddings)
- **Methodology:** 2 warm-up runs + 5 measured runs per case, reporting the median wall-clock time. Single thread, JIT warm.

Reproduce locally:

```bash
./gradlew :mosaic-samples:run -PmainClass=dev.mosaic.samples.BenchmarkSampleKt
```

## `mostSimilar(id, topK=10)`

Median time for a single `mostSimilar` call against the entire vocabulary. Implementation uses a fixed-size min-heap of size K for an O(N log K) scan (see `internal/TopKHeap.kt`).

| vocabSize | median latency |
|----------:|---------------:|
| 10 000    | **3.09 ms**    |
| 50 000    | **11.67 ms**   |
| 100 000   | **23.16 ms**   |

Scaling is sub-linear with vocab size in this range, consistent with O(N log K) for fixed K = 10. The PRD §3.2 acceptance criterion of "< 100 ms at vocab 10 000, dim 128" is met with a ~32× margin.

## Persistence (`save` / `load`)

Round-trip timings to local disk (no fsync, no compression). The format is `.bin` (16-byte header + raw float32 LE) + `.meta.json` sidecar with SHA-256.

| vocabSize | save (median) | load (median) |
|----------:|--------------:|--------------:|
| 10 000    | 11.10 ms      | 4.69 ms       |
| 50 000    | 50.48 ms      | 22.14 ms      |
| 100 000   | 108.06 ms     | 80.62 ms      |

Save is dominated by the SHA-256 hash over the entire payload; load is dominated by the file read plus SHA-256 verification.

## Memory residency

Approximate heap occupancy of a freshly-allocated `EmbeddingTable`, measured via `Runtime.getRuntime()` deltas around `System.gc()` calls. The theoretical column is `vocabSize × embeddingDim × 4 bytes`.

| vocabSize | theoretical | measured  |
|----------:|------------:|----------:|
| 10 000    | 4.88 MB     | 5.00 MB   |
| 50 000    | 24.41 MB    | 25.00 MB  |
| 100 000   | 48.83 MB    | 49.00 MB  |

Measured memory tracks the theoretical lower bound within ~1 % — confirming that the flat `FloatArray` storage (PRD §4.6) introduces effectively zero overhead per row, unlike a nested `Array<FloatArray>` which would add an `Array` object header plus a pointer per row.

## Notes

- These are micro-benchmarks, not JMH-grade measurements. They're useful as order-of-magnitude sanity checks, not as the final word on performance.
- Numbers will improve once `mostSimilar` is opt-in pre-normalized (see PRD §4.7 — currently each query recomputes both norms; pre-normalizing rows would halve the per-row work).
- Save/load could be sped up further by streaming the SHA-256 computation alongside the I/O (currently it's two passes over the buffer). Not currently a bottleneck.
