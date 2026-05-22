# Architecture

This document explains the **why** behind Mosaic's design choices. For the **what** (API surface, command reference, build instructions), see [`README.md`](./README.md), [`mosaic-core/README.md`](./mosaic-core/README.md), and [`mosaic-cli/README.md`](./mosaic-cli/README.md).

---

## Module layout

Three Gradle modules, with a strict dependency direction:

```
                                consumers
                                    │
                              ┌──── ▼ ────┐
                  ┌──────────►│ mosaic-cli│
                  │           └───────────┘
                  │
        ┌─────────┴─────────┐
        │   mosaic-core     │◄── mosaic-samples
        │  (the library)    │
        └─────────┬─────────┘
                  │
              ┌───▼────┐
              │tessera │   (JitPack: HectorIFC/tessera)
              └────────┘
```

- **`mosaic-core`** is the only published artifact. It has `explicitApi()` enabled, so its public surface is locked down — anything not marked `public` is `internal` and unreachable from `mosaic-cli` or `mosaic-samples`. This is structurally enforced by Kotlin's module-scoped `internal` visibility, not just convention.
- **`mosaic-cli`** is a runnable wrapper that exists primarily for interactive debugging. It must use only `mosaic-core`'s public API — a constraint that the module boundary makes impossible to violate (and which we double-checked by grepping the CLI sources for `dev.mosaic.internal` imports → zero matches).
- **`mosaic-samples`** is a separate module so the demo programs never bleed into the library's classpath.

Tessera is a hard dependency at the `api` configuration level: any consumer of `mosaic-core` gets `dev.tessera.*` transitively. This is intentional — `TesseraEmbeddings` exposes `BpeTokenizer` directly in its public signature.

---

## Storage: flat 1D `FloatArray`

The embedding matrix lives in a single contiguous `FloatArray` of size `vocabSize × embeddingDim`. Row `r` occupies indices `[r × embeddingDim, (r + 1) × embeddingDim)`.

### Why not `Array<FloatArray>`?

```kotlin
// ❌ The naïve layout
val data = Array(vocabSize) { FloatArray(embeddingDim) }

// ✅ What Mosaic does
val data = FloatArray(vocabSize * embeddingDim)
```

The naïve layout is an array of *pointers* to per-row arrays. Every row access dereferences a pointer; consecutive rows can land arbitrarily far apart in heap memory; each row carries an object header (~16 bytes on 64-bit JVM).

The flat layout:

1. Allocates one large primitive buffer — no boxing, no per-row object overhead, ~1 % heap waste over the theoretical bound (measured in [`BENCHMARKS.md`](./BENCHMARKS.md)).
2. Is cache-friendly: a `mostSimilar` scan of 50 000 rows × 128 dims streams 25 MB of sequential `float`s, which the CPU prefetcher loves.
3. Makes `save` and `load` trivial: dump the whole buffer in one `ByteBuffer.putFloat` loop; no need to walk a 2D structure.

The trade-off is that row indexing requires manual `row × cols` arithmetic. That lives in the `internal` `FlatMatrix` class — the public API never exposes the offset math.

---

## `Float`, not `Double`

Industry-standard embeddings are `float32` (PyTorch, TensorFlow, ONNX). Mosaic follows suit:

- Half the memory of `double` for no measurable accuracy loss in similarity tasks.
- Fits the on-disk format choice (IEEE-754 single precision, 4 bytes per value).
- Avoids autoboxing on the JVM — `FloatArray` is a true primitive array, like Java's `float[]`.

### Where `Double` does appear: accumulation

Long sums of small `Float`s drift. `dotProduct` and `norm` accumulate in `Double` internally and narrow back to `Float` on return:

```kotlin
public fun dotProduct(a: FloatArray, b: FloatArray): Float {
    requireSameSize(a, b)
    var sum = 0.0                                   // Double accumulator
    for (i in a.indices) sum += a[i].toDouble() * b[i].toDouble()
    return sum.toFloat()                            // narrow on return
}
```

The narrowing is intentional — callers should not see a `Double` in the public API, which would suggest more precision than the underlying storage provides.

---

## Top-K with a min-heap

A naïve `mostSimilar` would compute every score, sort the whole list, take the first K. That's `O(N log N)`. For `vocabSize = 50 000` and `topK = 10`, the sort dominates.

Mosaic uses a fixed-capacity min-heap of size K (`internal/TopKHeap.kt`):

```
1. Compute the query vector's norm once.
2. For each row r in [0, vocabSize):
     score := cosineSimilarity(query, row[r])
     if heap.size < K:        push (r, score)
     else if score > heap.peek().score: replaceTop(r, score)
3. Extract the heap, sort the K elements descending.
```

This is `O(N log K)` — for `K = 10` and `N = 100 000` that's about 10× fewer comparisons than a full sort, and the extracted result is a list of size 10 instead of 100 000.

### Why not pre-normalize all rows?

Pre-normalizing means cosine reduces to a single dot product (no per-row norm). It's roughly 2× faster but changes semantics — the table's rows are no longer the "raw" vectors you wrote with `set`. For now we keep `set/get` symmetric.

---

## Persistence format

Two files per saved table:

```
<path>             — binary payload
<path>.meta.json   — sidecar with metadata + integrity info
```

### Binary layout

```
  0..3   magic    (int32 LE)   — must equal 0x43534F4D ("MOSC" on disk)
  4..7   version  (int32 LE)   — 1 today; bumped on any binary breakage
  8..11  vocabSize    (int32 LE)
 12..15  embeddingDim (int32 LE)
 16..    vocabSize × embeddingDim × 4 bytes (IEEE-754 float32 LE)
```

The 16-byte header is fixed and self-describing — you can run `xxd` on any Mosaic file and instantly see what it claims to be. Dimensions are duplicated in both the binary header and the JSON sidecar; `load` checks they agree.

### Little-endian — why?

JVM `ByteBuffer` defaults to big-endian. We override that everywhere via `ByteOrder.LITTLE_ENDIAN`. Reason: the entire ML ecosystem (numpy, PyTorch, TensorFlow, ONNX) standardized on little-endian floats. A Mosaic file should be loadable by tools written in other languages without endian conversion.

### SHA-256 checksum

The sidecar records the SHA-256 of the binary payload (header + floats). `load` recomputes it on every call and fails fast if it doesn't match. This catches bit rot, partial downloads, and any in-flight corruption.

The cost is one extra pass over the bytes per save and per load. It's measurable but small — see `save` / `load` timings in [`BENCHMARKS.md`](./BENCHMARKS.md). The safety win is worth it. The `verifyChecksum` API exposes the check without the full load, for tools that only want integrity verification.

### Validation order on `load`

We check things in this order, so the user gets the most specific error first:

1. `.bin` file exists?
2. `.meta.json` file exists?
3. Metadata version compatible? (rejects future formats)
4. `.bin` file size matches `expectedBinarySize(vocabSize, embeddingDim)`? (catches truncation before we burn cycles on SHA)
5. SHA-256 of `.bin` matches the sidecar's checksum?
6. Binary header's magic / version / vocabSize / dim match the sidecar?

Steps 1–4 are O(1) or single-stat; step 5 is the expensive one but unavoidable.

---

## Tessera integration

`TesseraEmbeddings` is a thin wrapper:

```kotlin
public class TesseraEmbeddings(
    private val tokenizer: BpeTokenizer,
    public val embeddings: EmbeddingTable,
) {
    init {
        require(tokenizer.vocabSize == embeddings.vocabSize) { ... }
    }
    public fun encode(text: String): Array<FloatArray> = embeddings.get(tokenizer.encode(text))
    public fun encodeMeanPooled(text: String): FloatArray { ... }
}
```

The only constraint enforced at construction is that the tokenizer's vocab and the table's vocab line up. Catching this mismatch in `init` (instead of at the first `encode` call) is one of the few "fail eagerly" hot-paths in the API — vocab-size mismatches are catastrophic and almost always caused by reloading a tokenizer or table from the wrong snapshot.

`encodeMeanPooled` is a sentence-embedding baseline: average all token vectors element-wise. It's not state-of-the-art (transformer-style pooled outputs are far richer) but it's a reasonable starting point and costs nothing beyond `encode` + one pass.

### Why `BpeTokenizer.vocabSize` is the right canonical source

Tessera owns the tokenizer state — including special-token reservations that affect the effective vocab size. If a user constructs `EmbeddingTable` with a hardcoded vocab and the tokenizer ever changes (gains special tokens, etc), the runtime mismatch will be caught. The library tells you exactly where they disagree, not just "out of bounds at id 50000".

---

## CLI design

`mosaic-cli` is a separate module that **only** uses `mosaic-core`'s public API. This is enforced by Kotlin's `internal` visibility being module-scoped: even if a CLI command tried `import dev.mosaic.internal.FlatMatrix`, the compile would fail because `mosaic-core`'s `internal` symbols aren't visible across the module boundary.

### Manual argument parser

`Args.kt` is ~75 lines. We chose manual parsing over `kotlinx-cli` because:

- Mosaic's argument shapes are simple (`--name value` plus boolean flags).
- Tessera uses manual parsing — keeping the sister projects symmetric helps readers who jump between them.
- Zero additional dependencies on the CLI distribution.

The only non-trivial parse rule is distinguishing a short flag (`-h`) from a negative-number value (`-0.5`). The heuristic: a flag name starts with `-` followed by a letter; anything else (numbers, decimals) is a value.

### `runCli(args): Int` instead of `exitProcess`

Tests can't easily catch `exitProcess`. So:

- `main(args)` is a thin shim that calls `runCli(args)` and only invokes `exitProcess` if the return code is non-zero.
- `runCli` returns `Int` directly; tests invoke it and assert on the return code + captured stdout/stderr.

### Exit codes

- **`0`** — success
- **`1`** — usage error (missing/invalid argument, unknown command). Triggered by an `UsageError` thrown anywhere inside a command.
- **`2`** — runtime failure (file not found, corrupted file, checksum mismatch). Triggered by a `RuntimeFailure`.

These are caught in a single dispatcher (`dispatch { ... }`) so commands can just throw and never need to know about exit codes.

### `inspect` reports exit 2 on bad checksum

Unlike most tools where `inspect` is purely informational, `mosaic-cli inspect` returns exit code 2 (not 0) when the SHA-256 doesn't match — so it composes in shell scripts:

```bash
mosaic-cli inspect --input embeddings.bin || echo "file is corrupted"
```

---

## Initializers

`Initializer` is a `fun interface` with six built-in factories on the companion. Notable details:

- **Default = `uniformDefault`**, with bound `0.5 / dim` — matches PyTorch's `nn.Embedding`. Predictable to anyone arriving from a PyTorch background.
- **Every random initializer takes a `seed: Long` parameter**, defaulting to `42L`. Reproducibility is a hard requirement in ML pipelines; we never want a "the test passes locally but fails in CI" because of a re-seeded RNG.
- **The captured `Random` is stateful across rows**. A single initializer instance, used to fill 1 000 rows, produces 1 000 different draws — not the same draw 1 000 times. (We have a dedicated test for this exact gotcha.)

### Xavier and He bounds

```
xavier: bound = sqrt(6 / (fanIn + fanOut))
he:     bound = sqrt(6 / fanIn)
```

These are the *uniform-variant* bounds, not the Gaussian forms. They map cleanly onto `Initializer.uniform(bound, seed)`. For embedding layers, `uniformDefault` is usually the right choice; the other two are there for users wiring Mosaic vectors directly into a downstream MLP.

---

## What's deliberately missing

- **Training.** No backprop, no autograd, no Word2Vec, no GloVe. Mosaic is a *storage and lookup* layer. A separate project will (someday) read/write Mosaic tables as part of a training loop.
- **GPU.** JVM-only. CPU-only. The `mostSimilar` numbers in [`BENCHMARKS.md`](./BENCHMARKS.md) show that for vocab sizes up to ~100 K, this is fine.
- **Quantization.** No int8, no int4. Float32 is the format. If a future project wants quantized tables, it'll be a separate format (and probably a separate library).
- **Multi-platform.** JVM target only. The API would translate cleanly to Kotlin Multiplatform — the only `java.io.File` references are in convenience overloads — but it's not a current goal.
- **Approximate nearest neighbors.** Exact `mostSimilar` only. FAISS-style indexing it`s stretch goals and would belong in a separate library or module.

These are not oversights; they are scoping decisions made explicitly to keep `mosaic-core` small, auditable, and dependency-free.
