# mosaic-core

The Mosaic library — a lookup-based, trainable embedding table for the JVM, in pure Kotlin.

This module is the **published artifact**. It depends transitively on [Tessera](https://github.com/HectorIFC/tessera) (byte-level BPE tokenizer) via JitPack, but you don't need to declare that yourself.

## Adding the dependency

### Gradle (Kotlin DSL) — via JitPack

```kotlin
// settings.gradle.kts
dependencyResolutionManagement {
    repositories {
        mavenCentral()
        maven { url = uri("https://jitpack.io") }
    }
}

// build.gradle.kts
dependencies {
    implementation("com.github.HectorIFC:mosaic:mosaic-core-v0.0.4")
}
```

### Gradle (Kotlin DSL) — via GitHub Packages

```kotlin
// settings.gradle.kts
dependencyResolutionManagement {
    repositories {
        mavenCentral()
        maven {
            url = uri("https://maven.pkg.github.com/HectorIFC/mosaic")
            credentials {
                username = System.getenv("GITHUB_ACTOR")
                password = System.getenv("GITHUB_TOKEN")
            }
        }
    }
}

// build.gradle.kts
dependencies {
    implementation("dev.mosaic:mosaic-core:0.0.4")
}
```

## Runtime dependencies

| Dependency | Version | Why |
|---|---|---|
| `org.jetbrains.kotlin:kotlin-stdlib` | 2.3.21 | Base Kotlin runtime |
| `org.jetbrains.kotlinx:kotlinx-serialization-json` | 1.7.3 | JSON metadata sidecar for saved tables |
| `com.github.HectorIFC:tessera` | v0.0.6 | BPE tokenizer used by `TesseraEmbeddings` |

No ML libraries (no DJL, no KInference, no DL4J). No math libraries (no `multik`, no `kmath`).

## Public API at a glance

```kotlin
package dev.mosaic

// Core
class EmbeddingTable {
    val vocabSize: Int
    val embeddingDim: Int

    fun get(id: Int): FloatArray
    fun get(ids: IntArray): Array<FloatArray>
    fun set(id: Int, vector: FloatArray)
    fun update(id: Int, transform: (FloatArray) -> FloatArray)

    fun mostSimilar(id: Int, topK: Int = 10, includeSelf: Boolean = true): List<Similarity>
    fun mostSimilar(query: FloatArray, topK: Int = 10): List<Similarity>

    fun save(path: String); fun save(file: File)

    companion object {
        fun create(vocabSize: Int, embeddingDim: Int, initializer: Initializer = Initializer.uniformDefault()): EmbeddingTable
        fun load(path: String): EmbeddingTable; fun load(file: File): EmbeddingTable
    }
}

data class Similarity(val id: Int, val score: Float)

fun interface Initializer {
    fun fill(target: FloatArray, row: Int)
    companion object {
        fun uniformDefault(seed: Long = 42L): Initializer
        fun uniform(bound: Float, seed: Long = 42L): Initializer
        fun xavier(fanIn: Int, fanOut: Int, seed: Long = 42L): Initializer
        fun he(fanIn: Int, seed: Long = 42L): Initializer
        fun zeros(): Initializer
        fun constant(value: Float): Initializer
    }
}

object VectorOps {
    fun dotProduct(a: FloatArray, b: FloatArray): Float
    fun norm(v: FloatArray): Float
    fun cosineSimilarity(a: FloatArray, b: FloatArray): Float
    fun normalize(v: FloatArray): FloatArray
    fun normalizeInPlace(v: FloatArray)
}

class TesseraEmbeddings(tokenizer: BpeTokenizer, val embeddings: EmbeddingTable) {
    fun encode(text: String): Array<FloatArray>
    fun encodeMeanPooled(text: String): FloatArray
}

// Persistence-format inspection helpers
data class EmbeddingMetadata(val version: Int, val vocabSize: Int, val embeddingDim: Int,
                             val format: String, val byteOrder: String, val checksum: String,
                             val createdAt: String, val tesseraCompatible: Boolean)

object EmbeddingFormat {
    const val MAGIC: Int
    const val FORMAT_VERSION: Int
    const val HEADER_SIZE_BYTES: Int
    const val BYTES_PER_FLOAT: Int
    const val METADATA_EXTENSION: String

    fun expectedBinarySize(vocabSize: Int, embeddingDim: Int): Long
    fun readMetadata(path: String): EmbeddingMetadata; fun readMetadata(file: File): EmbeddingMetadata
    fun verifyChecksum(path: String): Boolean; fun verifyChecksum(file: File): Boolean
}
```

Every public symbol carries a KDoc comment. The module enables `explicitApi()`, which forces every top-level / member declaration to carry an explicit visibility modifier (`public` / `internal` / `private`) and explicit return types on public functions — the compiler rejects code that relies on Kotlin's default visibility for anything that would otherwise be exposed publicly. Anything we explicitly mark `internal` is unreachable from consumers because Kotlin's `internal` visibility is module-scoped.

## Building locally

From the repo root:

```bash
# Build just this module
./gradlew :mosaic-core:build

# Test + coverage
./gradlew :mosaic-core:test :mosaic-core:koverVerify
./gradlew :mosaic-core:koverHtmlReport     # opens at build/reports/kover/html/

# Install into ~/.m2 for local consumption
./gradlew :mosaic-core:publishToMavenLocal
```

## Cross-references

- [`../ARCHITECTURE.md`](../ARCHITECTURE.md) — design rationale (storage layout, top-K algorithm, Float vs Double, persistence format, Tessera integration)
- [`../BENCHMARKS.md`](../BENCHMARKS.md) — measured performance numbers
- [`../mosaic-cli`](../mosaic-cli/) — debug/inspection CLI built on top of this module's public API
- [`../mosaic-samples`](../mosaic-samples/) — runnable usage examples
