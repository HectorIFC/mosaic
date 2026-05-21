package dev.mosaic.samples

import dev.mosaic.EmbeddingTable
import dev.mosaic.Initializer

/**
 * Minimal end-to-end demo of the [EmbeddingTable] public API.
 *
 * Run with:
 * ```
 * ./gradlew :mosaic-samples:run -PmainClass=dev.mosaic.samples.QuickStartSampleKt
 * ```
 */
fun main() {
    println("=== Mosaic QuickStart ===")

    val table = EmbeddingTable.create(
        vocabSize = 1000,
        embeddingDim = 64,
        initializer = Initializer.uniformDefault(seed = 42L),
    )
    println("Created table: vocabSize=${table.vocabSize}, embeddingDim=${table.embeddingDim}")

    val v = table.get(42)
    println("Vector for token 42 — first 4 values: ${v.take(4).joinToString { "%.5f".format(it) }}")

    val customVector = FloatArray(64) { i -> 0.01f * i }
    table.set(id = 100, vector = customVector)
    println("Token 100 overwritten. Sum of new vector: ${"%.4f".format(table.get(100).sum())}")

    table.update(id = 100) { row -> FloatArray(row.size) { i -> row[i] * 10f } }
    println("Token 100 scaled 10x. Sum now: ${"%.4f".format(table.get(100).sum())}")

    println("Top-5 similar to token 100:")
    for (s in table.mostSimilar(id = 100, topK = 5)) {
        println("  token ${s.id.toString().padStart(4)}: ${"%.5f".format(s.score)}")
    }
}
