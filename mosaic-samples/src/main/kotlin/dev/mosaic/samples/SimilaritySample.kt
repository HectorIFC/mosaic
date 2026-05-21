package dev.mosaic.samples

import dev.mosaic.EmbeddingTable
import dev.mosaic.Initializer

/**
 * Plants 8 hand-crafted unit-ish vectors representing compass directions and
 * shows that [EmbeddingTable.mostSimilar] surfaces the geometrically nearby
 * neighbors first.
 *
 * Run with:
 * ```
 * ./gradlew :mosaic-samples:run -PmainClass=dev.mosaic.samples.SimilaritySampleKt
 * ```
 */
fun main() {
    println("=== Similarity Sample ===")

    val table = EmbeddingTable.create(vocabSize = 8, embeddingDim = 3, initializer = Initializer.zeros())

    val labels = listOf(
        "east",
        "near-east",
        "north",
        "west",
        "NE",
        "up",
        "up-NE",
        "NW",
    )
    val vectors = listOf(
        floatArrayOf(1f, 0f, 0f),
        floatArrayOf(0.9f, 0.1f, 0f),
        floatArrayOf(0f, 1f, 0f),
        floatArrayOf(-1f, 0f, 0f),
        floatArrayOf(0.7f, 0.7f, 0f),
        floatArrayOf(0f, 0f, 1f),
        floatArrayOf(0.6f, 0.6f, 0.5f),
        floatArrayOf(-0.5f, 0.5f, 0f),
    )
    for ((id, v) in vectors.withIndex()) table.set(id, v)

    println("Top-3 most similar to '${labels[0]}' (excluding self):")
    for (s in table.mostSimilar(id = 0, topK = 3, includeSelf = false)) {
        println("  ${labels[s.id].padEnd(10)} score=${"%+.4f".format(s.score)}")
    }

    println()
    println("Top-3 nearest to a custom query (0.5, 0.5, 0.7):")
    for (s in table.mostSimilar(query = floatArrayOf(0.5f, 0.5f, 0.7f), topK = 3)) {
        println("  ${labels[s.id].padEnd(10)} score=${"%+.4f".format(s.score)}")
    }
}
