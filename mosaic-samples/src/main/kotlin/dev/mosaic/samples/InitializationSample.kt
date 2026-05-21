package dev.mosaic.samples

import dev.mosaic.EmbeddingTable
import dev.mosaic.Initializer
import kotlin.math.sqrt

/**
 * Compares the distribution statistics (min / max / mean / stddev) produced by
 * the built-in initializers on a `1000 × 64` table.
 *
 * Run with:
 * ```
 * ./gradlew :mosaic-samples:run -PmainClass=dev.mosaic.samples.InitializationSampleKt
 * ```
 */
fun main() {
    println("=== Initialization Sample ===")

    val vocab = 1000
    val dim = 64

    val initializers = listOf<Pair<String, Initializer>>(
        "uniformDefault" to Initializer.uniformDefault(seed = 42L),
        "uniform(0.5)" to Initializer.uniform(bound = 0.5f, seed = 42L),
        "xavier(64,64)" to Initializer.xavier(fanIn = 64, fanOut = 64, seed = 42L),
        "he(64)" to Initializer.he(fanIn = 64, seed = 42L),
        "zeros" to Initializer.zeros(),
        "constant(0.1)" to Initializer.constant(0.1f),
    )

    val header = "%-18s %12s %12s %12s %12s".format("Initializer", "min", "max", "mean", "stddev")
    println(header)
    println("-".repeat(header.length))

    for ((name, init) in initializers) {
        val table = EmbeddingTable.create(vocab, dim, init)
        val stats = matrixStats(table)
        println("%-18s %12.5f %12.5f %12.5f %12.5f".format(name, stats[0], stats[1], stats[2], stats[3]))
    }
}

private fun matrixStats(table: EmbeddingTable): FloatArray {
    var min = Float.POSITIVE_INFINITY
    var max = Float.NEGATIVE_INFINITY
    var sum = 0.0
    var sumSq = 0.0
    var count = 0L
    for (id in 0 until table.vocabSize) {
        for (x in table.get(id)) {
            if (x < min) min = x
            if (x > max) max = x
            sum += x.toDouble()
            sumSq += x.toDouble() * x.toDouble()
            count++
        }
    }
    val mean = sum / count
    val variance = (sumSq / count) - (mean * mean)
    val stddev = sqrt(if (variance < 0) 0.0 else variance)
    return floatArrayOf(min, max, mean.toFloat(), stddev.toFloat())
}
