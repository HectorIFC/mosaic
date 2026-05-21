package dev.mosaic.samples

import dev.mosaic.EmbeddingTable
import dev.mosaic.Initializer
import java.io.File
import java.nio.file.Files

/**
 * Micro-benchmark for [EmbeddingTable.mostSimilar] and save/load across a few
 * representative vocab sizes. The numbers are captured into `BENCHMARKS.md`.
 *
 * Run with:
 * ```
 * ./gradlew :mosaic-samples:run -PmainClass=dev.mosaic.samples.BenchmarkSampleKt
 * ```
 */
fun main() {
    val dim = 128
    val vocabSizes = intArrayOf(10_000, 50_000, 100_000)
    val warmupIterations = 2
    val measureIterations = 5

    println("Mosaic benchmark — JVM ${System.getProperty("java.version")}, dim=$dim")
    println()
    println("%-12s %18s %18s %18s".format("vocabSize", "mostSimilar(ms)", "save(ms)", "load(ms)"))
    println("-".repeat(70))

    val tmpDir = Files.createTempDirectory("mosaic-benchmark").toFile()
    try {
        for (vocab in vocabSizes) {
            val table = EmbeddingTable.create(
                vocabSize = vocab,
                embeddingDim = dim,
                initializer = Initializer.uniformDefault(seed = 1L),
            )

            // Top-K timing
            repeat(warmupIterations) { table.mostSimilar(id = 0, topK = 10) }
            val topKTimes = LongArray(measureIterations)
            for (i in 0 until measureIterations) {
                val start = System.nanoTime()
                table.mostSimilar(id = 0, topK = 10)
                topKTimes[i] = System.nanoTime() - start
            }

            // Save timing
            val binFile = File(tmpDir, "bench-$vocab.bin")
            repeat(warmupIterations) { table.save(binFile) }
            val saveTimes = LongArray(measureIterations)
            for (i in 0 until measureIterations) {
                val start = System.nanoTime()
                table.save(binFile)
                saveTimes[i] = System.nanoTime() - start
            }

            // Load timing
            repeat(warmupIterations) { EmbeddingTable.load(binFile) }
            val loadTimes = LongArray(measureIterations)
            for (i in 0 until measureIterations) {
                val start = System.nanoTime()
                EmbeddingTable.load(binFile)
                loadTimes[i] = System.nanoTime() - start
            }

            println(
                "%-12s %18.2f %18.2f %18.2f".format(
                    vocab,
                    median(topKTimes) / NANOS_PER_MS,
                    median(saveTimes) / NANOS_PER_MS,
                    median(loadTimes) / NANOS_PER_MS,
                ),
            )
        }
    } finally {
        tmpDir.deleteRecursively()
    }

    println()
    printMemoryReport(dim, vocabSizes)
}

@Suppress("ExplicitGarbageCollectionCall")
private fun printMemoryReport(dim: Int, vocabSizes: IntArray) {
    println("Approximate heap residency (single table):")
    println("%-12s %18s %18s".format("vocabSize", "theoretical(MB)", "measured(MB)"))
    println("-".repeat(52))
    for (vocab in vocabSizes) {
        val theoreticalBytes = vocab.toLong() * dim * BYTES_PER_FLOAT
        val theoreticalMb = theoreticalBytes.toDouble() / BYTES_PER_MB
        val runtime = Runtime.getRuntime()
        System.gc()
        val baseline = runtime.totalMemory() - runtime.freeMemory()
        val table = EmbeddingTable.create(
            vocabSize = vocab,
            embeddingDim = dim,
            initializer = Initializer.uniformDefault(seed = 1L),
        )
        System.gc()
        val withTable = runtime.totalMemory() - runtime.freeMemory()
        val measuredMb = (withTable - baseline).coerceAtLeast(0).toDouble() / BYTES_PER_MB
        println("%-12s %18.2f %18.2f".format(vocab, theoreticalMb, measuredMb))
        // Keep `table` alive past the measurement so GC doesn't collect it mid-scope.
        require(table.vocabSize == vocab)
    }
}

private fun median(times: LongArray): Double {
    val sorted = times.sortedArray()
    return sorted[sorted.size / 2].toDouble()
}

private const val NANOS_PER_MS: Double = 1_000_000.0
private const val BYTES_PER_FLOAT: Int = 4
private const val BYTES_PER_MB: Double = 1024.0 * 1024.0
