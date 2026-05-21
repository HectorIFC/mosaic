package dev.mosaic.samples

import dev.mosaic.EmbeddingFormat
import dev.mosaic.EmbeddingTable
import dev.mosaic.Initializer
import java.io.File
import java.nio.file.Files

/**
 * Saves an embedding table to a temp directory, prints the on-disk layout
 * (binary + metadata sidecar), then loads it back and verifies that every
 * float survived the round-trip.
 *
 * Run with:
 * ```
 * ./gradlew :mosaic-samples:run -PmainClass=dev.mosaic.samples.PersistenceSampleKt
 * ```
 */
fun main() {
    println("=== Persistence Sample ===")

    val original = EmbeddingTable.create(
        vocabSize = 1000,
        embeddingDim = 64,
        initializer = Initializer.uniformDefault(seed = 7L),
    )

    val tmpDir = Files.createTempDirectory("mosaic-persistence-sample").toFile()
    val binFile = File(tmpDir, "demo.bin")
    val metaFile = File(tmpDir, "demo.bin${EmbeddingFormat.METADATA_EXTENSION}")

    try {
        original.save(binFile)

        val expected = EmbeddingFormat.expectedBinarySize(original.vocabSize, original.embeddingDim)
        println("Saved to: ${binFile.path}")
        println("  .bin size:        ${binFile.length()} bytes (expected $expected)")
        println("  .meta.json size:  ${metaFile.length()} bytes")
        println()
        println("Metadata sidecar:")
        for (line in metaFile.readLines()) println("  $line")
        println()

        val loaded = EmbeddingTable.load(binFile)
        println("Loaded back: vocabSize=${loaded.vocabSize}, embeddingDim=${loaded.embeddingDim}")

        var allEqual = true
        for (id in 0 until original.vocabSize) {
            if (!loaded.get(id).contentEquals(original.get(id))) {
                allEqual = false
                break
            }
        }
        println("Round-trip exact equality across all ${original.vocabSize} rows: $allEqual")
    } finally {
        tmpDir.deleteRecursively()
    }
}
