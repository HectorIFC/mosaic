package dev.mosaic.samples

import dev.mosaic.EmbeddingTable
import dev.mosaic.Initializer
import dev.mosaic.TesseraEmbeddings
import dev.tessera.Trainer
import dev.tessera.TrainingConfig

/**
 * Full pipeline: text → Tessera tokens → Mosaic vectors.
 *
 * The tokenizer is trained inline on a small corpus to keep the sample
 * self-contained; in real use you'd load a previously-trained tokenizer
 * via `BpeTokenizer.load(...)`.
 *
 * Run with:
 * ```
 * ./gradlew :mosaic-samples:run -PmainClass=dev.mosaic.samples.TesseraIntegrationSampleKt
 * ```
 */
fun main() {
    println("=== Tessera + Mosaic Integration ===")

    val corpus = buildString {
        repeat(60) {
            append("the quick brown fox jumps over the lazy dog. ")
            append("Olá mundo! Como vai você? ")
            append("hello world 1234 ")
        }
    }
    val tokenizer = Trainer(TrainingConfig(numMerges = 100, verbose = false)).train(corpus)
    println("Trained tokenizer: vocabSize=${tokenizer.vocabSize}")

    val table = EmbeddingTable.create(
        vocabSize = tokenizer.vocabSize,
        embeddingDim = 32,
        initializer = Initializer.uniformDefault(seed = 1L),
    )
    val pipeline = TesseraEmbeddings(tokenizer, table)

    val phrases = listOf("the quick fox", "lazy dog", "olá mundo", "hello world", "1234")
    println()
    for (phrase in phrases) {
        val vectors = pipeline.encode(phrase)
        val preview = vectors.first().take(4).joinToString { "%.4f".format(it) }
        println("'$phrase' → ${vectors.size} vectors of dim ${vectors.first().size}")
        println("  first vector preview: $preview")
    }

    val pooled = pipeline.encodeMeanPooled("the quick brown fox")
    val pooledPreview = pooled.take(4).joinToString { "%.4f".format(it) }
    println()
    println("Mean-pooled 'the quick brown fox' preview: $pooledPreview (dim=${pooled.size})")
}
