package dev.mosaic

import dev.tessera.BpeTokenizer
import dev.tessera.Trainer
import dev.tessera.TrainingConfig
import io.kotest.assertions.throwables.shouldThrow
import io.kotest.core.spec.style.StringSpec
import io.kotest.matchers.floats.plusOrMinus
import io.kotest.matchers.shouldBe
import io.kotest.matchers.string.shouldContain

private const val EPS = 1e-5f

class TesseraIntegrationTest : StringSpec({

    val corpus = buildString {
        repeat(40) {
            append("the quick brown fox jumps over the lazy dog. ")
            append("Olá mundo! Como vai você? ")
            append("hello world 1234 ")
        }
    }
    lateinit var tokenizer: BpeTokenizer

    beforeSpec {
        tokenizer = Trainer(TrainingConfig(numMerges = 50, verbose = false)).train(corpus)
    }

    "TesseraEmbeddings rejects vocabSize mismatch in the constructor" {
        val table = EmbeddingTable.create(vocabSize = tokenizer.vocabSize + 1, embeddingDim = 8)
        val ex = shouldThrow<IllegalArgumentException> { TesseraEmbeddings(tokenizer, table) }
        ex.message!! shouldContain "Vocab size mismatch"
    }

    "encode(text) returns one vector per token in order" {
        val dim = 4
        val table = EmbeddingTable.create(
            vocabSize = tokenizer.vocabSize,
            embeddingDim = dim,
            initializer = Initializer.uniformDefault(seed = 1L),
        )
        val pipeline = TesseraEmbeddings(tokenizer, table)

        val text = "hello world"
        val ids = tokenizer.encode(text)
        val vectors = pipeline.encode(text)

        vectors.size shouldBe ids.size
        for (i in ids.indices) {
            vectors[i].contentEquals(table.get(ids[i])) shouldBe true
        }
    }

    "encode(text) vectors have the configured embeddingDim" {
        val dim = 16
        val table = EmbeddingTable.create(
            vocabSize = tokenizer.vocabSize,
            embeddingDim = dim,
            initializer = Initializer.zeros(),
        )
        val pipeline = TesseraEmbeddings(tokenizer, table)

        val vectors = pipeline.encode("Olá mundo")
        vectors.all { it.size == dim } shouldBe true
    }

    "encodeMeanPooled returns a vector of length embeddingDim" {
        val dim = 8
        val table = EmbeddingTable.create(
            vocabSize = tokenizer.vocabSize,
            embeddingDim = dim,
            initializer = Initializer.uniformDefault(seed = 2L),
        )
        val pipeline = TesseraEmbeddings(tokenizer, table)
        val pooled = pipeline.encodeMeanPooled("hello world")
        pooled.size shouldBe dim
    }

    "encodeMeanPooled equals the per-element mean of the per-token vectors" {
        val dim = 6
        val table = EmbeddingTable.create(
            vocabSize = tokenizer.vocabSize,
            embeddingDim = dim,
            initializer = Initializer.uniform(bound = 1f, seed = 7L),
        )
        val pipeline = TesseraEmbeddings(tokenizer, table)

        val text = "the quick brown fox"
        val perToken = pipeline.encode(text)
        val pooled = pipeline.encodeMeanPooled(text)

        val expected = FloatArray(dim)
        for (v in perToken) for (i in 0 until dim) expected[i] += v[i]
        for (i in 0 until dim) expected[i] /= perToken.size.toFloat()

        for (i in 0 until dim) pooled[i] shouldBe expected[i].plusOrMinus(EPS)
    }

    "encodeMeanPooled on empty text returns a zero vector" {
        val dim = 4
        val table = EmbeddingTable.create(
            vocabSize = tokenizer.vocabSize,
            embeddingDim = dim,
            initializer = Initializer.constant(7f),
        )
        val pipeline = TesseraEmbeddings(tokenizer, table)
        // Tessera emits zero tokens for empty input; the pooled result must
        // be a zero vector of the configured dim (not NaN, not the constant
        // initializer fill — the input is empty, so pooling has nothing to
        // average and the documented contract is "return a zero vector").
        val pooled = pipeline.encodeMeanPooled("")
        pooled.size shouldBe dim
        pooled.all { !it.isNaN() } shouldBe true
        pooled.all { it == 0f } shouldBe true
    }
})
