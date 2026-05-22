package dev.mosaic

import io.kotest.assertions.throwables.shouldThrow
import io.kotest.core.spec.style.StringSpec
import io.kotest.matchers.collections.shouldHaveSize
import io.kotest.matchers.floats.plusOrMinus
import io.kotest.matchers.shouldBe

private const val EPS = 1e-5f

class EmbeddingTableTest : StringSpec({

    "create populates rows using the initializer" {
        val table = EmbeddingTable.create(vocabSize = 4, embeddingDim = 3, initializer = Initializer.constant(0.7f))
        for (id in 0 until 4) {
            table.get(id).all { it == 0.7f } shouldBe true
        }
    }

    "create rejects non-positive vocabSize / embeddingDim" {
        shouldThrow<IllegalArgumentException> { EmbeddingTable.create(vocabSize = 0, embeddingDim = 4) }
        shouldThrow<IllegalArgumentException> { EmbeddingTable.create(vocabSize = -1, embeddingDim = 4) }
        shouldThrow<IllegalArgumentException> { EmbeddingTable.create(vocabSize = 4, embeddingDim = 0) }
    }

    "set + get round-trip preserves floats exactly" {
        val table = EmbeddingTable.create(vocabSize = 10, embeddingDim = 5, initializer = Initializer.zeros())
        val expected = floatArrayOf(0.1f, -2.5f, 1e6f, Float.MIN_VALUE, Float.MAX_VALUE)
        table.set(id = 3, vector = expected)
        val got = table.get(3)
        got.contentEquals(expected) shouldBe true
    }

    "get returns a copy — mutating the result does not affect the table" {
        val table = EmbeddingTable.create(vocabSize = 2, embeddingDim = 3, initializer = Initializer.constant(1f))
        val row = table.get(0)
        row[0] = 999f
        table.get(0)[0] shouldBe 1f
    }

    "set copies the input — mutating the source array does not affect the table" {
        val table = EmbeddingTable.create(vocabSize = 2, embeddingDim = 3, initializer = Initializer.zeros())
        val src = floatArrayOf(1f, 2f, 3f)
        table.set(0, src)
        src[0] = 999f
        table.get(0)[0] shouldBe 1f
    }

    "get(IntArray) returns rows in order" {
        val table = EmbeddingTable.create(vocabSize = 5, embeddingDim = 2, initializer = Initializer.zeros())
        table.set(2, floatArrayOf(2f, 2f))
        table.set(4, floatArrayOf(4f, 4f))
        val rows = table.get(intArrayOf(4, 2, 0))
        rows shouldHaveSize 3
        rows[0][0] shouldBe 4f
        rows[1][0] shouldBe 2f
        rows[2][0] shouldBe 0f
    }

    "update applies the transform and writes the result back" {
        val table = EmbeddingTable.create(vocabSize = 3, embeddingDim = 4, initializer = Initializer.constant(1f))
        table.update(1) { row -> FloatArray(row.size) { row[it] * 2f } }
        table.get(1).all { it == 2f } shouldBe true
    }

    "invalid IDs throw with clear messages" {
        val table = EmbeddingTable.create(vocabSize = 5, embeddingDim = 4, initializer = Initializer.zeros())
        shouldThrow<IllegalArgumentException> { table.get(-1) }
        shouldThrow<IllegalArgumentException> { table.get(5) }
        shouldThrow<IllegalArgumentException> { table.set(5, FloatArray(4)) }
    }

    "set with wrong dimension throws" {
        val table = EmbeddingTable.create(vocabSize = 3, embeddingDim = 4, initializer = Initializer.zeros())
        shouldThrow<IllegalArgumentException> { table.set(0, FloatArray(3)) }
    }

    "mostSimilar(id) returns self first when includeSelf = true (score ≈ 1.0)" {
        val table = embeddingsWithSeed(vocabSize = 20, embeddingDim = 8, seed = 1L)
        val result = table.mostSimilar(id = 5, topK = 5, includeSelf = true)
        result.first().id shouldBe 5
        result.first().score shouldBe 1f.plusOrMinus(EPS)
    }

    "mostSimilar(id) excludes self when includeSelf = false" {
        val table = embeddingsWithSeed(vocabSize = 20, embeddingDim = 8, seed = 1L)
        val result = table.mostSimilar(id = 5, topK = 5, includeSelf = false)
        result.none { it.id == 5 } shouldBe true
    }

    "mostSimilar returns list sorted by score descending" {
        val table = embeddingsWithSeed(vocabSize = 50, embeddingDim = 8, seed = 2L)
        val result = table.mostSimilar(id = 0, topK = 10)
        for (i in 1 until result.size) {
            (result[i - 1].score >= result[i].score) shouldBe true
        }
    }

    "mostSimilar with topK = 0 returns empty list" {
        val table = embeddingsWithSeed(vocabSize = 5, embeddingDim = 4, seed = 3L)
        table.mostSimilar(id = 0, topK = 0) shouldBe emptyList()
    }

    "mostSimilar with topK > vocabSize returns all rows" {
        val table = embeddingsWithSeed(vocabSize = 5, embeddingDim = 4, seed = 3L)
        val result = table.mostSimilar(id = 0, topK = 100, includeSelf = true)
        result shouldHaveSize 5
    }

    "mostSimilar(vector) finds the closest row when planted" {
        val table = EmbeddingTable.create(vocabSize = 5, embeddingDim = 3, initializer = Initializer.zeros())
        table.set(0, floatArrayOf(1f, 0f, 0f)) // cos = 1.0 (perfect match)
        table.set(1, floatArrayOf(0.5f, 0.5f, 0f)) // cos ≈ 0.707
        table.set(2, floatArrayOf(0f, 0f, 1f)) // cos = 0
        table.set(3, floatArrayOf(0.9f, 0.1f, 0f)) // cos ≈ 0.994
        table.set(4, floatArrayOf(-1f, 0f, 0f)) // cos = -1

        val result = table.mostSimilar(query = floatArrayOf(1f, 0f, 0f), topK = 3)
        result[0].id shouldBe 0
        result[1].id shouldBe 3
        result[2].id shouldBe 1
    }

    "mostSimilar(vector) rejects mismatched dimension" {
        val table = embeddingsWithSeed(vocabSize = 5, embeddingDim = 4, seed = 3L)
        shouldThrow<IllegalArgumentException> { table.mostSimilar(query = FloatArray(3)) }
    }
})

private fun embeddingsWithSeed(vocabSize: Int, embeddingDim: Int, seed: Long): EmbeddingTable = EmbeddingTable.create(
    vocabSize = vocabSize,
    embeddingDim = embeddingDim,
    initializer = Initializer.uniform(bound = 1f, seed = seed),
)
