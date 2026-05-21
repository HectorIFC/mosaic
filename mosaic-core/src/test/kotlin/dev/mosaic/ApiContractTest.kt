package dev.mosaic

import io.kotest.core.spec.style.StringSpec
import io.kotest.matchers.shouldBe
import io.kotest.matchers.types.shouldBeInstanceOf

/**
 * Pins the public API surface of `mosaic-core`: if a symbol is renamed,
 * removed, or its signature changes, this test fails at compile time. That
 * makes it impossible to ship an accidental API break without explicitly
 * acknowledging it (and bumping the version).
 */
class ApiContractTest : StringSpec({

    "EmbeddingTable companion exposes create(vocabSize, embeddingDim) with default initializer" {
        val table: EmbeddingTable = EmbeddingTable.create(vocabSize = 4, embeddingDim = 2)
        table.vocabSize shouldBe 4
        table.embeddingDim shouldBe 2
    }

    "EmbeddingTable.create accepts an explicit Initializer" {
        val table = EmbeddingTable.create(vocabSize = 2, embeddingDim = 3, initializer = Initializer.zeros())
        table.get(0).all { it == 0f } shouldBe true
    }

    "EmbeddingTable.get(Int) returns FloatArray" {
        val table = EmbeddingTable.create(vocabSize = 2, embeddingDim = 3, initializer = Initializer.zeros())
        val row: FloatArray = table.get(0)
        row.size shouldBe 3
    }

    "EmbeddingTable.get(IntArray) returns Array<FloatArray>" {
        val table = EmbeddingTable.create(vocabSize = 2, embeddingDim = 3, initializer = Initializer.zeros())
        val rows: Array<FloatArray> = table.get(intArrayOf(0, 1))
        rows.size shouldBe 2
    }

    "EmbeddingTable.set(Int, FloatArray) and update(Int, (FloatArray) -> FloatArray) compile" {
        val table = EmbeddingTable.create(vocabSize = 2, embeddingDim = 3, initializer = Initializer.zeros())
        table.set(id = 0, vector = FloatArray(3) { 1f })
        table.update(id = 0) { it.copyOf() }
    }

    "EmbeddingTable.mostSimilar(Int, topK, includeSelf) returns List<Similarity>" {
        val table = EmbeddingTable.create(vocabSize = 3, embeddingDim = 2, initializer = Initializer.uniformDefault())
        val result: List<Similarity> = table.mostSimilar(id = 0, topK = 2, includeSelf = true)
        result.first().shouldBeInstanceOf<Similarity>()
    }

    "EmbeddingTable.mostSimilar(FloatArray, topK) returns List<Similarity>" {
        val table = EmbeddingTable.create(vocabSize = 3, embeddingDim = 2, initializer = Initializer.uniformDefault())
        val result: List<Similarity> = table.mostSimilar(query = FloatArray(2), topK = 1)
        result.size shouldBe 1
    }

    "Initializer factories all return Initializer" {
        val factories: List<Initializer> = listOf(
            Initializer.uniformDefault(),
            Initializer.uniform(bound = 0.1f),
            Initializer.xavier(fanIn = 4, fanOut = 4),
            Initializer.he(fanIn = 4),
            Initializer.zeros(),
            Initializer.constant(0f),
        )
        factories.size shouldBe 6
    }

    "VectorOps is an object exposing the documented functions" {
        val a = floatArrayOf(1f, 2f, 3f)
        val b = floatArrayOf(4f, 5f, 6f)
        val dot: Float = VectorOps.dotProduct(a, b)
        val norm: Float = VectorOps.norm(a)
        val cos: Float = VectorOps.cosineSimilarity(a, b)
        val nrm: FloatArray = VectorOps.normalize(a)
        VectorOps.normalizeInPlace(nrm)
        (dot + norm + cos + nrm[0]) shouldBe (dot + norm + cos + nrm[0])
    }

    "Similarity is a data class with id and score" {
        val s = Similarity(id = 7, score = 0.5f)
        s.id shouldBe 7
        s.score shouldBe 0.5f
    }
})
