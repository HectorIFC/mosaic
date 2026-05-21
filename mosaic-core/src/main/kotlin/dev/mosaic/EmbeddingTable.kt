package dev.mosaic

import dev.mosaic.internal.FlatMatrix
import dev.mosaic.internal.TopKHeap
import dev.mosaic.internal.requirePositiveDim
import dev.mosaic.internal.requireValidId
import dev.mosaic.internal.requireVectorDim

/**
 * A trainable embedding table mapping token IDs to dense [Float] vectors.
 *
 * Mosaic provides the storage, lookup, persistence, and basic vector
 * operations — it does **not** implement training. Callers update weights
 * externally via [set] or [update]; auto-supervised training (Word2Vec,
 * backprop) is out of scope.
 *
 * Typical usage:
 *
 * ```kotlin
 * val table = EmbeddingTable.create(vocabSize = 1000, embeddingDim = 64)
 * val vector = table.get(id = 42)
 * val similar = table.mostSimilar(id = 42, topK = 5)
 * ```
 *
 * Persistence (`save` / `load`) and Tessera integration arrive in Phase 2.
 */
public class EmbeddingTable internal constructor(
    public val vocabSize: Int,
    public val embeddingDim: Int,
    private val matrix: FlatMatrix,
) {

    /** Returns a copy of the row at [id]. Mutating the returned array does not affect the table. */
    public fun get(id: Int): FloatArray {
        requireValidId(id, vocabSize)
        return matrix.getRow(id)
    }

    /** Returns copies of the rows at the given [ids], in order. */
    public fun get(ids: IntArray): Array<FloatArray> {
        return Array(ids.size) { idx -> get(ids[idx]) }
    }

    /** Writes [vector] into the row at [id]. The source array is copied; the caller may mutate it freely afterwards. */
    public fun set(id: Int, vector: FloatArray) {
        requireValidId(id, vocabSize)
        requireVectorDim(vector, embeddingDim)
        matrix.setRow(id, vector)
    }

    /**
     * Reads the row at [id], applies [transform], and writes the result back.
     * Useful for external training loops that compute a gradient-style update
     * per token.
     */
    public fun update(id: Int, transform: (FloatArray) -> FloatArray) {
        val current = get(id)
        val updated = transform(current)
        set(id, updated)
    }

    /**
     * Returns the top-[topK] most similar IDs to the row at [id] by cosine
     * similarity, sorted descending. When [includeSelf] is `true` (default)
     * the query's own ID is eligible to appear — it typically scores `1.0f`
     * and is the first result. Set `false` to omit it.
     *
     * `topK <= 0` returns an empty list; `topK >= vocabSize` returns every ID.
     */
    public fun mostSimilar(id: Int, topK: Int = DEFAULT_TOP_K, includeSelf: Boolean = true): List<Similarity> {
        requireValidId(id, vocabSize)
        val query = matrix.getRow(id)
        return mostSimilarInternal(query, topK, excludeId = if (includeSelf) -1 else id)
    }

    /**
     * Returns the top-[topK] most similar IDs to the [query] vector by cosine
     * similarity, sorted descending. The query is not required to match any
     * existing row, but its dimension must equal [embeddingDim].
     */
    public fun mostSimilar(query: FloatArray, topK: Int = DEFAULT_TOP_K): List<Similarity> {
        requireVectorDim(query, embeddingDim, "query")
        return mostSimilarInternal(query, topK, excludeId = -1)
    }

    private fun mostSimilarInternal(query: FloatArray, topK: Int, excludeId: Int): List<Similarity> {
        if (topK <= 0) return emptyList()
        val effectiveK = if (topK > vocabSize) vocabSize else topK
        val heap = TopKHeap(effectiveK)
        val rowBuf = FloatArray(embeddingDim)
        for (row in 0 until vocabSize) {
            if (row == excludeId) continue
            matrix.getRow(row, rowBuf)
            val score = VectorOps.cosineSimilarity(query, rowBuf)
            heap.offer(row, score)
        }
        return heap.toSortedListDescending()
    }

    public companion object {

        /**
         * Creates a fresh table populated by [initializer]. The default initializer
         * mirrors PyTorch's `nn.Embedding` (uniform in `[-0.5/dim, +0.5/dim]`).
         */
        public fun create(
            vocabSize: Int,
            embeddingDim: Int,
            initializer: Initializer = Initializer.uniformDefault(),
        ): EmbeddingTable {
            requirePositiveDim(vocabSize, "vocabSize")
            requirePositiveDim(embeddingDim, "embeddingDim")
            val matrix = FlatMatrix(vocabSize, embeddingDim)
            val rowBuf = FloatArray(embeddingDim)
            for (row in 0 until vocabSize) {
                initializer.fill(rowBuf, row)
                matrix.setRow(row, rowBuf)
            }
            return EmbeddingTable(vocabSize, embeddingDim, matrix)
        }

        private const val DEFAULT_TOP_K: Int = 10
    }
}
