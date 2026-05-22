package dev.mosaic

import dev.tessera.BpeTokenizer

/**
 * End-to-end text → vectors pipeline combining a Tessera [BpeTokenizer] with
 * a Mosaic [EmbeddingTable]. The tokenizer's vocab must match the embedding
 * table's `vocabSize`; mismatches are caught at construction.
 *
 * Typical usage:
 *
 * ```kotlin
 * val tokenizer = BpeTokenizer.load("tessera.json")
 * val table = EmbeddingTable.create(vocabSize = tokenizer.vocabSize, embeddingDim = 128)
 * val pipeline = TesseraEmbeddings(tokenizer, table)
 * val vectors: Array<FloatArray> = pipeline.encode("Olá, mundo!")
 * ```
 */
public class TesseraEmbeddings(
    private val tokenizer: BpeTokenizer,
    public val embeddings: EmbeddingTable,
) {

    init {
        require(tokenizer.vocabSize == embeddings.vocabSize) {
            "Vocab size mismatch: tokenizer=${tokenizer.vocabSize}, embeddings=${embeddings.vocabSize}"
        }
    }

    /**
     * Tokenizes [text] with the wrapped tokenizer and returns the embedding
     * vector for each token, in order. Each returned vector is a copy.
     */
    public fun encode(text: String): Array<FloatArray> {
        val ids = tokenizer.encode(text)
        return embeddings.get(ids)
    }

    /**
     * Returns the element-wise mean of all token embeddings for [text]. A
     * baseline "sentence embedding" — adequate for simple similarity tasks but
     * not as expressive as transformer-style pooled outputs. Empty input yields
     * a zero vector of length [EmbeddingTable.embeddingDim].
     */
    public fun encodeMeanPooled(text: String): FloatArray {
        val vectors = encode(text)
        val dim = embeddings.embeddingDim
        val result = FloatArray(dim)
        if (vectors.isEmpty()) return result
        for (v in vectors) {
            for (i in 0 until dim) result[i] += v[i]
        }
        val n = vectors.size.toFloat()
        for (i in 0 until dim) result[i] /= n
        return result
    }
}
