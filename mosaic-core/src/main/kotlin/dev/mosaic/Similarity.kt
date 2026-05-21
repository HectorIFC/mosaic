package dev.mosaic

/**
 * Result of a similarity search: a token [id] paired with its similarity [score]
 * (cosine, range `[-1, 1]`). Returned by [EmbeddingTable.mostSimilar].
 */
public data class Similarity(public val id: Int, public val score: Float)
