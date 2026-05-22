package dev.mosaic

import kotlinx.serialization.Serializable

/**
 * Sidecar metadata serialized to JSON alongside the binary `.bin` file. Holds
 * everything needed to validate the binary plus human-readable provenance
 * (creation time, compatibility flags).
 *
 * Use [EmbeddingFormat.readMetadata] to parse a sidecar without loading the
 * full embedding matrix.
 */
@Serializable
public data class EmbeddingMetadata(
    public val version: Int,
    public val vocabSize: Int,
    public val embeddingDim: Int,
    public val format: String,
    public val byteOrder: String,
    public val checksum: String,
    public val createdAt: String,
    public val tesseraCompatible: Boolean,
)
