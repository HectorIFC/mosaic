package dev.mosaic

/**
 * Public constants describing the on-disk binary format of a saved
 * [EmbeddingTable]. Exposed so external tooling can read or inspect Mosaic
 * files without depending on the [EmbeddingTable.load] API.
 *
 * Layout of the `.bin` file:
 * ```
 *  0..3   magic (int32 LE)         — must equal [MAGIC]; on disk: bytes 'M' 'O' 'S' 'C'
 *  4..7   version (int32 LE)       — must equal [FORMAT_VERSION]
 *  8..11  vocabSize (int32 LE)
 * 12..15  embeddingDim (int32 LE)
 * 16..    floats (vocabSize * embeddingDim * 4 bytes, IEEE-754 float32 LE)
 * ```
 *
 * The metadata sidecar (`<bin-path>.meta.json`) stores the SHA-256 checksum,
 * creation timestamp, and a duplicate of the dimensions for human inspection.
 */
public object EmbeddingFormat {

    /**
     * Magic number at offset 0 of every `.bin` file, written as a little-endian
     * int32. On disk this serializes to the ASCII byte sequence `M O S C`.
     */
    public const val MAGIC: Int = 0x43534F4D

    /** Current on-disk format version. Bumped on any binary layout change. */
    public const val FORMAT_VERSION: Int = 1

    /** Size of the fixed header in bytes. */
    public const val HEADER_SIZE_BYTES: Int = 16

    /** Size of one float on disk (IEEE-754 single precision). */
    public const val BYTES_PER_FLOAT: Int = 4

    /** File extension for the metadata sidecar, appended to the binary path. */
    public const val METADATA_EXTENSION: String = ".meta.json"

    /** Returns the expected `.bin` file size for the given dimensions. */
    public fun expectedBinarySize(vocabSize: Int, embeddingDim: Int): Long {
        return HEADER_SIZE_BYTES.toLong() + vocabSize.toLong() * embeddingDim * BYTES_PER_FLOAT
    }
}
