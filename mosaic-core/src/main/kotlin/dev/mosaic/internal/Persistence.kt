package dev.mosaic.internal

import dev.mosaic.EmbeddingFormat
import dev.mosaic.EmbeddingMetadata
import dev.mosaic.EmbeddingTable
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.security.MessageDigest
import java.time.Instant

internal object Persistence {

    val json: Json = Json { prettyPrint = true }

    fun save(table: EmbeddingTable, binFile: File) {
        val metaFile = metaFileFor(binFile)
        val data = table.rawData
        val bytes = encodeBinary(table.vocabSize, table.embeddingDim, data)
        binFile.writeBytes(bytes)

        val meta = EmbeddingMetadata(
            version = EmbeddingFormat.FORMAT_VERSION,
            vocabSize = table.vocabSize,
            embeddingDim = table.embeddingDim,
            format = "float32-le",
            byteOrder = "little-endian",
            checksum = sha256Hex(bytes),
            createdAt = Instant.now().toString(),
            tesseraCompatible = true,
        )
        metaFile.writeText(json.encodeToString(meta))
    }

    fun load(binFile: File): EmbeddingTable {
        require(binFile.exists()) { "Embedding binary not found: ${binFile.path}" }
        val metaFile = metaFileFor(binFile)
        require(metaFile.exists()) { "Embedding metadata not found: ${metaFile.path}" }

        val meta = json.decodeFromString<EmbeddingMetadata>(metaFile.readText())
        require(meta.version == EmbeddingFormat.FORMAT_VERSION) {
            "Unsupported metadata version: ${meta.version} (expected ${EmbeddingFormat.FORMAT_VERSION})"
        }

        val expectedSize = EmbeddingFormat.expectedBinarySize(meta.vocabSize, meta.embeddingDim)
        require(binFile.length() == expectedSize) {
            "Corrupted .bin: expected $expectedSize bytes, got ${binFile.length()}"
        }

        val bytes = binFile.readBytes()
        val actualChecksum = sha256Hex(bytes)
        require(actualChecksum == meta.checksum) {
            "Checksum mismatch — the .bin file may be corrupted. " +
                "Metadata expected ${meta.checksum}, file is $actualChecksum"
        }

        return decodeBinary(bytes, meta.vocabSize, meta.embeddingDim)
    }

    private fun encodeBinary(vocabSize: Int, embeddingDim: Int, data: FloatArray): ByteArray {
        val total = EmbeddingFormat.expectedBinarySize(vocabSize, embeddingDim).toInt()
        val buffer = ByteBuffer.allocate(total).order(ByteOrder.LITTLE_ENDIAN)
        buffer.putInt(EmbeddingFormat.MAGIC)
        buffer.putInt(EmbeddingFormat.FORMAT_VERSION)
        buffer.putInt(vocabSize)
        buffer.putInt(embeddingDim)
        for (f in data) buffer.putFloat(f)
        return buffer.array()
    }

    private fun decodeBinary(bytes: ByteArray, vocabSize: Int, embeddingDim: Int): EmbeddingTable {
        val buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
        val magic = buffer.int
        require(magic == EmbeddingFormat.MAGIC) {
            "Invalid magic number: 0x${magic.toUInt().toString(HEX_RADIX)} (expected MOSC)"
        }
        val version = buffer.int
        require(version == EmbeddingFormat.FORMAT_VERSION) {
            "Unsupported binary version: $version"
        }
        val fileVocabSize = buffer.int
        val fileEmbeddingDim = buffer.int
        require(fileVocabSize == vocabSize && fileEmbeddingDim == embeddingDim) {
            "Header/metadata mismatch: header says ${fileVocabSize}x$fileEmbeddingDim, " +
                "metadata says ${vocabSize}x$embeddingDim"
        }

        val data = FloatArray(vocabSize * embeddingDim)
        for (i in data.indices) data[i] = buffer.float
        return EmbeddingTable.unsafeFromRawData(vocabSize, embeddingDim, data)
    }

    /** Parses the sidecar JSON without touching the binary payload. */
    fun readMetadata(binFile: File): EmbeddingMetadata {
        val metaFile = metaFileFor(binFile)
        require(metaFile.exists()) { "Embedding metadata not found: ${metaFile.path}" }
        return json.decodeFromString(metaFile.readText())
    }

    /**
     * Returns `true` iff the SHA-256 of the binary payload matches the
     * checksum recorded in the sidecar metadata. Returns `false` if the
     * checksum is wrong; throws if either file is missing or malformed.
     */
    fun verifyChecksum(binFile: File): Boolean {
        require(binFile.exists()) { "Embedding binary not found: ${binFile.path}" }
        val meta = readMetadata(binFile)
        val actual = sha256Hex(binFile.readBytes())
        return actual == meta.checksum
    }

    fun sha256Hex(bytes: ByteArray): String {
        val digest = MessageDigest.getInstance("SHA-256").digest(bytes)
        return buildString(digest.size * 2) {
            for (b in digest) {
                val v = b.toInt() and BYTE_MASK
                append(HEX_CHARS[v ushr NIBBLE_BITS])
                append(HEX_CHARS[v and NIBBLE_MASK])
            }
        }
    }

    fun metaFileFor(binFile: File): File = File(binFile.parentFile, binFile.name + EmbeddingFormat.METADATA_EXTENSION)

    private const val HEX_RADIX: Int = 16
    private const val BYTE_MASK: Int = 0xFF
    private const val NIBBLE_BITS: Int = 4
    private const val NIBBLE_MASK: Int = 0x0F
    private val HEX_CHARS: CharArray = "0123456789abcdef".toCharArray()
}
