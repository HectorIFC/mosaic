package dev.mosaic

import io.kotest.assertions.throwables.shouldThrow
import io.kotest.core.spec.style.StringSpec
import io.kotest.matchers.shouldBe
import io.kotest.matchers.string.shouldContain
import java.io.File
import java.nio.file.Files

private fun tempBinFile(): File {
    val dir = Files.createTempDirectory("mosaic-persistence").toFile()
    dir.deleteOnExit()
    return File(dir, "embeddings.bin")
}

class PersistenceTest : StringSpec({

    "save then load preserves the matrix exactly" {
        val original = EmbeddingTable.create(
            vocabSize = 20,
            embeddingDim = 8,
            initializer = Initializer.uniform(bound = 1f, seed = 17L),
        )
        val bin = tempBinFile()
        original.save(bin)

        val loaded = EmbeddingTable.load(bin)
        loaded.vocabSize shouldBe original.vocabSize
        loaded.embeddingDim shouldBe original.embeddingDim
        for (id in 0 until original.vocabSize) {
            loaded.get(id).contentEquals(original.get(id)) shouldBe true
        }
    }

    "save writes both .bin and <bin>.meta.json next to each other" {
        val table = EmbeddingTable.create(vocabSize = 3, embeddingDim = 4, initializer = Initializer.zeros())
        val bin = tempBinFile()
        table.save(bin)

        bin.exists() shouldBe true
        File(bin.parentFile, bin.name + ".meta.json").exists() shouldBe true
    }

    "binary file size matches header + vocabSize * embeddingDim * 4" {
        val vocab = 10
        val dim = 5
        val table = EmbeddingTable.create(vocabSize = vocab, embeddingDim = dim, initializer = Initializer.zeros())
        val bin = tempBinFile()
        table.save(bin)

        bin.length() shouldBe EmbeddingFormat.expectedBinarySize(vocab, dim)
    }

    "metadata JSON contains all required fields" {
        val table = EmbeddingTable.create(vocabSize = 5, embeddingDim = 3, initializer = Initializer.zeros())
        val bin = tempBinFile()
        table.save(bin)

        val metaText = File(bin.parentFile, bin.name + ".meta.json").readText()
        metaText shouldContain "\"version\": 1"
        metaText shouldContain "\"vocabSize\": 5"
        metaText shouldContain "\"embeddingDim\": 3"
        metaText shouldContain "\"format\": \"float32-le\""
        metaText shouldContain "\"byteOrder\": \"little-endian\""
        metaText shouldContain "\"checksum\":"
        metaText shouldContain "\"createdAt\":"
        metaText shouldContain "\"tesseraCompatible\": true"
    }

    "load fails with clear message when .bin is missing" {
        val bin = tempBinFile()
        val ex = shouldThrow<IllegalArgumentException> { EmbeddingTable.load(bin) }
        ex.message!! shouldContain "Embedding binary not found"
    }

    "load fails with clear message when .meta.json is missing" {
        val table = EmbeddingTable.create(vocabSize = 4, embeddingDim = 3, initializer = Initializer.zeros())
        val bin = tempBinFile()
        table.save(bin)
        File(bin.parentFile, bin.name + ".meta.json").delete()

        val ex = shouldThrow<IllegalArgumentException> { EmbeddingTable.load(bin) }
        ex.message!! shouldContain "Embedding metadata not found"
    }

    "load detects checksum mismatch when a single byte of the .bin is flipped" {
        val table = EmbeddingTable.create(
            vocabSize = 6,
            embeddingDim = 4,
            initializer = Initializer.uniformDefault(seed = 1L),
        )
        val bin = tempBinFile()
        table.save(bin)

        // Flip a byte deep in the float payload (past the 16-byte header)
        val bytes = bin.readBytes()
        val flipAt = EmbeddingFormat.HEADER_SIZE_BYTES + 5
        bytes[flipAt] = (bytes[flipAt].toInt() xor 0xFF).toByte()
        bin.writeBytes(bytes)

        val ex = shouldThrow<IllegalArgumentException> { EmbeddingTable.load(bin) }
        ex.message!! shouldContain "Checksum mismatch"
    }

    "load detects size mismatch when .bin is truncated" {
        val table = EmbeddingTable.create(
            vocabSize = 6,
            embeddingDim = 4,
            initializer = Initializer.uniformDefault(seed = 1L),
        )
        val bin = tempBinFile()
        table.save(bin)

        // Truncate the binary
        val bytes = bin.readBytes()
        bin.writeBytes(bytes.copyOf(bytes.size - 8))

        val ex = shouldThrow<IllegalArgumentException> { EmbeddingTable.load(bin) }
        ex.message!! shouldContain "Corrupted .bin"
    }

    "load rejects an unsupported metadata version" {
        val table = EmbeddingTable.create(vocabSize = 4, embeddingDim = 3, initializer = Initializer.zeros())
        val bin = tempBinFile()
        table.save(bin)
        val metaFile = File(bin.parentFile, bin.name + ".meta.json")
        metaFile.writeText(metaFile.readText().replace("\"version\": 1", "\"version\": 99"))

        val ex = shouldThrow<IllegalArgumentException> { EmbeddingTable.load(bin) }
        ex.message!! shouldContain "Unsupported metadata version"
    }

    "save(String) and load(String) round-trip" {
        val original = EmbeddingTable.create(vocabSize = 5, embeddingDim = 3, initializer = Initializer.constant(0.5f))
        val bin = tempBinFile()
        original.save(bin.absolutePath)
        val loaded = EmbeddingTable.load(bin.absolutePath)
        for (id in 0 until 5) {
            loaded.get(id).all { it == 0.5f } shouldBe true
        }
    }

    "binary header contains magic 'MOSC' at offset 0" {
        val table = EmbeddingTable.create(vocabSize = 2, embeddingDim = 2, initializer = Initializer.zeros())
        val bin = tempBinFile()
        table.save(bin)
        val bytes = bin.readBytes()
        bytes[0] shouldBe 'M'.code.toByte()
        bytes[1] shouldBe 'O'.code.toByte()
        bytes[2] shouldBe 'S'.code.toByte()
        bytes[3] shouldBe 'C'.code.toByte()
    }
})
