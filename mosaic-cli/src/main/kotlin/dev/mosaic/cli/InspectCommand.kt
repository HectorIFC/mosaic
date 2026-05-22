package dev.mosaic.cli

import dev.mosaic.EmbeddingFormat
import java.io.File

internal object InspectCommand {

    fun run(args: List<String>): Int {
        if ("--help" in args || "-h" in args) {
            println(help())
            return 0
        }
        val parsed = Args.parse(args)
        val inputPath = parsed.requireString("--input", "-i")
        val file = File(inputPath)
        if (!file.exists()) throw RuntimeFailure("Embedding file not found: $inputPath")

        val meta = try {
            EmbeddingFormat.readMetadata(file)
        } catch (e: IllegalArgumentException) {
            throw RuntimeFailure(e.message ?: "Failed to read metadata", e)
        }
        val checksumOk = try {
            EmbeddingFormat.verifyChecksum(file)
        } catch (e: IllegalArgumentException) {
            throw RuntimeFailure(e.message ?: "Failed to verify checksum", e)
        }

        println("Mosaic Embedding Table")
        println(Format.SEPARATOR)
        println("Version:             ${meta.version}")
        println("Vocab size:          ${Format.number(meta.vocabSize)}")
        println("Embedding dim:       ${meta.embeddingDim}")
        println("Format:              ${meta.format}")
        println("File size:           ${Format.bytes(file.length())}")
        val verdict = if (checksumOk) "valid ✓" else "INVALID ✗"
        println("Checksum:            ${Format.truncateHex(meta.checksum)} ($verdict)")
        println("Created:             ${meta.createdAt}")
        println("Tessera-compatible:  ${if (meta.tesseraCompatible) "yes" else "no"}")

        return if (checksumOk) 0 else 2
    }

    private fun help(): String = """
        Show metadata and checksum status for a saved embedding file.

        Usage:
          mosaic-cli inspect --input PATH

        Required:
          --input, -i PATH      Path to the .bin file (.meta.json sidecar must sit next to it)
    """.trimIndent()
}
