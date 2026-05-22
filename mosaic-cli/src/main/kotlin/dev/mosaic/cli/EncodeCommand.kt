package dev.mosaic.cli

import dev.mosaic.EmbeddingTable
import dev.mosaic.TesseraEmbeddings
import dev.tessera.BpeTokenizer
import java.io.File

internal object EncodeCommand {

    @Suppress("ThrowsCount", "TooGenericExceptionCaught")
    fun run(args: List<String>): Int {
        if ("--help" in args || "-h" in args) {
            println(help())
            return 0
        }
        val parsed = Args.parse(args)
        val embPath = parsed.requireString("--embeddings", "-e")
        val tokenizerPath = parsed.requireString("--tokenizer", "-t")
        val text = parsed.requireString("--text")
        val format = parsed.first("--format") ?: "pretty"

        val embFile = File(embPath)
        if (!embFile.exists()) throw RuntimeFailure("Embedding file not found: $embPath")
        val tokenizerFile = File(tokenizerPath)
        if (!tokenizerFile.exists()) throw RuntimeFailure("Tokenizer not found: $tokenizerPath")

        val table = try {
            EmbeddingTable.load(embFile)
        } catch (e: IllegalArgumentException) {
            throw RuntimeFailure(e.message ?: "Failed to load embedding table", e)
        }
        val tokenizer = try {
            BpeTokenizer.load(tokenizerFile)
        } catch (e: Exception) {
            throw RuntimeFailure("Failed to load tokenizer: ${e.message}", e)
        }
        if (tokenizer.vocabSize != table.vocabSize) {
            throw RuntimeFailure(
                "Vocab size mismatch: tokenizer=${tokenizer.vocabSize}, embeddings=${table.vocabSize}",
            )
        }

        val pipeline = TesseraEmbeddings(tokenizer, table)
        val ids = tokenizer.encode(text)
        val vectors = pipeline.encode(text)

        when (format) {
            "pretty" -> printPretty(text, ids, vectors)
            "json" -> printJson(ids, vectors)
            "csv" -> printCsv(ids, vectors)
            else -> throw UsageError("Unknown --format '$format'. Valid: pretty, json, csv")
        }
        return 0
    }

    private fun printPretty(text: String, ids: IntArray, vectors: Array<FloatArray>) {
        println("Encoded '$text' → ${vectors.size} vectors of dim ${vectors.firstOrNull()?.size ?: 0}")
        println(Format.SEPARATOR)
        for ((i, v) in vectors.withIndex()) {
            val preview = v.take(PREVIEW_VALUES).joinToString { "%+.4f".format(it) }
            val ellipsis = if (v.size > PREVIEW_VALUES) ", ..." else ""
            println("  token %-6d  [$preview$ellipsis]".format(ids[i]))
        }
    }

    private fun printJson(ids: IntArray, vectors: Array<FloatArray>) {
        val sb = StringBuilder()
        sb.append("{\n")
        sb.append("  \"ids\": [").append(ids.joinToString()).append("],\n")
        sb.append("  \"vectors\": [\n")
        for ((i, v) in vectors.withIndex()) {
            sb.append("    [").append(v.joinToString())
            sb.append(if (i == vectors.lastIndex) "]\n" else "],\n")
        }
        sb.append("  ]\n")
        sb.append("}")
        println(sb.toString())
    }

    private fun printCsv(ids: IntArray, vectors: Array<FloatArray>) {
        for ((i, v) in vectors.withIndex()) {
            print(ids[i])
            for (x in v) {
                print(',')
                print(x)
            }
            println()
        }
    }

    private fun help(): String = """
        Tokenize text and emit the resulting embedding vectors.

        Usage:
          mosaic-cli encode --tokenizer PATH --embeddings PATH --text "..." [--format FORMAT]

        Required:
          --embeddings, -e PATH   Path to the .bin file
          --tokenizer, -t PATH    Tessera tokenizer JSON
          --text "..."            Input text

        Optional:
          --format FORMAT         pretty (default), json, csv
    """.trimIndent()

    private const val PREVIEW_VALUES: Int = 4
}
