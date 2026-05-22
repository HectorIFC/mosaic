package dev.mosaic.cli

import dev.mosaic.EmbeddingTable
import dev.mosaic.TesseraEmbeddings
import dev.tessera.BpeTokenizer
import java.io.File

internal object SimilarCommand {

    @Suppress("ThrowsCount", "TooGenericExceptionCaught")
    fun run(args: List<String>): Int {
        if ("--help" in args || "-h" in args) {
            println(help())
            return 0
        }
        val parsed = Args.parse(args)
        val embPath = parsed.requireString("--embeddings", "-e")
        val topK = parsed.optionalInt(DEFAULT_TOP_K, "--top-k", "-k")
        val embFile = File(embPath)
        if (!embFile.exists()) throw RuntimeFailure("Embedding file not found: $embPath")

        val tokenIdArg = parsed.first("--id")
        val textArg = parsed.first("--text")
        val tokenizerPath = parsed.first("--tokenizer", "-t")

        if (tokenIdArg == null && textArg == null) {
            throw UsageError("Either --id or --text (with --tokenizer) is required")
        }
        if (tokenIdArg != null && textArg != null) {
            throw UsageError("--id and --text are mutually exclusive")
        }

        val table = try {
            EmbeddingTable.load(embFile)
        } catch (e: IllegalArgumentException) {
            throw RuntimeFailure(e.message ?: "Failed to load embedding table", e)
        }

        return if (tokenIdArg != null) {
            runById(table, tokenIdArg, topK)
        } else {
            val tp = tokenizerPath
                ?: throw UsageError("--text requires --tokenizer pointing to a Tessera JSON file")
            runByText(table, tp, textArg!!, topK)
        }
    }

    private fun runById(table: EmbeddingTable, idArg: String, topK: Int): Int {
        val id = idArg.toIntOrNull()
            ?: throw UsageError("--id must be an integer, got '$idArg'")
        if (id !in 0 until table.vocabSize) {
            throw UsageError("--id $id out of range [0, ${table.vocabSize})")
        }
        val results = table.mostSimilar(id = id, topK = topK, includeSelf = true)
        println("Top $topK most similar to token $id:")
        println(Format.SEPARATOR)
        for ((rank, sim) in results.withIndex()) {
            println(
                "  %2d. token %-6d  score: %+.4f".format(rank + 1, sim.id, sim.score),
            )
        }
        return 0
    }

    @Suppress("TooGenericExceptionCaught")
    private fun runByText(table: EmbeddingTable, tokenizerPath: String, text: String, topK: Int): Int {
        val tokenizerFile = File(tokenizerPath)
        if (!tokenizerFile.exists()) throw RuntimeFailure("Tokenizer not found: $tokenizerPath")
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
        val pooled = pipeline.encodeMeanPooled(text)
        val ids = tokenizer.encode(text)

        val results = table.mostSimilar(query = pooled, topK = topK)
        println("Top $topK most similar to '$text' (${ids.size} tokens: ${ids.toList()}):")
        println(Format.SEPARATOR)
        for ((rank, sim) in results.withIndex()) {
            println(
                "  %2d. token %-6d  score: %+.4f".format(rank + 1, sim.id, sim.score),
            )
        }
        return 0
    }

    private fun help(): String = """
        Top-K most similar tokens to a query (by token ID or by text).

        Usage:
          mosaic-cli similar --embeddings PATH --id N [--top-k K]
          mosaic-cli similar --embeddings PATH --tokenizer PATH --text "..." [--top-k K]

        Required:
          --embeddings, -e PATH   Path to the .bin file

        One of:
          --id N                  Query by token ID
          --text "..."            Query by text (mean-pooled vector); requires --tokenizer

        Optional:
          --tokenizer, -t PATH    Tessera tokenizer JSON (required with --text)
          --top-k, -k K           Number of results (default $DEFAULT_TOP_K)
    """.trimIndent()

    private const val DEFAULT_TOP_K: Int = 10
}
