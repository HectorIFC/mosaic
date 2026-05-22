package dev.mosaic.cli

import dev.mosaic.EmbeddingTable
import dev.mosaic.Initializer
import java.io.File

internal object CreateCommand {

    fun run(args: List<String>): Int {
        if ("--help" in args || "-h" in args) {
            println(help())
            return 0
        }
        val parsed = Args.parse(args)
        val vocabSize = parsed.requireInt("--vocab-size")
        val dim = parsed.requireInt("--dim")
        val seed = parsed.optionalLong(DEFAULT_SEED, "--seed")
        val initializerName = parsed.first("--initializer") ?: "uniform"
        val outputPath = parsed.requireString("--output", "-o")

        require(vocabSize > 0) { throw UsageError("--vocab-size must be > 0, got $vocabSize") }
        require(dim > 0) { throw UsageError("--dim must be > 0, got $dim") }

        val initializer = buildInitializer(initializerName, dim, seed, parsed)

        val table = EmbeddingTable.create(
            vocabSize = vocabSize,
            embeddingDim = dim,
            initializer = initializer,
        )
        val outFile = File(outputPath)
        outFile.parentFile?.mkdirs()
        table.save(outFile)

        println("Created ${Format.number(vocabSize)} × $dim embedding table")
        println("  initializer: $initializerName (seed=$seed)")
        println("  saved to:    ${outFile.path}")
        return 0
    }

    private fun buildInitializer(name: String, dim: Int, seed: Long, parsed: Args): Initializer = when (name) {
        "uniform" -> {
            val bound = parsed.optionalFloat("--bound")
            if (bound != null) Initializer.uniform(bound, seed) else Initializer.uniformDefault(seed)
        }
        "xavier" -> Initializer.xavier(fanIn = dim, fanOut = dim, seed = seed)
        "he" -> Initializer.he(fanIn = dim, seed = seed)
        "zeros" -> Initializer.zeros()
        "constant" -> {
            val value = parsed.optionalFloat("--value")
                ?: throw UsageError("Initializer 'constant' requires --value")
            Initializer.constant(value)
        }
        else -> throw UsageError(
            "Unknown initializer '$name'. Valid: uniform, xavier, he, zeros, constant",
        )
    }

    private fun help(): String = """
        Create a new embedding table and save it to disk.

        Usage:
          mosaic-cli create --vocab-size N --dim D --output PATH [options]

        Required:
          --vocab-size N        Number of token rows
          --dim D               Embedding dimension
          --output, -o PATH     Output .bin path (sidecar .meta.json written next to it)

        Optional:
          --initializer NAME    uniform (default) | xavier | he | zeros | constant
          --seed N              Seed for random initializers (default 42)
          --bound F             Bound for uniform initializer (default 0.5/dim)
          --value F             Required when --initializer constant
    """.trimIndent()

    private const val DEFAULT_SEED: Long = 42L
}
