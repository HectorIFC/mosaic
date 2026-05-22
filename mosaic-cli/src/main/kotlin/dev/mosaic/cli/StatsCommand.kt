package dev.mosaic.cli

import dev.mosaic.EmbeddingTable
import dev.mosaic.VectorOps
import java.io.File
import kotlin.math.sqrt

internal object StatsCommand {

    fun run(args: List<String>): Int {
        if ("--help" in args || "-h" in args) {
            println(help())
            return 0
        }
        val parsed = Args.parse(args)
        val inputPath = parsed.requireString("--input", "-i")
        val file = File(inputPath)
        if (!file.exists()) throw RuntimeFailure("Embedding file not found: $inputPath")

        val table = try {
            EmbeddingTable.load(file)
        } catch (e: IllegalArgumentException) {
            throw RuntimeFailure(e.message ?: "Failed to load embedding table", e)
        }

        var minVal = Float.POSITIVE_INFINITY
        var maxVal = Float.NEGATIVE_INFINITY
        var sum = 0.0
        var sumSq = 0.0
        var totalValues = 0L

        var minNorm = Float.POSITIVE_INFINITY
        var maxNorm = Float.NEGATIVE_INFINITY
        var normSum = 0.0

        for (id in 0 until table.vocabSize) {
            val row = table.get(id)
            for (x in row) {
                if (x < minVal) minVal = x
                if (x > maxVal) maxVal = x
                sum += x.toDouble()
                sumSq += x.toDouble() * x.toDouble()
                totalValues++
            }
            val n = VectorOps.norm(row)
            if (n < minNorm) minNorm = n
            if (n > maxNorm) maxNorm = n
            normSum += n.toDouble()
        }

        val mean = sum / totalValues
        val variance = (sumSq / totalValues) - (mean * mean)
        val stddev = sqrt(if (variance < 0) 0.0 else variance)
        val meanNorm = normSum / table.vocabSize

        println("Statistics for ${file.name}")
        println(Format.SEPARATOR)
        println(
            "Total values:    ${Format.number(totalValues)} " +
                "(${Format.number(table.vocabSize)} × ${table.embeddingDim})",
        )
        println("Min:             ${"%.4f".format(minVal)}")
        println("Max:             ${"%.4f".format(maxVal)}")
        println("Mean:            ${"%.4f".format(mean)}")
        println("Std dev:         ${"%.4f".format(stddev)}")
        println("Mean row norm:   ${"%.4f".format(meanNorm)}")
        println("Min row norm:    ${"%.4f".format(minNorm)}")
        println("Max row norm:    ${"%.4f".format(maxNorm)}")
        return 0
    }

    private fun help(): String = """
        Compute distribution statistics for a saved embedding table.

        Usage:
          mosaic-cli stats --input PATH

        Required:
          --input, -i PATH      Path to the .bin file
    """.trimIndent()
}
