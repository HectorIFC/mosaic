package dev.mosaic.cli

import kotlin.system.exitProcess

private val helpFlags = setOf("--help", "-h", "help")

public fun main(args: Array<String>) {
    if (args.isEmpty() || args.first() in helpFlags) {
        printHelp()
        return
    }
    System.err.println("Unknown command: ${args.first()}")
    System.err.println()
    printHelp()
    exitProcess(1)
}

private fun printHelp() {
    println(
        """
        Mosaic CLI — embedding table inspection and debug

        Usage: mosaic-cli <command> [options]

        Commands:
          create     Create a new embedding table  (coming in Phase 4)
          inspect    Show metadata for an embedding file  (coming in Phase 4)
          stats      Detailed statistics for a matrix  (coming in Phase 4)
          similar    Top-K most similar tokens  (coming in Phase 4)
          encode     Pipeline text → vectors  (coming in Phase 4)

        Run 'mosaic-cli <command> --help' for command-specific options.
        """.trimIndent(),
    )
}
