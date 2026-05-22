package dev.mosaic.cli

import kotlin.system.exitProcess

private val helpFlags = setOf("--help", "-h", "help")

fun main(args: Array<String>) {
    val code = runCli(args)
    if (code != 0) exitProcess(code)
}

/**
 * Testable entry point. Returns:
 * - `0` on success
 * - `1` on usage errors (missing/invalid args, unknown command)
 * - `2` on runtime errors (file not found, corrupted file, etc.)
 *
 * Writes structured output to stdout; errors and help text to stderr only
 * when the invocation itself failed. `--help`-style invocations print to
 * stdout and return 0.
 */
fun runCli(args: Array<String>): Int {
    if (args.isEmpty() || args.first() in helpFlags) {
        printGlobalHelp()
        return 0
    }
    val command = args.first()
    val rest = args.drop(1)
    return when (command) {
        "create" -> dispatch { CreateCommand.run(rest) }
        "inspect" -> dispatch { InspectCommand.run(rest) }
        "stats" -> dispatch { StatsCommand.run(rest) }
        "similar" -> dispatch { SimilarCommand.run(rest) }
        "encode" -> dispatch { EncodeCommand.run(rest) }
        else -> {
            System.err.println("Unknown command: $command")
            System.err.println()
            printGlobalHelp(System.err)
            1
        }
    }
}

// The catch-all `Exception` arm is the safety net for the "no stack traces in
// normal use" contract — any uncaught throw inside a command
// becomes a friendly "Error: ..." line on stderr and exit code 2 instead of a
// Kotlin stack dump. detekt's TooGenericExceptionCaught is suppressed because
// this is exactly the right place to be generic.
@Suppress("TooGenericExceptionCaught")
private inline fun dispatch(block: () -> Int): Int = try {
    block()
} catch (e: UsageError) {
    System.err.println("Error: ${e.message}")
    1
} catch (e: RuntimeFailure) {
    System.err.println("Error: ${e.message}")
    2
} catch (e: Exception) {
    System.err.println("Error: ${e.message ?: "Unexpected failure (${e::class.simpleName})"}")
    2
}

private fun printGlobalHelp(stream: java.io.PrintStream = System.out) {
    stream.println(
        """
        Mosaic CLI — embedding table inspection and debug

        Usage: mosaic-cli <command> [options]

        Commands:
          create     Create a new embedding table and save it to disk
          inspect    Show metadata for a saved embedding file
          stats      Compute distribution statistics for a saved table
          similar    Top-K most similar tokens (by ID or by text)
          encode     Tokenize text and emit the resulting embedding vectors

        Run 'mosaic-cli <command> --help' for command-specific options.
        """.trimIndent(),
    )
}
