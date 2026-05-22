package dev.mosaic.cli

/**
 * Tiny manual argument parser. Accepts `--name value` and `-x value` pairs
 * plus boolean flags (e.g. `--help`). Negative number values like `-0.5` are
 * treated as values, not flag names — the heuristic is "a flag name starts
 * with `-` followed by a letter".
 *
 * The parser is intentionally permissive: it ignores positional arguments
 * (there are none in any Mosaic command) and lets each command query the
 * flags it cares about via [require], [optional], [int], etc.
 */
internal class Args(
    private val pairs: Map<String, String>,
    private val flags: Set<String>,
) {

    /** Returns the first value found among [names], or `null` if none provided. */
    fun first(vararg names: String): String? {
        for (n in names) {
            pairs[n]?.let { return it }
        }
        return null
    }

    fun requireString(vararg names: String): String =
        first(*names) ?: throw UsageError("Missing required argument: ${names.first()}")

    fun requireInt(vararg names: String): Int {
        val raw = requireString(*names)
        return raw.toIntOrNull()
            ?: throw UsageError("Argument ${names.first()} must be an integer, got '$raw'")
    }

    fun optionalInt(default: Int, vararg names: String): Int {
        val raw = first(*names) ?: return default
        return raw.toIntOrNull()
            ?: throw UsageError("Argument ${names.first()} must be an integer, got '$raw'")
    }

    fun optionalFloat(vararg names: String): Float? {
        val raw = first(*names) ?: return null
        return raw.toFloatOrNull()
            ?: throw UsageError("Argument ${names.first()} must be a number, got '$raw'")
    }

    fun optionalLong(default: Long, vararg names: String): Long {
        val raw = first(*names) ?: return default
        return raw.toLongOrNull()
            ?: throw UsageError("Argument ${names.first()} must be an integer, got '$raw'")
    }

    fun has(vararg names: String): Boolean = names.any { it in flags }

    companion object {
        fun parse(args: List<String>): Args {
            val pairs = mutableMapOf<String, String>()
            val flags = mutableSetOf<String>()
            var i = 0
            while (i < args.size) {
                val token = args[i]
                if (!isFlagName(token)) {
                    i++
                    continue
                }
                if (i + 1 < args.size && !isFlagName(args[i + 1])) {
                    pairs[token] = args[i + 1]
                    i += 2
                } else {
                    flags += token
                    i++
                }
            }
            return Args(pairs, flags)
        }

        private fun isFlagName(token: String): Boolean {
            if (token.length < 2 || !token.startsWith("-")) return false
            val nameStart = if (token.startsWith("--")) 2 else 1
            if (nameStart >= token.length) return false
            return token[nameStart].isLetter()
        }
    }
}

/** Thrown by commands when the user supplied bad arguments. Maps to exit code 1. */
internal class UsageError(message: String) : RuntimeException(message)

/** Thrown by commands when an input file is missing or malformed. Maps to exit code 2. */
internal class RuntimeFailure(message: String, cause: Throwable? = null) : RuntimeException(message, cause)
