package dev.mosaic.cli

import kotlin.math.abs

internal object Format {

    /** Horizontal rule used between section header and body in human output. */
    const val SEPARATOR: String = "─────────────────────────────────"

    /** Formats a non-negative integer with thousands separators (e.g. 50000 → "50,000"). */
    fun number(n: Long): String {
        if (n == 0L) return "0"
        val sign = if (n < 0) "-" else ""
        val abs = abs(n).toString()
        val sb = StringBuilder(sign)
        val firstGroup = abs.length % 3
        if (firstGroup != 0) sb.append(abs, 0, firstGroup)
        var i = firstGroup
        while (i < abs.length) {
            if (sb.length > sign.length) sb.append(',')
            sb.append(abs, i, i + 3)
            i += 3
        }
        return sb.toString()
    }

    fun number(n: Int): String = number(n.toLong())

    /** Formats a byte count with binary units (B / KB / MB / GB). */
    fun bytes(n: Long): String {
        val units = listOf("B", "KB", "MB", "GB", "TB")
        var value = n.toDouble()
        var unitIdx = 0
        while (value >= UNIT_THRESHOLD && unitIdx < units.size - 1) {
            value /= UNIT_THRESHOLD
            unitIdx++
        }
        return if (unitIdx == 0) {
            "$n B"
        } else {
            "%.2f %s".format(value, units[unitIdx])
        }
    }

    /** Truncates a hex string to its first [keep] characters, suffixed with "...". */
    fun truncateHex(hex: String, keep: Int = HEX_TRUNCATE_DEFAULT): String =
        if (hex.length <= keep) hex else hex.substring(0, keep) + "..."

    private const val UNIT_THRESHOLD: Double = 1024.0
    private const val HEX_TRUNCATE_DEFAULT: Int = 16
}
