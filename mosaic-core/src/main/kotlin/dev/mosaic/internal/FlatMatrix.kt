package dev.mosaic.internal

/**
 * Storage backing for a 2D float matrix laid out as a single contiguous
 * [FloatArray]. Row `r` occupies indices `[r*cols, (r+1)*cols)`.
 *
 * The flat layout is mandated — `Array<FloatArray>` would
 * add a level of indirection and hurt cache locality during top-K scans over
 * the whole matrix.
 */
internal class FlatMatrix(val rows: Int, val cols: Int) {

    init {
        requirePositiveDim(rows, "rows")
        requirePositiveDim(cols, "cols")
    }

    val data: FloatArray = FloatArray(rows * cols)

    /**
     * Secondary constructor that copies [initialData] into the freshly-allocated
     * backing array. The caller may mutate [initialData] freely afterwards.
     */
    constructor(rows: Int, cols: Int, initialData: FloatArray) : this(rows, cols) {
        require(initialData.size == rows * cols) {
            "initialData has size ${initialData.size}, expected ${rows * cols}"
        }
        System.arraycopy(initialData, 0, data, 0, data.size)
    }

    /** Returns a fresh copy of row [row]. */
    fun getRow(row: Int): FloatArray {
        val result = FloatArray(cols)
        System.arraycopy(data, row * cols, result, 0, cols)
        return result
    }

    /** Writes row [row] into the caller-provided [into] buffer (no allocation). */
    fun getRow(row: Int, into: FloatArray): FloatArray {
        System.arraycopy(data, row * cols, into, 0, cols)
        return into
    }

    /** Copies [vector] into row [row]. The source array may be mutated freely afterwards. */
    fun setRow(row: Int, vector: FloatArray) {
        System.arraycopy(vector, 0, data, row * cols, cols)
    }
}
