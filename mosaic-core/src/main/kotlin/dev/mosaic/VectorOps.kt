package dev.mosaic

import dev.mosaic.internal.requireSameSize
import kotlin.math.sqrt

/**
 * Stateless vector operations on [FloatArray]. All operations assume vectors
 * share the same dimension; mismatches throw [IllegalArgumentException].
 *
 * Acumulation is performed in [Double] internally and the result is narrowed
 * back to [Float] on return. This avoids the precision drift that would happen
 * if many small `Float` additions were summed directly.
 */
public object VectorOps {

    /** Returns `Σ a[i] * b[i]`. */
    public fun dotProduct(a: FloatArray, b: FloatArray): Float {
        requireSameSize(a, b)
        var sum = 0.0
        for (i in a.indices) {
            sum += a[i].toDouble() * b[i].toDouble()
        }
        return sum.toFloat()
    }

    /** Returns `sqrt(Σ v[i]²)`. */
    public fun norm(v: FloatArray): Float {
        var sum = 0.0
        for (i in v.indices) {
            val x = v[i].toDouble()
            sum += x * x
        }
        return sqrt(sum).toFloat()
    }

    /**
     * Cosine similarity in `[-1, 1]`. Returns `0.0f` when either vector has
     * zero norm — documented as the convention for ill-defined cases rather
     * than `NaN`, which would be surprising to callers.
     */
    public fun cosineSimilarity(a: FloatArray, b: FloatArray): Float {
        requireSameSize(a, b)
        val normA = norm(a)
        val normB = norm(b)
        if (normA == 0f || normB == 0f) return 0f
        return dotProduct(a, b) / (normA * normB)
    }

    /**
     * Returns a normalized copy. For a zero vector returns a fresh copy of
     * the input (no division by zero).
     */
    public fun normalize(v: FloatArray): FloatArray {
        val result = v.copyOf()
        normalizeInPlace(result)
        return result
    }

    /** Normalizes [v] in place. No-op for a zero vector. */
    public fun normalizeInPlace(v: FloatArray) {
        val n = norm(v)
        if (n == 0f) return
        val invN = 1f / n
        for (i in v.indices) {
            v[i] = v[i] * invN
        }
    }
}
