package dev.mosaic

import java.util.Random
import kotlin.math.sqrt

/**
 * Strategy for populating an embedding row with initial values. Implementations
 * fill the caller-provided [target] buffer; [row] is supplied for stateful
 * initializers that want to vary behavior per row (most do not).
 *
 * All built-in initializers exposed via the companion object accept a `seed`
 * parameter for deterministic output, which is essential for reproducible
 * tests and debugging.
 */
public fun interface Initializer {

    /** Fills [target] with initial values for the given [row] index. */
    public fun fill(target: FloatArray, row: Int)

    public companion object {

        /**
         * Uniform distribution in `[-0.5/dim, +0.5/dim]`, matching the default
         * of PyTorch's `nn.Embedding`. Recommended starting point.
         */
        public fun uniformDefault(seed: Long = DEFAULT_SEED): Initializer {
            val random = Random(seed)
            return Initializer { target, _ ->
                val bound = HALF / target.size
                fillUniform(target, bound, random)
            }
        }

        /** Uniform distribution in `[-bound, +bound]`. */
        public fun uniform(bound: Float, seed: Long = DEFAULT_SEED): Initializer {
            val random = Random(seed)
            return Initializer { target, _ -> fillUniform(target, bound, random) }
        }

        /**
         * Xavier/Glorot uniform: bound = `sqrt(6 / (fanIn + fanOut))`. Suitable
         * for layers using symmetric activations (tanh, sigmoid).
         */
        public fun xavier(fanIn: Int, fanOut: Int, seed: Long = DEFAULT_SEED): Initializer {
            val bound = sqrt(XAVIER_NUMERATOR / (fanIn + fanOut).toDouble()).toFloat()
            return uniform(bound, seed)
        }

        /**
         * He uniform: bound = `sqrt(6 / fanIn)`. Suitable for layers using
         * ReLU-family activations.
         */
        public fun he(fanIn: Int, seed: Long = DEFAULT_SEED): Initializer {
            val bound = sqrt(XAVIER_NUMERATOR / fanIn.toDouble()).toFloat()
            return uniform(bound, seed)
        }

        /** All zeros. Rarely useful for embeddings but provided for completeness. */
        public fun zeros(): Initializer = Initializer { target, _ -> target.fill(0f) }

        /** Every element set to [value]. */
        public fun constant(value: Float): Initializer = Initializer { target, _ -> target.fill(value) }

        private const val DEFAULT_SEED: Long = 42L
        private const val HALF: Float = 0.5f
        private const val XAVIER_NUMERATOR: Double = 6.0

        private fun fillUniform(target: FloatArray, bound: Float, random: Random) {
            val range = bound * 2
            for (i in target.indices) {
                target[i] = random.nextFloat() * range - bound
            }
        }
    }
}
