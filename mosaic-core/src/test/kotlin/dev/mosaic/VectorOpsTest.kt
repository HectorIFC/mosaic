package dev.mosaic

import io.kotest.core.spec.style.StringSpec
import io.kotest.matchers.floats.plusOrMinus
import io.kotest.matchers.shouldBe
import io.kotest.matchers.shouldNotBe
import io.kotest.matchers.throwable.shouldHaveMessage

private const val EPS = 1e-6f

class VectorOpsTest : StringSpec({

    "dotProduct: known values" {
        val a = floatArrayOf(1f, 2f, 3f)
        val b = floatArrayOf(4f, -5f, 6f)
        VectorOps.dotProduct(a, b) shouldBe (4f - 10f + 18f).plusOrMinus(EPS)
    }

    "dotProduct: dimension mismatch throws" {
        val ex = runCatching { VectorOps.dotProduct(FloatArray(3), FloatArray(4)) }.exceptionOrNull()
        ex shouldNotBe null
        ex!! shouldHaveMessage "Vector dimension mismatch: 3 vs 4"
    }

    "norm: known values" {
        val v = floatArrayOf(3f, 4f)
        VectorOps.norm(v) shouldBe 5f.plusOrMinus(EPS)
    }

    "norm: zero vector is 0.0f" {
        VectorOps.norm(FloatArray(8)) shouldBe 0f
    }

    "cosineSimilarity(v, v) is ~1.0 for non-zero vector" {
        val v = floatArrayOf(0.5f, -0.3f, 1.7f, 0.0f)
        VectorOps.cosineSimilarity(v, v) shouldBe 1f.plusOrMinus(EPS)
    }

    "cosineSimilarity(v, -v) is ~-1.0" {
        val v = floatArrayOf(0.5f, -0.3f, 1.7f, 0.0f)
        val neg = FloatArray(v.size) { -v[it] }
        VectorOps.cosineSimilarity(v, neg) shouldBe (-1f).plusOrMinus(EPS)
    }

    "cosineSimilarity is symmetric" {
        val a = floatArrayOf(1f, 2f, 3f, 4f)
        val b = floatArrayOf(-0.5f, 0.25f, 1.5f, -2f)
        val ab = VectorOps.cosineSimilarity(a, b)
        val ba = VectorOps.cosineSimilarity(b, a)
        ab shouldBe ba.plusOrMinus(EPS)
    }

    "cosineSimilarity returns 0.0f when either vector has zero norm" {
        val v = floatArrayOf(1f, 2f, 3f)
        val zero = FloatArray(3)
        VectorOps.cosineSimilarity(v, zero) shouldBe 0f
        VectorOps.cosineSimilarity(zero, v) shouldBe 0f
        VectorOps.cosineSimilarity(zero, zero) shouldBe 0f
    }

    "normalize returns unit-length vector" {
        val v = floatArrayOf(3f, 4f)
        val n = VectorOps.normalize(v)
        VectorOps.norm(n) shouldBe 1f.plusOrMinus(EPS)
    }

    "normalize does not mutate the input" {
        val v = floatArrayOf(3f, 4f)
        val original = v.copyOf()
        VectorOps.normalize(v)
        v.contentEquals(original) shouldBe true
    }

    "normalize on zero vector returns a fresh zero copy (no NaN)" {
        val zero = FloatArray(5)
        val result = VectorOps.normalize(zero)
        result.contentEquals(FloatArray(5)) shouldBe true
        (result === zero) shouldBe false
    }

    "normalizeInPlace mutates the array" {
        val v = floatArrayOf(0f, 5f, 0f)
        VectorOps.normalizeInPlace(v)
        v[1] shouldBe 1f.plusOrMinus(EPS)
    }

    "normalizeInPlace on zero vector is a no-op (no NaN)" {
        val zero = FloatArray(4)
        VectorOps.normalizeInPlace(zero)
        zero.all { it == 0f } shouldBe true
    }
})
