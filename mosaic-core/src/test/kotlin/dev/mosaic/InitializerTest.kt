package dev.mosaic

import io.kotest.core.spec.style.StringSpec
import io.kotest.matchers.floats.shouldBeGreaterThanOrEqual
import io.kotest.matchers.floats.shouldBeLessThanOrEqual
import io.kotest.matchers.shouldBe
import kotlin.math.sqrt

class InitializerTest : StringSpec({

    "uniformDefault stays within [-0.5/dim, +0.5/dim]" {
        val dim = 32
        val target = FloatArray(dim)
        val init = Initializer.uniformDefault(seed = 1L)
        repeat(50) { row ->
            init.fill(target, row)
            val bound = 0.5f / dim
            target.forEach {
                it shouldBeGreaterThanOrEqual -bound
                it shouldBeLessThanOrEqual bound
            }
        }
    }

    "uniform with explicit bound stays within [-bound, +bound]" {
        val bound = 2f
        val target = FloatArray(16)
        val init = Initializer.uniform(bound, seed = 7L)
        repeat(50) { row ->
            init.fill(target, row)
            target.forEach {
                it shouldBeGreaterThanOrEqual -bound
                it shouldBeLessThanOrEqual bound
            }
        }
    }

    "same seed produces identical output" {
        val a = FloatArray(8)
        val b = FloatArray(8)
        Initializer.uniform(1f, seed = 99L).fill(a, 0)
        Initializer.uniform(1f, seed = 99L).fill(b, 0)
        a.contentEquals(b) shouldBe true
    }

    "different seeds produce different output" {
        val a = FloatArray(8)
        val b = FloatArray(8)
        Initializer.uniform(1f, seed = 1L).fill(a, 0)
        Initializer.uniform(1f, seed = 2L).fill(b, 0)
        a.contentEquals(b) shouldBe false
    }

    "consecutive fill calls draw different random sequences" {
        val first = FloatArray(8)
        val second = FloatArray(8)
        val init = Initializer.uniform(1f, seed = 42L)
        init.fill(first, 0)
        init.fill(second, 1)
        first.contentEquals(second) shouldBe false
    }

    "xavier produces values within sqrt(6 / (fanIn + fanOut))" {
        val fanIn = 64
        val fanOut = 128
        val bound = sqrt(6.0 / (fanIn + fanOut)).toFloat()
        val target = FloatArray(32)
        Initializer.xavier(fanIn, fanOut, seed = 5L).fill(target, 0)
        target.forEach {
            it shouldBeGreaterThanOrEqual -bound
            it shouldBeLessThanOrEqual bound
        }
    }

    "he produces values within sqrt(6 / fanIn)" {
        val fanIn = 64
        val bound = sqrt(6.0 / fanIn).toFloat()
        val target = FloatArray(32)
        Initializer.he(fanIn, seed = 5L).fill(target, 0)
        target.forEach {
            it shouldBeGreaterThanOrEqual -bound
            it shouldBeLessThanOrEqual bound
        }
    }

    "zeros fills with 0.0f" {
        val target = FloatArray(10) { it.toFloat() }
        Initializer.zeros().fill(target, 0)
        target.all { it == 0f } shouldBe true
    }

    "constant fills with the given value" {
        val target = FloatArray(10)
        Initializer.constant(3.14f).fill(target, 0)
        target.all { it == 3.14f } shouldBe true
    }
})
