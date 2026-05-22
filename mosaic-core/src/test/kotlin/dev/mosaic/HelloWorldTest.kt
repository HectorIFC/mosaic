package dev.mosaic

import io.kotest.core.spec.style.StringSpec
import io.kotest.matchers.shouldBe

class HelloWorldTest : StringSpec({
    "mosaic-core module compiles and tests run" {
        val result = 1 + 1
        result shouldBe 2
    }
})
