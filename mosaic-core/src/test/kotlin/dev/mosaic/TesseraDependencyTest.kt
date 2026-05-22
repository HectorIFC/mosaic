package dev.mosaic

import dev.tessera.BpeTokenizer
import io.kotest.core.spec.style.StringSpec
import io.kotest.matchers.shouldBe

/**
 * Validates that the Tessera dependency resolves through JitPack and that
 * its public API is importable from `mosaic-core`. This guarantees the
 * external dependency wiring works end-to-end before the Phase 1 integration
 * (`TesseraEmbeddings`) is written.
 */
class TesseraDependencyTest : StringSpec({
    "Tessera BpeTokenizer class is resolvable via JitPack" {
        BpeTokenizer::class.simpleName shouldBe "BpeTokenizer"
    }
})
