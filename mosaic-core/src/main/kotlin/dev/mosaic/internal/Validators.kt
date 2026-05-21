package dev.mosaic.internal

internal fun requireValidId(id: Int, vocabSize: Int) {
    require(id in 0 until vocabSize) {
        "Token ID $id is out of range [0, $vocabSize)"
    }
}

internal fun requirePositiveDim(value: Int, name: String) {
    require(value > 0) { "$name must be positive, got $value" }
}

internal fun requireVectorDim(vector: FloatArray, expectedDim: Int, name: String = "vector") {
    require(vector.size == expectedDim) {
        "$name has size ${vector.size}, expected $expectedDim"
    }
}

internal fun requireSameSize(a: FloatArray, b: FloatArray) {
    require(a.size == b.size) {
        "Vector dimension mismatch: ${a.size} vs ${b.size}"
    }
}
