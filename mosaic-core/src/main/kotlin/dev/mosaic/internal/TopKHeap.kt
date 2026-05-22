package dev.mosaic.internal

import dev.mosaic.Similarity

/**
 * Fixed-capacity min-heap over `(id, score)` pairs ordered by `score`. Used
 * by `mostSimilar` to track the top-K highest scores in a single O(N log K)
 * pass instead of sorting the full vocabulary.
 *
 * The heap root is the *smallest* score currently held; a new candidate
 * replaces the root iff its score exceeds the root's. After all candidates
 * have been offered, [toSortedListDescending] returns the contents sorted
 * from highest to lowest score.
 */
internal class TopKHeap(private val capacity: Int) {

    init {
        require(capacity >= 0) { "TopKHeap capacity must be ≥ 0, got $capacity" }
    }

    private val ids = IntArray(capacity)
    private val scores = FloatArray(capacity)
    private var heapSize = 0

    val size: Int get() = heapSize

    /** Offers a candidate. Inserts if the heap has room, otherwise replaces the root iff [score] is larger. */
    fun offer(id: Int, score: Float) {
        if (capacity == 0) return
        if (heapSize < capacity) {
            ids[heapSize] = id
            scores[heapSize] = score
            heapSize++
            siftUp(heapSize - 1)
        } else if (score > scores[0]) {
            ids[0] = id
            scores[0] = score
            siftDown(0)
        }
    }

    /** Returns the heap contents as a list of [Similarity] sorted by score descending. */
    fun toSortedListDescending(): List<Similarity> {
        val result = ArrayList<Similarity>(heapSize)
        for (i in 0 until heapSize) {
            result.add(Similarity(ids[i], scores[i]))
        }
        result.sortByDescending { it.score }
        return result
    }

    private fun siftUp(startIndex: Int) {
        var i = startIndex
        while (i > 0) {
            val parent = (i - 1) / 2
            if (scores[i] >= scores[parent]) return
            swap(i, parent)
            i = parent
        }
    }

    private fun siftDown(startIndex: Int) {
        var i = startIndex
        while (true) {
            val left = 2 * i + 1
            val right = 2 * i + 2
            var smallest = i
            if (left < heapSize && scores[left] < scores[smallest]) smallest = left
            if (right < heapSize && scores[right] < scores[smallest]) smallest = right
            if (smallest == i) return
            swap(i, smallest)
            i = smallest
        }
    }

    private fun swap(a: Int, b: Int) {
        val tmpId = ids[a]
        ids[a] = ids[b]
        ids[b] = tmpId
        val tmpScore = scores[a]
        scores[a] = scores[b]
        scores[b] = tmpScore
    }
}
