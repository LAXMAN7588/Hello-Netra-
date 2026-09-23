package com.example.objectdetection.safety

/**
 * Encapsulates a single 4x4 (16-zone) distance frame from the VL53L5CX Time-of-Flight sensor.
 *
 * Grid Layout (Looking forward from the device):
 * [ Zone 0  ] [ Zone 1  ] [ Zone 2  ] [ Zone 3  ]  (Top row / Upper)
 * [ Zone 4  ] [ Zone 5  ] [ Zone 6  ] [ Zone 7  ]
 * [ Zone 8  ] [ Zone 9  ] [ Zone 10 ] [ Zone 11 ]
 * [ Zone 12 ] [ Zone 13 ] [ Zone 14 ] [ Zone 15 ]  (Bottom row / Ground)
 *
 * Distances are represented in millimeters.
 * Value 0 represents no valid target, out of range, or invalid sensor status.
 */
data class TofFrame(
    val timestampMs: Long,
    val distances: IntArray // Array of 16 distances in mm (0 = invalid)
) {
    companion object {
        const val NUM_ZONES = 16
        const val GRID_SIZE = 4
        const val MIN_VALID_DISTANCE_MM = 20
        const val MAX_VALID_DISTANCE_MM = 4000
    }

    init {
        require(distances.size == NUM_ZONES) { "TofFrame must contain exactly 16 zone distances" }
    }

    fun getDistance(row: Int, col: Int): Int {
        if (row !in 0 until GRID_SIZE || col !in 0 until GRID_SIZE) return 0
        return distances[row * GRID_SIZE + col]
    }

    fun isValidZone(index: Int): Boolean {
        if (index !in 0 until NUM_ZONES) return false
        val d = distances[index]
        return d in MIN_VALID_DISTANCE_MM..MAX_VALID_DISTANCE_MM
    }

    fun validZoneCount(): Int {
        var count = 0
        for (d in distances) {
            if (d in MIN_VALID_DISTANCE_MM..MAX_VALID_DISTANCE_MM) {
                count++
            }
        }
        return count
    }

    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (javaClass != other?.javaClass) return false
        other as TofFrame
        if (timestampMs != other.timestampMs) return false
        if (!distances.contentEquals(other.distances)) return false
        return true
    }

    override fun hashCode(): Int {
        var result = timestampMs.hashCode()
        result = 31 * result + distances.contentHashCode()
        return result
    }

    override fun toString(): String {
        val sb = StringBuilder()
        sb.append("TofFrame(ts=$timestampMs, valid=${validZoneCount()}/16):\n")
        for (r in 0 until GRID_SIZE) {
            sb.append("  [")
            for (c in 0 until GRID_SIZE) {
                val d = getDistance(r, c)
                sb.append("%5d".format(d))
                if (c < GRID_SIZE - 1) sb.append(", ")
            }
            sb.append("]\n")
        }
        return sb.toString()
    }
}
