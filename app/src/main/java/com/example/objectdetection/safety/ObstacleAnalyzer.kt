package com.example.objectdetection.safety

/**
 * Analyzes the 4x4 spatial depth grid from the VL53L5CX Time-of-Flight sensor.
 *
 * Sector Definitions:
 * - Columns 0: Left Column
 * - Columns 1..2: Center Corridor (Main forward path)
 * - Columns 3: Right Column
 * - Rows 0..1: Upper Sector (Chest / Head level obstacles)
 * - Rows 2..3: Lower Sector (Knee / Ground / Drop-off obstacles)
 *
 * NOTE: As per architecture rules, no hardcoded alarm thresholds are defined here.
 * This class extracts spatial sector telemetry for safety policy consumption.
 */
class ObstacleAnalyzer {

    data class SectorSummary(
        val minDistanceMm: Int,
        val avgDistanceMm: Int,
        val validCount: Int,
        val totalCount: Int
    )

    data class SpatialAnalysis(
        val left: SectorSummary,
        val center: SectorSummary,
        val right: SectorSummary,
        val upper: SectorSummary,
        val lower: SectorSummary,
        val overallMinDistanceMm: Int,
        val totalValidZones: Int
    )

    /**
     * Performs spatial aggregation across all sectors of the 4x4 grid.
     */
    fun analyze(frame: TofFrame): SpatialAnalysis {
        val leftIndices = intArrayOf(0, 4, 8, 12)
        val centerIndices = intArrayOf(1, 2, 5, 6, 9, 10, 13, 14)
        val rightIndices = intArrayOf(3, 7, 11, 15)
        val upperIndices = intArrayOf(0, 1, 2, 3, 4, 5, 6, 7)
        val lowerIndices = intArrayOf(8, 9, 10, 11, 12, 13, 14, 15)

        val left = calculateSector(frame, leftIndices)
        val center = calculateSector(frame, centerIndices)
        val right = calculateSector(frame, rightIndices)
        val upper = calculateSector(frame, upperIndices)
        val lower = calculateSector(frame, lowerIndices)

        var minAll = 0
        var totalValid = 0

        for (i in 0 until TofFrame.NUM_ZONES) {
            if (frame.isValidZone(i)) {
                val d = frame.distances[i]
                totalValid++
                if (minAll == 0 || d < minAll) {
                    minAll = d
                }
            }
        }

        return SpatialAnalysis(
            left = left,
            center = center,
            right = right,
            upper = upper,
            lower = lower,
            overallMinDistanceMm = minAll,
            totalValidZones = totalValid
        )
    }

    private fun calculateSector(frame: TofFrame, indices: IntArray): SectorSummary {
        var min = 0
        var sum = 0
        var count = 0

        for (idx in indices) {
            if (frame.isValidZone(idx)) {
                val d = frame.distances[idx]
                count++
                sum += d
                if (min == 0 || d < min) {
                    min = d
                }
            }
        }

        val avg = if (count > 0) sum / count else 0
        return SectorSummary(
            minDistanceMm = min,
            avgDistanceMm = avg,
            validCount = count,
            totalCount = indices.size
        )
    }
}
