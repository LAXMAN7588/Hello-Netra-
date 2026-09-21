package com.example.objectdetection

import android.graphics.RectF

/**
 * Data class representing a single detected object in coordinates matching the source camera frame.
 */
data class Detection(
    val classId: Int,
    val label: String,
    val confidence: Float,
    val left: Float,
    val top: Float,
    val right: Float,
    val bottom: Float
) {
    val width: Float get() = right - left
    val height: Float get() = bottom - top
    val rectF: RectF get() = RectF(left, top, right, bottom)
}