package com.example.objectdetection

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.util.AttributeSet
import android.view.View

class DetectionOverlay @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null
) : View(context, attrs) {

    // Distinct vibrant color palette for generic dynamic classes
    private val classColors = intArrayOf(
        Color.rgb(244, 67, 54),   // 0: Red
        Color.rgb(255, 152, 0),  // 1: Orange
        Color.rgb(255, 235, 59),  // 2: Yellow
        Color.rgb(76, 175, 80),   // 3: Green
        Color.rgb(33, 150, 243),  // 4: Blue
        Color.rgb(156, 39, 176),  // 5: Purple
        Color.rgb(233, 30, 99),   // 6: Deep Pink
        Color.rgb(0, 188, 212),   // 7: Cyan
        Color.rgb(139, 195, 74),  // 8: Light Green
        Color.rgb(255, 87, 34),   // 9: Deep Orange
        Color.rgb(63, 81, 181),   // 10: Indigo
        Color.rgb(0, 150, 136),   // 11: Teal
        Color.rgb(121, 85, 72),   // 12: Brown
        Color.rgb(96, 125, 139),  // 13: Blue Grey
        Color.rgb(255, 193, 7),   // 14: Amber
        Color.rgb(103, 58, 183)   // 15: Deep Purple
    )

    private val boxPaint = Paint().apply {
        style = Paint.Style.STROKE
        strokeWidth = 6f
        isAntiAlias = true
    }

    private val textBgPaint = Paint().apply {
        style = Paint.Style.FILL
        isAntiAlias = true
    }

    private val textPaint = Paint().apply {
        style = Paint.Style.FILL
        color = Color.WHITE
        textSize = 36f
        isFakeBoldText = true
        isAntiAlias = true
    }

    private var detections: List<Detection> = emptyList()
    private var sourceWidth = 0
    private var sourceHeight = 0

    fun setDetections(
        newDetections: List<Detection>,
        imageWidth: Int,
        imageHeight: Int
    ) {
        this.detections = newDetections
        this.sourceWidth = imageWidth
        this.sourceHeight = imageHeight
        postInvalidate()
    }

    fun clearDetections() {
        this.detections = emptyList()
        postInvalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)

        if (sourceWidth <= 0 || sourceHeight <= 0 || detections.isEmpty()) return

        val viewW = width.toFloat()
        val viewH = height.toFloat()

        val scale = minOf(viewW / sourceWidth, viewH / sourceHeight)
        val offsetX = (viewW - sourceWidth * scale) / 2f
        val offsetY = (viewH - sourceHeight * scale) / 2f

        for (detection in detections) {
            val colorIndex = Math.floorMod(detection.classId, classColors.size)
            val color = classColors[colorIndex]
            boxPaint.color = color
            textBgPaint.color = color

            val left = detection.left * scale + offsetX
            val top = detection.top * scale + offsetY
            val right = detection.right * scale + offsetX
            val bottom = detection.bottom * scale + offsetY

            // Draw bounding box
            val boxRect = RectF(left, top, right, bottom)
            canvas.drawRoundRect(boxRect, 8f, 8f, boxPaint)

            // Format: "class name + confidence percentage" e.g., "person 92%"
            val confidencePct = (detection.confidence * 100).toInt()
            val labelText = "${detection.label} $confidencePct%"

            val textWidth = textPaint.measureText(labelText)
            val textHeight = 42f
            val padding = 8f

            val labelTop = if (top - textHeight - padding >= offsetY) {
                top - textHeight - padding
            } else {
                top
            }
            val labelBottom = labelTop + textHeight + padding

            val bgRect = RectF(left, labelTop, left + textWidth + (padding * 2), labelBottom)
            canvas.drawRoundRect(bgRect, 6f, 6f, textBgPaint)

            canvas.drawText(
                labelText,
                left + padding,
                labelBottom - padding - 4f,
                textPaint
            )
        }
    }
}