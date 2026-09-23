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

    // Distinct vibrant colors for the 13 classes
    private val classColors = intArrayOf(
        Color.rgb(244, 67, 54),   // 0: accident (Red)
        Color.rgb(255, 152, 0),  // 1: auto (Orange)
        Color.rgb(255, 235, 59),  // 2: bicycle (Yellow)
        Color.rgb(76, 175, 80),   // 3: bus (Green)
        Color.rgb(33, 150, 243),  // 4: car (Blue)
        Color.rgb(156, 39, 176),  // 5: chair (Purple)
        Color.rgb(233, 30, 99),   // 6: emergency-responder (Deep Pink)
        Color.rgb(0, 188, 212),   // 7: emergency-vehicle (Cyan)
        Color.rgb(139, 195, 74),  // 8: laptop (Light Green)
        Color.rgb(255, 87, 34),   // 9: person (Deep Orange)
        Color.rgb(63, 81, 181),   // 10: truck (Indigo)
        Color.rgb(0, 150, 136),   // 11: two wheeler (Teal)
        Color.rgb(121, 85, 72)    // 12: wrecked-vehicle (Brown)
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

        // Maintain aspect fit alignment matching the ImageView's fitCenter scaling
        val viewW = width.toFloat()
        val viewH = height.toFloat()

        val scale = minOf(viewW / sourceWidth, viewH / sourceHeight)
        val offsetX = (viewW - sourceWidth * scale) / 2f
        val offsetY = (viewH - sourceHeight * scale) / 2f

        for (detection in detections) {
            val color = classColors[detection.classId % classColors.size]
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