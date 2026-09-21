package com.example.objectdetection

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.Paint
import androidx.camera.core.ImageProxy
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer

/**
 * Result data class containing the preprocessed tensor data and transformation parameters
 * needed to map bounding boxes back to the original camera frame.
 */
data class PreprocessResult(
    val floatBuffer: FloatBuffer,
    val shape: LongArray,
    val scale: Float,
    val padX: Float,
    val padY: Float,
    val sourceWidth: Int,
    val sourceHeight: Int
)

class ImagePreprocessor(
    val targetWidth: Int = 640,
    val targetHeight: Int = 640
) {
    private val letterboxPaint = Paint(Paint.FILTER_BITMAP_FLAG)
    private val backgroundPaint = Paint().apply {
        color = Color.rgb(114, 114, 114) // Standard YOLO padding color
        style = Paint.Style.FILL
    }

    // Reusable buffers to minimize memory allocations per frame
    private var letterboxBitmap: Bitmap = Bitmap.createBitmap(targetWidth, targetHeight, Bitmap.Config.ARGB_8888)
    private var letterboxCanvas: Canvas = Canvas(letterboxBitmap)
    private var pixels = IntArray(targetWidth * targetHeight)
    private var floatBuffer: FloatBuffer = ByteBuffer
        .allocateDirect(1 * 3 * targetWidth * targetHeight * 4)
        .order(ByteOrder.nativeOrder())
        .asFloatBuffer()

    /**
     * Converts a CameraX ImageProxy to a rotated Bitmap and prepares a letterboxed 1x3x640x640 CHW FloatBuffer.
     */
    fun preprocess(imageProxy: ImageProxy): PreprocessResult {
        // 1. Convert ImageProxy to Bitmap
        val bitmap = imageProxy.toBitmap()
        val rotationDegrees = imageProxy.imageInfo.rotationDegrees

        // 2. Rotate Bitmap if needed to match portrait/device display
        val rotatedBitmap = if (rotationDegrees != 0) {
            val matrix = Matrix().apply { postRotate(rotationDegrees.toFloat()) }
            val rotated = Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
            bitmap.recycle()
            rotated
        } else {
            bitmap
        }

        val srcWidth = rotatedBitmap.width
        val srcHeight = rotatedBitmap.height

        // 3. Calculate YOLO letterbox scale and padding
        val scale = minOf(
            targetWidth.toFloat() / srcWidth,
            targetHeight.toFloat() / srcHeight
        )
        val scaledWidth = srcWidth * scale
        val scaledHeight = srcHeight * scale
        val padX = (targetWidth - scaledWidth) / 2.0f
        val padY = (targetHeight - scaledHeight) / 2.0f

        // 4. Draw padded letterbox image
        letterboxCanvas.drawRect(0f, 0f, targetWidth.toFloat(), targetHeight.toFloat(), backgroundPaint)
        val drawMatrix = Matrix().apply {
            postScale(scale, scale)
            postTranslate(padX, padY)
        }
        letterboxCanvas.drawBitmap(rotatedBitmap, drawMatrix, letterboxPaint)
        rotatedBitmap.recycle()

        // 5. Extract RGB pixels and normalize to [0.0, 1.0] in CHW layout (1, 3, 640, 640)
        letterboxBitmap.getPixels(pixels, 0, targetWidth, 0, 0, targetWidth, targetHeight)
        floatBuffer.clear()

        val numPixels = targetWidth * targetHeight
        val rOffset = 0
        val gOffset = numPixels
        val bOffset = numPixels * 2

        // Write directly to FloatBuffer in CHW format
        for (i in 0 until numPixels) {
            val pixel = pixels[i]
            val r = ((pixel shr 16) and 0xFF) / 255.0f
            val g = ((pixel shr 8) and 0xFF) / 255.0f
            val b = (pixel and 0xFF) / 255.0f

            floatBuffer.put(rOffset + i, r)
            floatBuffer.put(gOffset + i, g)
            floatBuffer.put(bOffset + i, b)
        }
        floatBuffer.position(0)

        return PreprocessResult(
            floatBuffer = floatBuffer,
            shape = longArrayOf(1, 3, targetWidth.toLong(), targetHeight.toLong()),
            scale = scale,
            padX = padX,
            padY = padY,
            sourceWidth = srcWidth,
            sourceHeight = srcHeight
        )
    }

    fun release() {
        if (!letterboxBitmap.isRecycled) {
            letterboxBitmap.recycle()
        }
    }
}
