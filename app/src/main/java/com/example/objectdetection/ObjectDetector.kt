package com.example.objectdetection

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.os.SystemClock
import android.util.Log
import java.io.File
import java.io.FileOutputStream
import java.nio.FloatBuffer
import java.util.Collections

class ObjectDetector(
    private val context: Context,
    var confidenceThreshold: Float = 0.40f,
    var iouThreshold: Float = 0.45f
) {
    companion object {
        private const val TAG = "ObjectDetector"

        // 13 strictly ordered classes
        val LABELS = arrayOf(
            "accident",
            "auto",
            "bicycle",
            "bus",
            "car",
            "chair",
            "emergency-responder",
            "emergency-vehicle",
            "laptop",
            "person",
            "truck",
            "two wheeler",
            "wrecked-vehicle"
        )
        const val NUM_CLASSES = 13
        const val NUM_ANCHORS = 8400
        const val NUM_CHANNELS = 49 // 4 box coords + 13 classes + 32 mask coefficients
    }

    private var ortEnv: OrtEnvironment = OrtEnvironment.getEnvironment()
    private var ortSession: OrtSession? = null
    private var inputName: String = "images"

    init {
        loadModel()
    }

    private fun loadModel() {
        try {
            // Find model asset name (supporting both spellings)
            val assetList = context.assets.list("") ?: emptyArray()
            val modelName = when {
                "object_recognizition_final.onnx" in assetList -> "object_recognizition_final.onnx"
                "object_recognition_final.onnx" in assetList -> "object_recognition_final.onnx"
                else -> "object_recognition_final.onnx"
            }

            Log.i(TAG, "Loading object detection model: $modelName")

            // Copy to internal cache for ONNX Session initialization
            val modelFile = File(context.cacheDir, modelName)
            if (!modelFile.exists() || modelFile.length() == 0L) {
                context.assets.open(modelName).use { input ->
                    FileOutputStream(modelFile).use { output ->
                        input.copyTo(output)
                    }
                }
            }

            val sessionOptions = OrtSession.SessionOptions().apply {
                setIntraOpNumThreads(4)
                setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
            }

            ortSession = ortEnv.createSession(modelFile.absolutePath, sessionOptions)

            // Log model inspection details
            ortSession?.let { session ->
                Log.i(TAG, "--- ONNX Object Model Inputs ---")
                for ((name, info) in session.inputInfo) {
                    Log.i(TAG, "Input: $name -> $info")
                    inputName = name
                }
                Log.i(TAG, "--- ONNX Object Model Outputs ---")
                for ((name, info) in session.outputInfo) {
                    Log.i(TAG, "Output: $name -> $info")
                }
            }

            Log.i(TAG, "Object detection model initialized successfully.")
        } catch (e: Exception) {
            Log.e(TAG, "Error loading ONNX Object Recognition model", e)
            throw RuntimeException("Failed to load Object Recognition ONNX model: ${e.message}", e)
        }
    }

    data class DetectionResult(
        val detections: List<Detection>,
        val inferenceTimeMs: Long
    )

    /**
     * Executes real-time inference on preprocessed tensor and returns decoded, NMS-filtered detections.
     */
    fun detect(preprocessResult: PreprocessResult): DetectionResult {
        val session = ortSession ?: return DetectionResult(emptyList(), 0)

        val startTime = SystemClock.uptimeMillis()
        var inputTensor: OnnxTensor? = null
        var outputResults: OrtSession.Result? = null

        try {
            inputTensor = OnnxTensor.createTensor(
                ortEnv,
                preprocessResult.floatBuffer,
                preprocessResult.shape
            )

            outputResults = session.run(Collections.singletonMap(inputName, inputTensor))

            val inferenceTimeMs = SystemClock.uptimeMillis() - startTime

            // Output0 has shape [1, 49, 8400]
            val outputTensor = outputResults.get(0) as? OnnxTensor
            if (outputTensor == null) {
                Log.e(TAG, "Output tensor at index 0 is null")
                return DetectionResult(emptyList(), inferenceTimeMs)
            }

            val tensorInfo = outputTensor.info
            val shape = tensorInfo.shape
            if (shape.size < 3 || shape[1] < (4 + NUM_CLASSES) || shape[2] != NUM_ANCHORS.toLong()) {
                Log.w(TAG, "Unexpected output shape: ${shape.contentToString()}, expected [1, 49, 8400]")
            }

            val buffer = outputTensor.floatBuffer
            val detections = decodeAndNms(buffer, preprocessResult)

            return DetectionResult(detections, inferenceTimeMs)
        } catch (e: Exception) {
            Log.e(TAG, "Inference error", e)
            return DetectionResult(emptyList(), 0)
        } finally {
            inputTensor?.close()
            outputResults?.close()
        }
    }

    /**
     * Decodes [1, 49, 8400] output tensor, converts coordinates, and applies class-aware NMS.
     */
    private fun decodeAndNms(buffer: FloatBuffer, preprocessResult: PreprocessResult): List<Detection> {
        val candidates = mutableListOf<Detection>()
        val scale = preprocessResult.scale
        val padX = preprocessResult.padX
        val padY = preprocessResult.padY
        val maxW = preprocessResult.sourceWidth.toFloat()
        val maxH = preprocessResult.sourceHeight.toFloat()

        val numAnchors = NUM_ANCHORS

        // Iterate through all 8400 candidate anchor predictions
        for (i in 0 until numAnchors) {
            // Find class with maximum score among the 13 classes
            var maxClassScore = 0.0f
            var maxClassId = -1

            for (c in 0 until NUM_CLASSES) {
                // channel index: 4 + c
                val score = buffer.get((4 + c) * numAnchors + i)
                if (score > maxClassScore) {
                    maxClassScore = score
                    maxClassId = c
                }
            }

            if (maxClassScore >= confidenceThreshold && maxClassId != -1) {
                val cx = buffer.get(0 * numAnchors + i)
                val cy = buffer.get(1 * numAnchors + i)
                val w = buffer.get(2 * numAnchors + i)
                val h = buffer.get(3 * numAnchors + i)

                // Map letterboxed 640x640 coords back to original camera dimensions
                val left = maxOf(0f, minOf(maxW, (cx - w / 2f - padX) / scale))
                val top = maxOf(0f, minOf(maxH, (cy - h / 2f - padY) / scale))
                val right = maxOf(0f, minOf(maxW, (cx + w / 2f - padX) / scale))
                val bottom = maxOf(0f, minOf(maxH, (cy + h / 2f - padY) / scale))

                if (right > left && bottom > top) {
                    candidates.add(
                        Detection(
                            classId = maxClassId,
                            label = LABELS[maxClassId],
                            confidence = maxClassScore,
                            left = left,
                            top = top,
                            right = right,
                            bottom = bottom
                        )
                    )
                }
            }
        }

        // Apply Class-Aware Non-Maximum Suppression (NMS)
        return applyClassAwareNms(candidates, iouThreshold)
    }

    private fun applyClassAwareNms(candidates: List<Detection>, iouThreshold: Float): List<Detection> {
        if (candidates.isEmpty()) return emptyList()

        val finalDetections = mutableListOf<Detection>()
        val groupedByClass = candidates.groupBy { it.classId }

        for ((_, classDetections) in groupedByClass) {
            val sorted = classDetections.sortedByDescending { it.confidence }.toMutableList()

            while (sorted.isNotEmpty()) {
                val best = sorted.removeAt(0)
                finalDetections.add(best)

                val iterator = sorted.iterator()
                while (iterator.hasNext()) {
                    val next = iterator.next()
                    if (calculateIoU(best, next) >= iouThreshold) {
                        iterator.remove()
                    }
                }
            }
        }

        return finalDetections
    }

    private fun calculateIoU(boxA: Detection, boxB: Detection): Float {
        val xA = maxOf(boxA.left, boxB.left)
        val yA = maxOf(boxA.top, boxB.top)
        val xB = minOf(boxA.right, boxB.right)
        val yB = minOf(boxA.bottom, boxB.bottom)

        val intersectionArea = maxOf(0f, xB - xA) * maxOf(0f, yB - yA)
        val boxAArea = boxA.width * boxA.height
        val boxBArea = boxB.width * boxB.height

        val unionArea = boxAArea + boxBArea - intersectionArea
        return if (unionArea <= 0f) 0f else intersectionArea / unionArea
    }

    fun close() {
        try {
            ortSession?.close()
            ortEnv.close()
        } catch (e: Exception) {
            Log.e(TAG, "Error closing ONNX detector session", e)
        }
    }
}
