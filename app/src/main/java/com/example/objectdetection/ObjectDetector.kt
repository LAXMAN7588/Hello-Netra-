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
    var confidenceThreshold: Float = 0.60f, // 60% confidence rule
    var iouThreshold: Float = 0.45f
) {
    companion object {
        private const val TAG = "ObjectDetector"

        // Exact 13 classes for object_recognition_final.onnx
        val DEFAULT_LABELS = arrayOf(
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
    }

    private var ortEnv: OrtEnvironment = OrtEnvironment.getEnvironment()
    private var ortSession: OrtSession? = null
    private var inputName: String = "images"
    private var labels: List<String> = DEFAULT_LABELS.toList()

    init {
        loadModel()
    }

    fun getLabels(): List<String> = labels

    private fun loadModel() {
        try {
            val assetList = context.assets.list("") ?: emptyArray()

            // Explicitly prioritize object recognition models over people.onnx
            val modelName = when {
                "object_recognition_final.onnx" in assetList -> "object_recognition_final.onnx"
                "object_recognizition_final.onnx" in assetList -> "object_recognizition_final.onnx"
                "object recognition.onnx" in assetList -> "object recognition.onnx"
                "people.onnx" in assetList -> "people.onnx"
                else -> assetList.firstOrNull { it.endsWith(".onnx") && !it.startsWith("en_US") } ?: "object_recognition_final.onnx"
            }

            Log.i(TAG, "[ONNX] Loading model: $modelName")

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

            val session = ortEnv.createSession(modelFile.absolutePath, sessionOptions)
            ortSession = session

            // Inspect input tensors
            Log.i(TAG, "--- ONNX Object Model Inputs ---")
            for ((name, info) in session.inputInfo) {
                Log.i(TAG, "Input: $name -> $info")
                inputName = name
            }

            // Inspect output tensors
            Log.i(TAG, "--- ONNX Object Model Outputs ---")
            for ((name, info) in session.outputInfo) {
                Log.i(TAG, "Output: $name -> $info")
            }

            // Attempt to dynamically parse metadata if exposed by the runtime
            try {
                val metadataMethod = session.javaClass.methods.firstOrNull { it.name == "getMetadata" || it.name == "metadata" }
                if (metadataMethod != null) {
                    val metadataObj = metadataMethod.invoke(session)
                    if (metadataObj != null) {
                        val customMapMethod = metadataObj.javaClass.methods.firstOrNull { it.name == "getCustomMetadata" || it.name == "customMetadata" }
                        val customMap = customMapMethod?.invoke(metadataObj) as? Map<*, *>
                        if (customMap != null) {
                            val namesVal = customMap["names"]?.toString()
                            if (!namesVal.isNullOrEmpty()) {
                                val parsed = parseMetadataNames(namesVal)
                                if (parsed.isNotEmpty()) {
                                    labels = parsed
                                    Log.i(TAG, "[ONNX] Successfully loaded ${labels.size} classes dynamically from metadata")
                                }
                            }
                        }
                    }
                }
            } catch (e: Exception) {
                Log.d(TAG, "[ONNX] Note: Using configured model class mapping (${labels.size} classes): ${e.message}")
            }

            Log.i(TAG, "[ONNX] Active model classes (${labels.size}): ${labels.joinToString(", ")}")
            Log.i(TAG, "[ONNX] Object detection model initialized with confidenceThreshold = $confidenceThreshold")
        } catch (e: Exception) {
            Log.e(TAG, "[ONNX] Error loading ONNX Object Recognition model", e)
        }
    }

    /**
     * Parses Python dictionary format:
     * {0: 'accident', 1: 'auto', 2: 'bicycle', 3: 'bus', 4: 'car', ...}
     */
    private fun parseMetadataNames(namesStr: String): List<String> {
        val classMap = mutableMapOf<Int, String>()
        val regex = Regex("""['"]?(\d+)['"]?\s*:\s*['"]([^'"]+)['"]""")
        for (match in regex.findAll(namesStr)) {
            val id = match.groups[1]?.value?.toIntOrNull()
            val name = match.groups[2]?.value?.trim()
            if (id != null && !name.isNullOrEmpty()) {
                classMap[id] = name
            }
        }

        if (classMap.isEmpty()) {
            return emptyList()
        }

        val maxId = classMap.keys.maxOrNull() ?: return emptyList()
        val result = ArrayList<String>(maxId + 1)
        for (i in 0..maxId) {
            result.add(classMap[i] ?: "class_$i")
        }
        return result
    }

    data class DetectionResult(
        val detections: List<Detection>,
        val inferenceTimeMs: Long
    )

    /**
     * Executes inference on preprocessed tensor and returns decoded, NMS-filtered detections.
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

            val outputTensor = outputResults.get(0) as? OnnxTensor
            if (outputTensor == null) {
                Log.e(TAG, "Output tensor at index 0 is null")
                return DetectionResult(emptyList(), inferenceTimeMs)
            }

            val buffer = outputTensor.floatBuffer
            val shape = outputTensor.info.shape
            
            // Expected output shape: [1, channels, anchors] (e.g. [1, 49, 8400])
            val channels = if (shape.size >= 2) shape[1].toInt() else (4 + labels.size)
            val numAnchors = if (shape.size >= 3) shape[2].toInt() else 8400

            // Determine number of classes to read
            val numClasses = minOf(labels.size, channels - 4)

            val detections = decodeAndNms(buffer, preprocessResult, numClasses, numAnchors)

            return DetectionResult(detections, inferenceTimeMs)
        } catch (e: Exception) {
            Log.e(TAG, "[DETECT] Inference error", e)
            return DetectionResult(emptyList(), 0)
        } finally {
            inputTensor?.close()
            outputResults?.close()
        }
    }

    /**
     * Decodes output tensor, converts coordinates, filters by confidence, and applies class-aware NMS.
     */
    private fun decodeAndNms(
        buffer: FloatBuffer,
        preprocessResult: PreprocessResult,
        numClasses: Int,
        numAnchors: Int
    ): List<Detection> {
        val candidates = mutableListOf<Detection>()
        val scale = preprocessResult.scale
        val padX = preprocessResult.padX
        val padY = preprocessResult.padY
        val maxW = preprocessResult.sourceWidth.toFloat()
        val maxH = preprocessResult.sourceHeight.toFloat()

        for (i in 0 until numAnchors) {
            var maxClassScore = 0.0f
            var maxClassId = -1

            for (c in 0 until numClasses) {
                val score = buffer.get((4 + c) * numAnchors + i)
                if (score > maxClassScore) {
                    maxClassScore = score
                    maxClassId = c
                }
            }

            // Confidence threshold filtering
            if (maxClassScore >= confidenceThreshold && maxClassId != -1) {
                val cx = buffer.get(0 * numAnchors + i)
                val cy = buffer.get(1 * numAnchors + i)
                val w = buffer.get(2 * numAnchors + i)
                val h = buffer.get(3 * numAnchors + i)

                val left = maxOf(0f, minOf(maxW, (cx - w / 2f - padX) / scale))
                val top = maxOf(0f, minOf(maxH, (cy - h / 2f - padY) / scale))
                val right = maxOf(0f, minOf(maxW, (cx + w / 2f - padX) / scale))
                val bottom = maxOf(0f, minOf(maxH, (cy + h / 2f - padY) / scale))

                if (right > left && bottom > top) {
                    val label = labels.getOrElse(maxClassId) { "class_$maxClassId" }
                    candidates.add(
                        Detection(
                            classId = maxClassId,
                            label = label,
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
