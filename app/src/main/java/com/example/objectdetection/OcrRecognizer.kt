package com.example.objectdetection

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.graphics.Bitmap
import android.os.SystemClock
import android.util.Log
import java.io.BufferedReader
import java.io.File
import java.io.FileOutputStream
import java.io.InputStreamReader
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer

/**
 * Hello-Netra English OCR
 *
 * Custom PP-OCRv5 mobile recognition model.
 *
 * Model:
 *   en_PP-OCRv5_custom.onnx
 *
 * Input:
 *   [1, 3, 48, W]
 *
 * Output:
 *   [1, sequence_length, 438]
 *
 * Vocabulary:
 *   0       = CTC blank
 *   1..436  = PP-OCRv5 English dictionary
 *   437     = space
 *
 * Preprocessing follows PaddleOCR RecResizeImg / resize_norm_img_chinese:
 *   - preserve aspect ratio
 *   - resize height to 48
 *   - pad to width 320 with zeros
 *   - BGR channel order
 *   - normalize: (pixel / 255 - 0.5) / 0.5
 */
class OcrRecognizer(private val context: Context) {

    companion object {
        private const val TAG = "OcrRecognizer"

        private const val MODEL_NAME =
            "en_PP-OCRv5_custom.onnx"

        private const val DICT_NAME =
            "ppocrv5_en_dict.txt"

        private const val TARGET_HEIGHT = 48
        private const val TARGET_WIDTH = 320

        private const val NUM_CHANNELS = 3
        private const val NUM_CLASSES = 438
    }

    private val ortEnv: OrtEnvironment =
        OrtEnvironment.getEnvironment()

    private var ortSession: OrtSession? = null

    private var inputName = "x"

    /**
     * Index:
     *   0       = blank
     *   1..436  = characters
     *   437     = space
     */
    private val charMap = mutableListOf<String>()

    data class OcrResult(
        val text: String,
        val confidence: Float,
        val inferenceTimeMs: Long
    )

    init {
        loadDictionary()
        loadModel()
    }

    /**
     * Load PP-OCRv5 English dictionary.
     */
    private fun loadDictionary() {

        charMap.clear()

        // CTC blank
        charMap.add("")

        context.assets.open(DICT_NAME).use { input ->

            BufferedReader(
                InputStreamReader(
                    input,
                    Charsets.UTF_8
                )
            ).use { reader ->

                var line = reader.readLine()

                while (line != null) {

                    charMap.add(
                        line.trim('\r', '\n')
                    )

                    line = reader.readLine()
                }
            }
        }

        // PP-OCR space token
        charMap.add(" ")

        Log.i(
            TAG,
            "OCR dictionary loaded: ${charMap.size} tokens"
        )

        if (charMap.size != NUM_CLASSES) {
            throw RuntimeException(
                "Dictionary size mismatch. " +
                    "Expected $NUM_CLASSES tokens, " +
                    "got ${charMap.size}"
            )
        }
    }

    /**
     * Load ONNX model from Android assets.
     */
    private fun loadModel() {

        try {

            Log.i(
                TAG,
                "Loading custom OCR model: $MODEL_NAME"
            )

            val modelFile =
                File(
                    context.cacheDir,
                    MODEL_NAME
                )

            if (!modelFile.exists() ||
                modelFile.length() == 0L
            ) {

                context.assets.open(MODEL_NAME).use { input ->

                    FileOutputStream(modelFile).use { output ->

                        input.copyTo(output)
                    }
                }
            }

            Log.i(
                TAG,
                "OCR model size: ${modelFile.length()} bytes"
            )

            val sessionOptions =
                OrtSession.SessionOptions().apply {

                    setIntraOpNumThreads(4)

                    setOptimizationLevel(
                        OrtSession.SessionOptions.OptLevel.ALL_OPT
                    )
                }

            ortSession =
                ortEnv.createSession(
                    modelFile.absolutePath,
                    sessionOptions
                )

            ortSession?.let { session ->

                Log.i(
                    TAG,
                    "=== OCR INPUTS ==="
                )

                for ((name, info) in session.inputInfo) {

                    Log.i(
                        TAG,
                        "$name -> $info"
                    )

                    inputName = name
                }

                Log.i(
                    TAG,
                    "=== OCR OUTPUTS ==="
                )

                for ((name, info) in session.outputInfo) {

                    Log.i(
                        TAG,
                        "$name -> $info"
                    )
                }
            }

            Log.i(
                TAG,
                "Custom PP-OCRv5 initialized successfully"
            )

        } catch (e: Exception) {

            Log.e(
                TAG,
                "Failed to load OCR model",
                e
            )

            throw RuntimeException(
                "Failed to load OCR model: ${e.message}",
                e
            )
        }
    }

    /**
     * Run OCR on a Bitmap.
     */
    fun recognize(bitmap: Bitmap): OcrResult {

        val session =
            ortSession
                ?: return OcrResult(
                    "",
                    0f,
                    0
                )

        val startTime =
            SystemClock.uptimeMillis()

        var inputTensor: OnnxTensor? = null
        var outputResults: OrtSession.Result? = null

        try {

            /*
             * Exact PaddleOCR-style preprocessing:
             *
             * Bitmap
             *   ↓
             * preserve aspect ratio
             *   ↓
             * height = 48
             *   ↓
             * BGR
             *   ↓
             * normalize [-1,1]
             *   ↓
             * zero-pad width = 320
             */
            val inputBuffer =
                preprocessBitmap(bitmap)

            val inputShape =
                longArrayOf(
                    1,
                    NUM_CHANNELS.toLong(),
                    TARGET_HEIGHT.toLong(),
                    TARGET_WIDTH.toLong()
                )

            inputTensor =
                OnnxTensor.createTensor(
                    ortEnv,
                    inputBuffer,
                    inputShape
                )

            outputResults =
                session.run(
                    mapOf(
                        inputName to inputTensor
                    )
                )

            val inferenceTimeMs =
                SystemClock.uptimeMillis() - startTime

            val outputTensor =
                outputResults.get(0)
                    as? OnnxTensor
                    ?: return OcrResult(
                        "",
                        0f,
                        inferenceTimeMs
                    )

            val outputShape =
                outputTensor.info.shape

            if (outputShape.size != 3) {

                Log.w(
                    TAG,
                    "Unexpected output shape: " +
                        outputShape.contentToString()
                )

                return OcrResult(
                    "",
                    0f,
                    inferenceTimeMs
                )
            }

            val seqLen =
                outputShape[1].toInt()

            val numClasses =
                outputShape[2].toInt()

            if (numClasses != NUM_CLASSES) {

                Log.w(
                    TAG,
                    "Expected $NUM_CLASSES classes, " +
                        "got $numClasses"
                )
            }

            val outputBuffer =
                outputTensor.floatBuffer

            val decoded =
                ctcDecode(
                    outputBuffer,
                    seqLen,
                    numClasses
                )

            return OcrResult(
                text = decoded.first,
                confidence = decoded.second,
                inferenceTimeMs = inferenceTimeMs
            )

        } catch (e: Exception) {

            Log.e(
                TAG,
                "OCR inference failed",
                e
            )

            return OcrResult(
                "",
                0f,
                0
            )

        } finally {

            inputTensor?.close()
            outputResults?.close()
        }
    }

    /**
     * Exact PaddleOCR-style preprocessing.
     *
     * The ONNX model technically accepts dynamic width,
     * but the training/evaluation pipeline uses a 320-wide
     * padded tensor.
     */
    private fun preprocessBitmap(
        bitmap: Bitmap
    ): FloatBuffer {

        val srcWidth =
            bitmap.width

        val srcHeight =
            bitmap.height

        if (srcWidth <= 0 || srcHeight <= 0) {

            throw IllegalArgumentException(
                "Invalid bitmap dimensions"
            )
        }

        /*
         * Preserve original aspect ratio.
         */
        val ratio =
            srcWidth.toFloat() /
                srcHeight.toFloat()

        var resizedWidth =
            kotlin.math.ceil(
                TARGET_HEIGHT * ratio
            ).toInt()

        /*
         * PaddleOCR cannot exceed the padded width.
         */
        resizedWidth =
            resizedWidth.coerceAtMost(
                TARGET_WIDTH
            )

        resizedWidth =
            resizedWidth.coerceAtLeast(1)

        /*
         * Resize only to the actual content width.
         */
        val resized =
            Bitmap.createScaledBitmap(
                bitmap,
                resizedWidth,
                TARGET_HEIGHT,
                true
            )

        val pixels =
            IntArray(
                resizedWidth *
                    TARGET_HEIGHT
            )

        resized.getPixels(
            pixels,
            0,
            resizedWidth,
            0,
            0,
            resizedWidth,
            TARGET_HEIGHT
        )

        /*
         * Full padded tensor:
         *
         * [3,48,320]
         *
         * Initialized to zero exactly like
         * PaddleOCR's padding_im.
         */
        val totalValues =
            NUM_CHANNELS *
                TARGET_HEIGHT *
                TARGET_WIDTH

        val byteBuffer =
            ByteBuffer
                .allocateDirect(
                    totalValues * 4
                )
                .order(
                    ByteOrder.nativeOrder()
                )

        val floatBuffer =
            byteBuffer.asFloatBuffer()

        /*
         * Channel offsets for CHW.
         */
        val planeSize =
            TARGET_HEIGHT *
                TARGET_WIDTH

        val bOffset = 0
        val gOffset = planeSize
        val rOffset = planeSize * 2

        /*
         * Android Bitmap is ARGB/RGB.
         *
         * PaddleOCR DecodeImage uses BGR.
         *
         * Therefore write B → G → R.
         */
        for (y in 0 until TARGET_HEIGHT) {

            for (x in 0 until resizedWidth) {

                val pixel =
                    pixels[
                        y * resizedWidth + x
                    ]

                val r =
                    (pixel shr 16) and 0xFF

                val g =
                    (pixel shr 8) and 0xFF

                val b =
                    pixel and 0xFF

                /*
                 * PaddleOCR:
                 *
                 * pixel / 255
                 *     - 0.5
                 *     / 0.5
                 *
                 * = pixel / 127.5 - 1
                 */
                val bValue =
                    b / 127.5f - 1.0f

                val gValue =
                    g / 127.5f - 1.0f

                val rValue =
                    r / 127.5f - 1.0f

                val index =
                    y * TARGET_WIDTH + x

                floatBuffer.put(
                    bOffset + index,
                    bValue
                )

                floatBuffer.put(
                    gOffset + index,
                    gValue
                )

                floatBuffer.put(
                    rOffset + index,
                    rValue
                )
            }
        }

        floatBuffer.position(0)

        if (resized !== bitmap) {
            resized.recycle()
        }

        return floatBuffer
    }

    /**
     * Greedy CTC decoding.
     *
     * CTC:
     *   0 = blank
     *
     * Consecutive identical tokens are collapsed.
     */
    private fun ctcDecode(
        buffer: FloatBuffer,
        seqLen: Int,
        numClasses: Int
    ): Pair<String, Float> {

        val result =
            StringBuilder()

        var totalConfidence = 0f
        var confidenceCount = 0

        var lastClassId = -1

        for (t in 0 until seqLen) {

            var maxValue =
                -Float.MAX_VALUE

            var maxClass =
                0

            val base =
                t * numClasses

            for (c in 0 until numClasses) {

                val value =
                    buffer.get(base + c)

                if (value > maxValue) {

                    maxValue = value
                    maxClass = c
                }
            }

            /*
             * Blank.
             */
            if (maxClass == 0) {

                lastClassId = maxClass
                continue
            }

            /*
             * CTC repeated token.
             */
            if (maxClass == lastClassId) {

                continue
            }

            /*
             * Valid dictionary token.
             */
            if (maxClass in charMap.indices) {

                result.append(
                    charMap[maxClass]
                )

                /*
                 * The exported PP-OCRv5 graph already
                 * contains its output activation.
                 *
                 * Clamp only for safe presentation.
                 */
                totalConfidence +=
                    maxValue.coerceIn(
                        0f,
                        1f
                    )

                confidenceCount++
            }

            lastClassId =
                maxClass
        }

        val text =
            result
                .toString()
                .trim()

        val confidence =
            if (confidenceCount > 0) {
                totalConfidence /
                    confidenceCount
            } else {
                0f
            }

        return Pair(
            text,
            confidence
        )
    }

    /**
     * Release ONNX resources.
     */
    fun close() {

        try {

            ortSession?.close()
            ortSession = null

        } catch (e: Exception) {

            Log.e(
                TAG,
                "Error closing OCR session",
                e
            )
        }
    }
}