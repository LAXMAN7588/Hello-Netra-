package com.hellonetra.ocr

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioTrack
import android.os.SystemClock
import android.util.Log
import java.io.File
import java.io.FileOutputStream
import java.nio.FloatBuffer
import java.nio.LongBuffer
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicLong

/**
 * ONNX-based Text-To-Speech engine using Piper VITS model (en_US-ryan-medium.onnx).
 *
 * Adapted from the original Hello-Netra ObjectDetection implementation for HelloNetraOCR.
 */
class PiperTTS(private val context: Context) {

    companion object {
        private const val TAG = "HelloNetraOCR"
        private const val SAMPLE_RATE = 22050

        // Piper special control tokens
        private const val PAD: Long = 0L
        private const val BOS: Long = 1L
        private const val EOS: Long = 2L
        private const val SPACE: Long = 3L
        private const val COMMA: Long = 8L
        private const val PERIOD: Long = 10L

        // Character to Direct Piper Phoneme ID map (from en_US-ryan-medium.onnx.json)
        private val CHAR_PHONEME_MAP = mapOf(
            ' ' to 3L,
            '!' to 4L,
            '\'' to 5L,
            ',' to 8L,
            '-' to 9L,
            '.' to 10L,
            ':' to 11L,
            ';' to 12L,
            '?' to 13L,
            'a' to 14L,
            'b' to 15L,
            'c' to 16L,
            'd' to 17L,
            'e' to 18L,
            'f' to 19L,
            'h' to 20L,
            'i' to 21L,
            'j' to 22L,
            'k' to 23L,
            'l' to 24L,
            'm' to 25L,
            'n' to 26L,
            'o' to 27L,
            'p' to 28L,
            'q' to 29L,
            'r' to 30L,
            's' to 31L,
            't' to 32L,
            'u' to 33L,
            'v' to 34L,
            'w' to 35L,
            'x' to 36L,
            'y' to 37L,
            'z' to 38L,
            '0' to 130L,
            '1' to 131L,
            '2' to 132L,
            '3' to 133L,
            '4' to 134L,
            '5' to 135L,
            '6' to 136L,
            '7' to 137L,
            '8' to 138L,
            '9' to 139L
        )

        // Exact phoneme token sequences generated from official Piper phonemizer (en-us voice)
        private val WORD_PHONEME_MAP = mapOf(
            // Numbers
            "zero" to longArrayOf(38, 120, 21, 59, 88, 27, 100),
            "one" to longArrayOf(35, 121, 102, 26),
            "two" to longArrayOf(32, 120, 33, 122),
            "three" to longArrayOf(126, 88, 120, 21, 122),
            "four" to longArrayOf(19, 120, 54, 122, 88),
            "five" to longArrayOf(19, 120, 14, 74, 34),
            "six" to longArrayOf(31, 120, 74, 23, 31),
            "seven" to longArrayOf(31, 120, 61, 34, 59, 26),
            "eight" to longArrayOf(120, 18, 74, 32),
            "nine" to longArrayOf(26, 120, 14, 74, 26),
            "ten" to longArrayOf(32, 120, 61, 26),
            "eleven" to longArrayOf(128, 24, 120, 61, 34, 59, 26),
            "twelve" to longArrayOf(32, 35, 120, 61, 24, 34),
            "thirteen" to longArrayOf(126, 120, 62, 122, 32, 21, 122, 26),
            "fourteen" to longArrayOf(19, 120, 54, 122, 88, 32, 21, 122, 26),
            "fifteen" to longArrayOf(19, 120, 74, 19, 32, 21, 122, 26),
            "sixteen" to longArrayOf(31, 120, 74, 23, 31, 32, 21, 122, 26),
            "seventeen" to longArrayOf(31, 120, 61, 34, 59, 26, 32, 121, 21, 122, 26),
            "eighteen" to longArrayOf(120, 18, 74, 32, 21, 122, 26),
            "nineteen" to longArrayOf(26, 120, 14, 74, 26, 32, 21, 122, 26),
            "twenty" to longArrayOf(32, 35, 120, 61, 26, 32, 21),
            "thirty" to longArrayOf(126, 120, 62, 122, 92, 21),
            "forty" to longArrayOf(19, 120, 54, 122, 88, 92, 21),
            "fifty" to longArrayOf(19, 120, 74, 19, 32, 21),
            "sixty" to longArrayOf(31, 120, 74, 23, 31, 32, 21),
            "seventy" to longArrayOf(31, 120, 61, 34, 59, 26, 32, 21),
            "eighty" to longArrayOf(120, 18, 74, 92, 21),
            "ninety" to longArrayOf(26, 120, 14, 74, 26, 32, 21),
            "hundred" to longArrayOf(20, 120, 102, 26, 17, 88, 74, 17),

            // Words
            "accident" to longArrayOf(120, 39, 23, 31, 74, 17, 59, 26, 32),
            "and" to longArrayOf(39, 26, 17),
            "architecture" to longArrayOf(120, 51, 122, 88, 23, 74, 32, 61, 23, 32, 62, 122, 88),
            "detected" to longArrayOf(17, 74, 32, 120, 61, 23, 32, 128, 17),
            "innovation" to longArrayOf(74, 26, 59, 34, 120, 18, 74, 124, 59, 26),
            "meets" to longArrayOf(25, 120, 21, 122, 32, 31),
            "no" to longArrayOf(26, 120, 27, 100),
            "ocr" to longArrayOf(120, 27, 100, 31, 21, 122, 120, 51, 122, 88),
            "ready" to longArrayOf(88, 120, 61, 17, 21),
            "reading" to longArrayOf(88, 120, 21, 122, 17, 74, 44),
            "simplicity" to longArrayOf(31, 74, 25, 28, 24, 74, 31, 74, 32, 21),
            "text" to longArrayOf(32, 120, 61, 23, 31, 32)
        )
    }

    private val ortEnv: OrtEnvironment = OrtEnvironment.getEnvironment()
    private var ortSession: OrtSession? = null
    private val executor: ExecutorService = Executors.newSingleThreadExecutor()

    private var currentAudioTrack: AudioTrack? = null
    private val isSpeaking = AtomicBoolean(false)
    private val currentSpeechId = AtomicLong(0L)
    private var lastInferenceTimeMs = 0L

    init {
        loadModel()
    }

    private fun loadModel() {
        try {
            val modelName = "en_US-ryan-medium.onnx"
            val modelFile = File(context.cacheDir, modelName)

            if (!modelFile.exists() || modelFile.length() == 0L) {
                Log.i(TAG, "PiperTTS: Extracting model asset to cache: $modelName")
                context.assets.open(modelName).use { input ->
                    FileOutputStream(modelFile).use { output ->
                        input.copyTo(output)
                    }
                }
            }

            val sessionOptions = OrtSession.SessionOptions().apply {
                setIntraOpNumThreads(2)
                setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
            }

            ortSession = ortEnv.createSession(modelFile.absolutePath, sessionOptions)
            Log.i(TAG, "PiperTTS: Model loaded successfully (sampleRate=$SAMPLE_RATE)")
        } catch (e: Exception) {
            Log.e(TAG, "PiperTTS: Failed to load model: ${e.message}", e)
        }
    }

    fun getLastInferenceTimeMs(): Long = lastInferenceTimeMs

    /**
     * Converts text (words, phrases, arbitrary OCR text) into Piper phoneme token IDs.
     */
    fun textToPhonemeIds(text: String): LongArray {
        val tokens = mutableListOf<Long>()
        tokens.add(BOS)
        tokens.add(PAD)

        val cleanText = text.lowercase().replace("-", " ").replace("—", " ")
        val wordsAndPunct = cleanText.split(Regex("\\s+"))

        var addedWord = false
        for (rawItem in wordsAndPunct) {
            var item = rawItem.trim()
            if (item.isEmpty()) continue

            var hasComma = false
            var hasPeriod = false

            if (item.endsWith(",")) {
                hasComma = true
                item = item.substring(0, item.length - 1)
            } else if (item.endsWith(".")) {
                hasPeriod = true
                item = item.substring(0, item.length - 1)
            }

            val knownPhonemes = WORD_PHONEME_MAP[item]
            if (knownPhonemes != null) {
                if (addedWord) {
                    tokens.add(SPACE)
                    tokens.add(PAD)
                }

                for (id in knownPhonemes) {
                    tokens.add(id)
                    tokens.add(PAD)
                }
                addedWord = true
            } else {
                // Character-by-character phonetic synthesis for arbitrary OCR text
                if (addedWord) {
                    tokens.add(SPACE)
                    tokens.add(PAD)
                }

                for (char in item) {
                    val charToken = CHAR_PHONEME_MAP[char]
                    if (charToken != null) {
                        tokens.add(charToken)
                        tokens.add(PAD)
                    }
                }
                addedWord = true
            }

            if (hasComma) {
                tokens.add(COMMA)
                tokens.add(PAD)
            } else if (hasPeriod) {
                tokens.add(PERIOD)
                tokens.add(PAD)
            }
        }

        tokens.add(EOS)
        return tokens.toLongArray()
    }

    /**
     * Stops any currently playing speech immediately.
     */
    fun stop() {
        currentSpeechId.incrementAndGet()
        stopCurrentAudio()
        isSpeaking.set(false)
    }

    /**
     * Synthesizes audio and plays it asynchronously off the UI thread.
     */
    fun speak(text: String, onComplete: (() -> Unit)? = null) {
        val trimmed = text.trim()
        if (trimmed.isEmpty()) {
            onComplete?.invoke()
            return
        }

        Log.d(TAG, "PiperTTS: Received text to speak: '$trimmed'")
        val speechId = currentSpeechId.incrementAndGet()

        executor.execute {
            stopCurrentAudio()
            if (speechId != currentSpeechId.get()) {
                onComplete?.invoke()
                return@execute
            }

            isSpeaking.set(true)
            try {
                val phonemeIds = textToPhonemeIds(trimmed)
                if (phonemeIds.size <= 3 || speechId != currentSpeechId.get()) {
                    isSpeaking.set(false)
                    onComplete?.invoke()
                    return@execute
                }

                val audioSamples = synthesize(phonemeIds)
                if (speechId != currentSpeechId.get()) {
                    isSpeaking.set(false)
                    onComplete?.invoke()
                    return@execute
                }

                if (audioSamples != null && audioSamples.isNotEmpty()) {
                    val durationMs = (audioSamples.size.toDouble() / SAMPLE_RATE * 1000).toLong()
                    Log.d(TAG, "PiperTTS: Generated audio: sampleRate=$SAMPLE_RATE, samplesCount=${audioSamples.size}, durationMs=$durationMs")
                    playAudio(audioSamples, speechId)
                }
            } catch (e: Exception) {
                Log.e(TAG, "PiperTTS: Error during speech synthesis: ${e.message}", e)
            } finally {
                isSpeaking.set(false)
                onComplete?.invoke()
            }
        }
    }

    private fun synthesize(phonemeIds: LongArray): FloatArray? {
        val session = ortSession ?: run {
            Log.e(TAG, "PiperTTS: Cannot synthesize, ONNX session is null")
            return null
        }

        val startTime = SystemClock.uptimeMillis()

        var inputTensor: OnnxTensor? = null
        var lengthsTensor: OnnxTensor? = null
        var scalesTensor: OnnxTensor? = null
        var result: OrtSession.Result? = null

        try {
            val inputBuffer = LongBuffer.wrap(phonemeIds)
            inputTensor = OnnxTensor.createTensor(
                ortEnv,
                inputBuffer,
                longArrayOf(1, phonemeIds.size.toLong())
            )

            val lengthsBuffer = LongBuffer.wrap(longArrayOf(phonemeIds.size.toLong()))
            lengthsTensor = OnnxTensor.createTensor(
                ortEnv,
                lengthsBuffer,
                longArrayOf(1)
            )

            val scalesBuffer = FloatBuffer.wrap(floatArrayOf(0.667f, 1.0f, 0.8f))
            scalesTensor = OnnxTensor.createTensor(
                ortEnv,
                scalesBuffer,
                longArrayOf(3)
            )

            val inputs = mapOf(
                "input" to inputTensor,
                "input_lengths" to lengthsTensor,
                "scales" to scalesTensor
            )

            result = session.run(inputs)
            lastInferenceTimeMs = SystemClock.uptimeMillis() - startTime

            val outputTensor = result.get(0) as? OnnxTensor ?: return null
            val buffer = outputTensor.floatBuffer
            val audioSamples = FloatArray(buffer.remaining())
            buffer.get(audioSamples)

            return audioSamples
        } catch (e: Exception) {
            Log.e(TAG, "PiperTTS: ONNX inference error: ${e.message}", e)
            return null
        } finally {
            inputTensor?.close()
            lengthsTensor?.close()
            scalesTensor?.close()
            result?.close()
        }
    }

    private fun playAudio(samples: FloatArray, speechId: Long) {
        val shortBuffer = ShortArray(samples.size)
        for (i in samples.indices) {
            val clamped = maxOf(-1.0f, minOf(1.0f, samples[i]))
            shortBuffer[i] = (clamped * 32767.0f).toInt().toShort()
        }

        val minBufferSize = AudioTrack.getMinBufferSize(
            SAMPLE_RATE,
            AudioFormat.CHANNEL_OUT_MONO,
            AudioFormat.ENCODING_PCM_16BIT
        )
        val bufferSize = maxOf(minBufferSize, shortBuffer.size * 2)

        val audioTrack = try {
            AudioTrack.Builder()
                .setAudioAttributes(
                    AudioAttributes.Builder()
                        .setUsage(AudioAttributes.USAGE_ASSISTANCE_ACCESSIBILITY)
                        .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
                        .build()
                )
                .setAudioFormat(
                    AudioFormat.Builder()
                        .setEncoding(AudioFormat.ENCODING_PCM_16BIT)
                        .setSampleRate(SAMPLE_RATE)
                        .setChannelMask(AudioFormat.CHANNEL_OUT_MONO)
                        .build()
                )
                .setBufferSizeInBytes(bufferSize)
                .setTransferMode(AudioTrack.MODE_STATIC)
                .build()
        } catch (e: Exception) {
            Log.e(TAG, "PiperTTS: Failed to create AudioTrack: ${e.message}", e)
            return
        }

        currentAudioTrack = audioTrack

        if (speechId != currentSpeechId.get()) {
            audioTrack.release()
            return
        }

        try {
            audioTrack.write(shortBuffer, 0, shortBuffer.size)
            Log.d(TAG, "PiperTTS: Playback started")
            audioTrack.play()

            val durationMs = (shortBuffer.size.toDouble() / SAMPLE_RATE * 1000).toLong()
            val startTime = SystemClock.uptimeMillis()
            while (SystemClock.uptimeMillis() - startTime < durationMs + 50) {
                if (speechId != currentSpeechId.get()) {
                    break
                }
                Thread.sleep(30)
            }
            Log.d(TAG, "PiperTTS: Playback ended (completed: ${speechId == currentSpeechId.get()})")
        } catch (e: Exception) {
            Log.e(TAG, "PiperTTS: Playback error: ${e.message}", e)
        } finally {
            try {
                audioTrack.stop()
                audioTrack.release()
            } catch (_: Throwable) {}
            if (currentAudioTrack === audioTrack) {
                currentAudioTrack = null
            }
        }
    }

    private fun stopCurrentAudio() {
        try {
            currentAudioTrack?.let {
                it.stop()
                it.release()
            }
            currentAudioTrack = null
        } catch (e: Exception) {
            Log.w(TAG, "PiperTTS: Error stopping audio track: ${e.message}")
        }
    }

    fun close() {
        stop()
        executor.shutdownNow()
        try {
            ortSession?.close()
        } catch (e: Exception) {
            Log.e(TAG, "PiperTTS: Error closing session: ${e.message}", e)
        }
    }
}
