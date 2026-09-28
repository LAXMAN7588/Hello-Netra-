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
import org.json.JSONObject
import java.io.File
import java.io.FileOutputStream
import java.nio.FloatBuffer
import java.nio.LongBuffer
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicLong

/**
 * Piper VITS TTS engine using native eSpeak-NG for G2P (grapheme-to-phoneme)
 * and the Indian English model (en_IN-hello-netra.onnx).
 *
 * Pipeline:
 *   text -> SpeechTextFormatter (unchanged) -> NativePhonemizer (eSpeak-NG JNI)
 *        -> IPA phoneme string -> phoneme_id_map (from JSON) -> ONNX inference -> AudioTrack
 *
 * Technical Architecture & Model Interface:
 * - Model: tts/en_IN-hello-netra.onnx (Piper VITS architecture)
 * - Sample Rate: 22,050 Hz
 * - Model Inputs:
 *     1. "input": int64 [1, sequence_length] (phoneme IDs with BOS/PAD/EOS formatting)
 *     2. "input_lengths": int64 [1] (total sequence length)
 *     3. "scales": float32 [3] (noise_scale=0.667, length_scale, noise_w=0.8)
 * - Model Output:
 *     1. "output": float32 [1, 1, 1, num_samples] (Raw 22.05 kHz audio waveform)
 */
class PiperTTS(private val context: Context) {

    companion object {
        private const val TAG = "PiperTTS"
        private const val SAMPLE_RATE = 22050
        private const val SPEECH_LENGTH_SCALE = 1.15f

        // Piper special control tokens (same across all Piper models)
        private const val PAD: Long = 0L   // "_"
        private const val BOS: Long = 1L   // "^"
        private const val EOS: Long = 2L   // "$"
    }

    private val ortEnv: OrtEnvironment = OrtEnvironment.getEnvironment()
    private var ortSession: OrtSession? = null
    private val executor: ExecutorService = Executors.newSingleThreadExecutor()

    private var currentAudioTrack: AudioTrack? = null
    private val isSpeaking = AtomicBoolean(false)
    private val currentSpeechId = AtomicLong(0L)
    private var lastInferenceTimeMs = 0L

    // Parsed from en_IN-hello-netra.onnx.json at runtime
    // Maps IPA phoneme string (possibly multi-codepoint) -> list of Piper IDs
    private val phonemeIdMap = mutableMapOf<String, List<Long>>()

    private var espeakReady = false

    init {
        initEspeak()
        loadPhonemeIdMap()
        loadModel()
    }

    /**
     * Extracts espeak-ng-data from assets to cache and initializes native eSpeak-NG.
     */
    private fun initEspeak() {
        try {
            val espeakDataDir = File(context.cacheDir, "espeak-ng-data")

            // Only extract if not already present (or if it's empty)
            if (!espeakDataDir.exists() || espeakDataDir.list()?.isEmpty() != false) {
                Log.i(TAG, "Extracting espeak-ng-data from assets...")
                extractAssetDir("espeak-ng-data", espeakDataDir)
                Log.i(TAG, "espeak-ng-data extracted to: ${espeakDataDir.absolutePath}")
            }

            // espeak_Initialize expects the path to the directory CONTAINING espeak-ng-data
            val parentPath = context.cacheDir.absolutePath
            Log.i(TAG, "Initializing eSpeak-NG with path: $parentPath")
            espeakReady = NativePhonemizer.initializeEspeak(parentPath)
            Log.i(TAG, "eSpeak-NG initialized: $espeakReady")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize eSpeak-NG", e)
            espeakReady = false
        }
    }

    /**
     * Recursively extracts an asset directory to a filesystem directory.
     */
    private fun extractAssetDir(assetPath: String, targetDir: File) {
        targetDir.mkdirs()
        val entries = context.assets.list(assetPath) ?: return
        if (entries.isEmpty()) {
            // It's a file
            context.assets.open(assetPath).use { input ->
                FileOutputStream(targetDir).use { output ->
                    input.copyTo(output)
                }
            }
            return
        }
        for (entry in entries) {
            val childAssetPath = "$assetPath/$entry"
            val childTarget = File(targetDir, entry)
            val childEntries = context.assets.list(childAssetPath)
            if (childEntries != null && childEntries.isNotEmpty()) {
                // It's a subdirectory
                extractAssetDir(childAssetPath, childTarget)
            } else {
                // It's a file
                context.assets.open(childAssetPath).use { input ->
                    FileOutputStream(childTarget).use { output ->
                        input.copyTo(output)
                    }
                }
            }
        }
    }

    /**
     * Parses the phoneme_id_map from the Piper JSON config.
     * Each key is a phoneme string (single or multi-char IPA), each value is an array of IDs.
     */
    private fun loadPhonemeIdMap() {
        try {
            val jsonStr = context.assets.open("tts/en_IN-hello-netra.onnx.json")
                .bufferedReader(Charsets.UTF_8).use { it.readText() }
            val json = JSONObject(jsonStr)
            val mapObj = json.getJSONObject("phoneme_id_map")
            val keys = mapObj.keys()
            while (keys.hasNext()) {
                val key = keys.next()
                val arr = mapObj.getJSONArray(key)
                val ids = mutableListOf<Long>()
                for (i in 0 until arr.length()) {
                    ids.add(arr.getLong(i))
                }
                phonemeIdMap[key] = ids
            }
            Log.i(TAG, "Loaded phoneme_id_map with ${phonemeIdMap.size} entries")
            Log.d(TAG, "Sample entries: 'a'->${phonemeIdMap["a"]}, 'ə'->${phonemeIdMap["ə"]}, 'ˈ'->${phonemeIdMap["ˈ"]}")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to load phoneme_id_map", e)
        }
    }

    private fun loadModel() {
        try {
            val modelAssetPath = "tts/en_IN-hello-netra.onnx"
            val modelFile = File(context.cacheDir, "en_IN-hello-netra.onnx")

            if (!modelFile.exists() || modelFile.length() == 0L) {
                Log.i(TAG, "Extracting TTS model asset to cache: $modelAssetPath")
                context.assets.open(modelAssetPath).use { input ->
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

            ortSession?.let { session ->
                Log.i(TAG, "--- ONNX Piper TTS Model Inputs ---")
                for ((name, info) in session.inputInfo) {
                    Log.i(TAG, "TTS Input: $name -> $info")
                }
                Log.i(TAG, "--- ONNX Piper TTS Model Outputs ---")
                for ((name, info) in session.outputInfo) {
                    Log.i(TAG, "TTS Output: $name -> $info")
                }
            }

            Log.i(TAG, "Piper TTS model loaded successfully (sampleRate=$SAMPLE_RATE Hz).")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to load Piper TTS model", e)
        }
    }

    fun getLastInferenceTimeMs(): Long = lastInferenceTimeMs

    /**
     * Converts text into Piper phoneme token IDs using native eSpeak-NG G2P.
     *
     * Flow: text -> eSpeak-NG (native JNI) -> IPA phoneme string
     *       -> iterate codepoints -> look up each in phoneme_id_map
     *       -> intersperse PAD tokens -> wrap with BOS/EOS
     */
    fun textToPhonemeIds(text: String): LongArray {
        if (!espeakReady || phonemeIdMap.isEmpty()) {
            Log.e(TAG, "eSpeak not ready ($espeakReady) or phoneme map empty (${phonemeIdMap.size})")
            return longArrayOf(BOS, PAD, EOS)
        }

        val ipaString = NativePhonemizer.textToPhonemes(text)
        Log.d(TAG, "eSpeak IPA for '$text': '$ipaString'")

        if (ipaString.isBlank()) {
            Log.w(TAG, "eSpeak returned empty phonemes for: '$text'")
            return longArrayOf(BOS, PAD, EOS)
        }

        val tokens = mutableListOf<Long>()
        tokens.add(BOS)
        tokens.add(PAD)

        // Parse the IPA string character by character.
        // Some phonemes in the map are multi-codepoint (e.g. combining diacritics).
        // We use a greedy longest-match strategy.
        val codepoints = ipaString.codePoints().toArray()
        var i = 0
        while (i < codepoints.size) {
            // Try longest match first (up to 3 codepoints)
            var matched = false
            val maxLen = minOf(3, codepoints.size - i)
            for (len in maxLen downTo 1) {
                val candidate = String(codepoints, i, len)
                val ids = phonemeIdMap[candidate]
                if (ids != null) {
                    for (id in ids) {
                        tokens.add(id)
                        tokens.add(PAD)
                    }
                    i += len
                    matched = true
                    break
                }
            }
            if (!matched) {
                // Skip unknown codepoint
                val cp = String(codepoints, i, 1)
                Log.w(TAG, "Unknown phoneme codepoint: '$cp' (U+${Integer.toHexString(codepoints[i])})")
                i++
            }
        }

        tokens.add(EOS)

        Log.d(TAG, "Phoneme IDs (${tokens.size} tokens): ${tokens.take(30)}...")
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
     * Synthesizes audio and plays it asynchronously off the UI thread without blocking camera or UI.
     */
    fun speak(text: String, onComplete: (() -> Unit)? = null) {
        val trimmed = text.trim()
        if (trimmed.isEmpty()) {
            onComplete?.invoke()
            return
        }

        Log.d("HelloNetraOCR", "Speech input: '$trimmed'")
        Log.d("HelloNetraOCR", "Speech length scale: $SPEECH_LENGTH_SCALE")
        Log.d(TAG, "Text sent to Piper: '$trimmed'")
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
                Log.d(TAG, "Phoneme token sequence length: ${phonemeIds.size}")

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
                    Log.d(TAG, "Output sample count: ${audioSamples.size}")
                    Log.d(TAG, "Sample rate: $SAMPLE_RATE Hz")
                    Log.d(TAG, "AudioTrack format: MONO 16-bit PCM @ $SAMPLE_RATE Hz")
                    playAudio(audioSamples, speechId)
                }
            } catch (e: Exception) {
                Log.e(TAG, "Error during TTS speech synthesis", e)
            } finally {
                isSpeaking.set(false)
                onComplete?.invoke()
            }
        }
    }

    private fun synthesize(phonemeIds: LongArray): FloatArray? {
        val session = ortSession ?: return null

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

            val scalesBuffer = FloatBuffer.wrap(floatArrayOf(0.667f, SPEECH_LENGTH_SCALE, 0.8f))
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
            Log.d(TAG, "Model inference duration: ${lastInferenceTimeMs} ms")

            val outputTensor = result.get(0) as? OnnxTensor ?: return null
            val buffer = outputTensor.floatBuffer
            val audioSamples = FloatArray(buffer.remaining())
            buffer.get(audioSamples)

            return audioSamples
        } catch (e: Exception) {
            Log.e(TAG, "TTS ONNX inference error", e)
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
            Log.e(TAG, "Failed to create AudioTrack", e)
            return
        }

        currentAudioTrack = audioTrack

        if (speechId != currentSpeechId.get()) {
            audioTrack.release()
            return
        }

        try {
            audioTrack.write(shortBuffer, 0, shortBuffer.size)
            Log.d(TAG, "Playback started")
            audioTrack.play()

            val durationMs = (shortBuffer.size.toDouble() / SAMPLE_RATE * 1000).toLong()
            val startTime = SystemClock.uptimeMillis()
            while (SystemClock.uptimeMillis() - startTime < durationMs + 50) {
                if (speechId != currentSpeechId.get()) {
                    break
                }
                Thread.sleep(30)
            }
            Log.d(TAG, "Playback ended")
        } catch (e: Exception) {
            Log.e(TAG, "Playback error", e)
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

    fun stopCurrentAudio() {
        try {
            currentAudioTrack?.let {
                it.stop()
                it.release()
            }
            currentAudioTrack = null
        } catch (e: Exception) {
            Log.w(TAG, "Error stopping audio track", e)
        }
    }

    fun close() {
        stop()
        executor.shutdownNow()
        try {
            ortSession?.close()
        } catch (e: Exception) {
            Log.e(TAG, "Error closing Piper TTS", e)
        }
    }
}
