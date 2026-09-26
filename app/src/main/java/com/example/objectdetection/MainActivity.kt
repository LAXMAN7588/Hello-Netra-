package com.example.objectdetection

import android.Manifest
import android.content.Context
import android.content.SharedPreferences
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Color
import android.os.Bundle
import android.os.SystemClock
import android.speech.tts.TextToSpeech
import android.util.Log
import android.view.View
import android.widget.Button
import android.widget.EditText
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.json.JSONObject
import java.net.HttpURLConnection
import java.net.URL
import java.util.Locale
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity(), TextToSpeech.OnInitListener {

    companion object {
        private const val TAG = "MainActivity"
        private const val PREFS_NAME = "object_detection_prefs"
        private const val PREF_KEY_PI_IP = "saved_pi_ip"
        private const val DEFAULT_PI_IP = "192.168.1.100"
        private const val DEFAULT_PI_PORT = 8000
        private const val TOTAL_PI_FRAMES = 6
        private const val STATUS_TIMEOUT_MS = 30000L
        private const val STATUS_POLL_INTERVAL_MS = 250L
    }

    enum class PiState {
        IDLE,
        STARTING,
        WAITING_FOR_PI,
        CAPTURING,
        DOWNLOADING,
        DETECTING,
        FINISHED,
        ERROR
    }

    // ------------------------------------------------------------
    // UI Components
    // ------------------------------------------------------------
    private lateinit var previewView: PreviewView
    private lateinit var frameView: ImageView
    private lateinit var detectionOverlay: DetectionOverlay

    private lateinit var connectionStatusText: TextView
    private lateinit var frameStatsText: TextView
    private lateinit var fpsText: TextView
    private lateinit var inferenceText: TextView
    private lateinit var ramText: TextView
    private lateinit var countText: TextView
    private lateinit var breakdownText: TextView
    private lateinit var ttsStatusText: TextView

    private lateinit var btnPhoneCamera: Button
    private lateinit var btnPiMode: Button
    private lateinit var piControlPanel: LinearLayout
    private lateinit var piIpAddress: EditText
    private lateinit var btnDetectPi: Button
    private lateinit var piStatusText: TextView

    // ------------------------------------------------------------
    // ML Models & Audio Engines
    // ------------------------------------------------------------
    private var objectDetector: ObjectDetector? = null
    private var imagePreprocessor: ImagePreprocessor? = null
    private var androidTts: TextToSpeech? = null
    private var isAndroidTtsReady = false
    private var piperTTS: PiperTTS? = null
    private var announcementManager: AnnouncementManager? = null

    // ------------------------------------------------------------
    // State & Background Management
    // ------------------------------------------------------------
    private val cameraExecutor: ExecutorService = Executors.newSingleThreadExecutor()
    private var isPiMode = false
    private var piDetectionJob: Job? = null
    private var currentState = PiState.IDLE
    private lateinit var sharedPreferences: SharedPreferences

    // Phone Camera Metrics
    private var frameCount = 0
    private var lastFpsTimestamp = 0L
    private var currentFps = 0.0

    // ------------------------------------------------------------
    // Permissions
    // ------------------------------------------------------------
    private val cameraPermissionLauncher =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { isGranted ->
            if (isGranted) {
                if (!isPiMode) {
                    startCamera()
                }
            } else {
                Toast.makeText(this, "Camera permission denied", Toast.LENGTH_SHORT).show()
            }
        }

    // ============================================================
    // ACTIVITY LIFECYCLE
    // ============================================================

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        sharedPreferences = getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)

        bindViews()
        initTtsEngines()
        initializeModels()
        setupListeners()

        // Load saved IP address
        val savedIp = sharedPreferences.getString(PREF_KEY_PI_IP, DEFAULT_PI_IP) ?: DEFAULT_PI_IP
        piIpAddress.setText(savedIp)

        // Default to Phone Camera mode
        switchToPhoneCameraMode()
    }

    private fun bindViews() {
        previewView = findViewById(R.id.previewView)
        frameView = findViewById(R.id.frameView)
        detectionOverlay = findViewById(R.id.detectionOverlay)

        connectionStatusText = findViewById(R.id.connectionStatusText)
        frameStatsText = findViewById(R.id.frameStatsText)
        fpsText = findViewById(R.id.fpsText)
        inferenceText = findViewById(R.id.inferenceText)
        ramText = findViewById(R.id.ramText)
        countText = findViewById(R.id.countText)
        breakdownText = findViewById(R.id.breakdownText)
        ttsStatusText = findViewById(R.id.ttsStatusText)

        btnPhoneCamera = findViewById(R.id.btnPhoneCamera)
        btnPiMode = findViewById(R.id.btnPiMode)
        piControlPanel = findViewById(R.id.piControlPanel)
        piIpAddress = findViewById(R.id.piIpAddress)
        btnDetectPi = findViewById(R.id.btnDetectPi)
        piStatusText = findViewById(R.id.piStatusText)
    }

    private fun setupListeners() {
        btnPhoneCamera.setOnClickListener {
            switchToPhoneCameraMode()
        }

        btnPiMode.setOnClickListener {
            switchToPiMode()
        }

        btnDetectPi.setOnClickListener {
            startPiDetectionCycle()
        }
    }

    // ============================================================
    // TTS INITIALIZATION
    // ============================================================

    private fun initTtsEngines() {
        try {
            // 1. Android Native TextToSpeech initialization
            androidTts = TextToSpeech(this, this)

            // 2. Piper ONNX TTS initialization
            piperTTS = PiperTTS(this)

            Log.i(TAG, "[TTS] Initializing text-to-speech engines...")
        } catch (e: Exception) {
            Log.e(TAG, "[TTS] Error initializing TTS engine: ${e.message}", e)
        }
    }

    override fun onInit(status: Int) {
        if (status == TextToSpeech.SUCCESS) {
            val result = androidTts?.setLanguage(Locale.US)
            if (result == TextToSpeech.LANG_MISSING_DATA || result == TextToSpeech.LANG_NOT_SUPPORTED) {
                Log.w(TAG, "[TTS] English (US) language is not supported on this device.")
            } else {
                isAndroidTtsReady = true
                Log.i(TAG, "[TTS] Android TextToSpeech initialized successfully.")
            }
        } else {
            Log.e(TAG, "[TTS] Android TextToSpeech initialization failed with status $status.")
        }
    }

    private fun speakAnnouncement(phrase: String) {
        if (phrase.isBlank()) return

        Log.i(TAG, "[TTS] Announcing final result: \"$phrase\"")
        runOnUiThread {
            ttsStatusText.text = "TTS: \"$phrase\""
        }

        // Try PiperTTS first, fallback to Android Native TTS
        try {
            if (piperTTS != null) {
                piperTTS?.speak(phrase) {
                    // Playback finished callback
                }
            } else if (isAndroidTtsReady && androidTts != null) {
                androidTts?.speak(phrase, TextToSpeech.QUEUE_FLUSH, null, "pi_result_${System.currentTimeMillis()}")
            }
        } catch (e: Exception) {
            Log.w(TAG, "[TTS] Piper speech failed, using Android TextToSpeech fallback: ${e.message}")
            if (isAndroidTtsReady && androidTts != null) {
                androidTts?.speak(phrase, TextToSpeech.QUEUE_FLUSH, null, "pi_result_fallback")
            }
        }
    }

    // ============================================================
    // MODEL INITIALIZATION
    // ============================================================

    private fun initializeModels() {
        try {
            imagePreprocessor = ImagePreprocessor(targetWidth = 640, targetHeight = 640)
            objectDetector = ObjectDetector(this, confidenceThreshold = 0.60f, iouThreshold = 0.45f)

            piperTTS?.let { tts ->
                announcementManager = AnnouncementManager(tts).apply {
                    onAnnouncementStateChanged = { phrase ->
                        runOnUiThread {
                            ttsStatusText.text = "TTS: \"$phrase\""
                        }
                    }
                }
            }

            val classesCount = objectDetector?.getLabels()?.size ?: 0
            connectionStatusText.text = "Model Loaded ($classesCount classes)"
            connectionStatusText.setTextColor(Color.GREEN)
            Log.i(TAG, "[ONNX] Model initialized with $classesCount supported classes")
        } catch (e: Exception) {
            Log.e(TAG, "[ONNX] Model initialization error", e)
            connectionStatusText.text = "Model Error"
            connectionStatusText.setTextColor(Color.RED)
        }
    }

    // ============================================================
    // MODE SWITCHING
    // ============================================================

    private fun switchToPhoneCameraMode() {
        isPiMode = false
        cancelPiJob()

        previewView.visibility = View.VISIBLE
        frameView.visibility = View.GONE
        piControlPanel.visibility = View.GONE

        btnPhoneCamera.setBackgroundColor(Color.DKGRAY)
        btnPiMode.setBackgroundColor(Color.TRANSPARENT)

        connectionStatusText.text = "Mode: Phone Camera"
        connectionStatusText.setTextColor(Color.GREEN)

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
            startCamera()
        } else {
            cameraPermissionLauncher.launch(Manifest.permission.CAMERA)
        }
    }

    private fun switchToPiMode() {
        isPiMode = true
        stopCamera()

        previewView.visibility = View.GONE
        frameView.visibility = View.VISIBLE
        piControlPanel.visibility = View.VISIBLE

        btnPhoneCamera.setBackgroundColor(Color.TRANSPARENT)
        btnPiMode.setBackgroundColor(Color.DKGRAY)

        connectionStatusText.text = "Mode: Raspberry Pi"
        connectionStatusText.setTextColor(Color.CYAN)

        setPiState(PiState.IDLE, "Raspberry Pi: Ready")

        detectionOverlay.clearDetections()
        frameStatsText.text = "Frames: 0 / $TOTAL_PI_FRAMES"
        countText.text = "Objects Detected: 0"
        breakdownText.text = "Detections: None"
        fpsText.text = "FPS: 0.0"
        inferenceText.text = "Inference: -- ms | Total: -- ms"
        ttsStatusText.text = "TTS: Ready"
    }

    // ============================================================
    // STATE MACHINE
    // ============================================================

    private fun setPiState(state: PiState, message: String? = null, isError: Boolean = false) {
        currentState = state
        runOnUiThread {
            btnDetectPi.isEnabled = (state == PiState.IDLE || state == PiState.FINISHED || state == PiState.ERROR)

            val displayMsg = message ?: when (state) {
                PiState.IDLE -> "Raspberry Pi: Ready"
                PiState.STARTING -> "Raspberry Pi: Connecting..."
                PiState.WAITING_FOR_PI -> "Waiting for Raspberry Pi..."
                PiState.CAPTURING -> "Capturing frames..."
                PiState.DOWNLOADING -> "Downloading frames..."
                PiState.DETECTING -> "Processing detections..."
                PiState.FINISHED -> "Raspberry Pi: Cycle Complete"
                PiState.ERROR -> "Raspberry Pi: Error occurred"
            }

            piStatusText.text = displayMsg
            piStatusText.setTextColor(
                when {
                    isError -> Color.RED
                    state == PiState.FINISHED -> Color.GREEN
                    state == PiState.IDLE -> Color.WHITE
                    else -> Color.YELLOW
                }
            )
        }
    }

    // ============================================================
    // PHONE CAMERA (CameraX)
    // ============================================================

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            try {
                val cameraProvider = cameraProviderFuture.get()
                val preview = Preview.Builder().build().also {
                    it.surfaceProvider = previewView.surfaceProvider
                }

                val imageAnalysis = ImageAnalysis.Builder()
                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                    .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                    .build()
                    .also { analysis ->
                        analysis.setAnalyzer(cameraExecutor) { imageProxy ->
                            processPhoneImageFrame(imageProxy)
                        }
                    }

                val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(this, cameraSelector, preview, imageAnalysis)
            } catch (e: Exception) {
                Log.e(TAG, "Camera binding failed", e)
            }
        }, ContextCompat.getMainExecutor(this))
    }

    private fun stopCamera() {
        try {
            val cameraProvider = ProcessCameraProvider.getInstance(this).get()
            cameraProvider.unbindAll()
        } catch (e: Exception) {
            Log.e(TAG, "Failed to stop camera", e)
        }
    }

    private fun processPhoneImageFrame(imageProxy: ImageProxy) {
        val detector = objectDetector
        val preprocessor = imagePreprocessor

        if (detector == null || preprocessor == null || isPiMode) {
            imageProxy.close()
            return
        }

        try {
            calculateFps()
            val preprocessResult = preprocessor.preprocess(imageProxy)
            val detectionResult = detector.detect(preprocessResult)
            val detections = detectionResult.detections

            announcementManager?.onDetections(detections)

            val ramUsageMb = calculateRamUsageMb()
            val countsSummary = buildCountsSummary(detections)

            runOnUiThread {
                detectionOverlay.setDetections(
                    detections,
                    preprocessResult.sourceWidth,
                    preprocessResult.sourceHeight
                )
                fpsText.text = "FPS: ${"%.1f".format(currentFps)}"
                inferenceText.text = "Inference: ${detectionResult.inferenceTimeMs}ms"
                ramText.text = "RAM: ${ramUsageMb}MB"
                countText.text = "Objects Detected: ${detections.size}"
                breakdownText.text = "Detections: $countsSummary"
                frameStatsText.text = "Frames Processed: $frameCount"
            }
        } catch (e: Exception) {
            Log.e(TAG, "Phone frame processing error", e)
        } finally {
            imageProxy.close()
        }
    }

    // ============================================================
    // RASPBERRY PI 6-FRAME RECOGNITION CYCLE
    // ============================================================

    private fun startPiDetectionCycle() {
        if (currentState != PiState.IDLE && currentState != PiState.FINISHED && currentState != PiState.ERROR) {
            Log.d(TAG, "[PI] Recognition cycle already in progress")
            return
        }

        val rawInput = piIpAddress.text.toString().trim()
        if (rawInput.isEmpty()) {
            setPiState(PiState.ERROR, "Raspberry Pi: Enter IP address", true)
            return
        }

        // Parse and sanitize IP / URL
        val formattedBaseUrl = formatPiBaseUrl(rawInput)
        val cleanHost = extractHost(rawInput)

        // Save valid IP to SharedPreferences
        sharedPreferences.edit().putString(PREF_KEY_PI_IP, cleanHost).apply()

        // Reset UI metrics for new cycle
        detectionOverlay.clearDetections()
        frameStatsText.text = "Frames: 0 / $TOTAL_PI_FRAMES"
        countText.text = "Objects Detected: 0"
        breakdownText.text = "Detections: Waiting..."
        ttsStatusText.text = "TTS: Processing..."
        inferenceText.text = "Inference: -- ms | Total: -- ms"

        setPiState(PiState.STARTING, "Raspberry Pi: Connecting to $cleanHost:$DEFAULT_PI_PORT...")

        val cycleStartTime = SystemClock.elapsedRealtime()

        piDetectionJob = lifecycleScope.launch(Dispatchers.IO) {
            try {
                Log.i(TAG, "[PI] Connecting to $formattedBaseUrl")

                // ----------------------------------------------------
                // Step 1: Query initial status
                // ----------------------------------------------------
                var status = fetchPiStatus(formattedBaseUrl)
                Log.i(TAG, "[PI] Status = $status")

                if (status == "unknown") {
                    throw Exception("Cannot connect to Raspberry Pi at $formattedBaseUrl")
                }

                // ----------------------------------------------------
                // Step 2: Wait for Pi capture cycle (status: waiting -> capturing -> done)
                // ----------------------------------------------------
                val waitStart = SystemClock.elapsedRealtime()
                while (status != "done") {
                    if (SystemClock.elapsedRealtime() - waitStart > STATUS_TIMEOUT_MS) {
                        throw Exception("Raspberry Pi capture timeout (>30s)")
                    }

                    when (status) {
                        "waiting" -> setPiState(PiState.WAITING_FOR_PI, "Waiting for Raspberry Pi...")
                        "capturing" -> setPiState(PiState.CAPTURING, "Capturing frames...")
                        "error" -> throw Exception("Camera error reported by Raspberry Pi")
                    }

                    delay(STATUS_POLL_INTERVAL_MS)
                    status = fetchPiStatus(formattedBaseUrl)
                    Log.d(TAG, "[PI] Status = $status")

                    if (status == "unknown") {
                        throw Exception("Lost connection to Raspberry Pi during capture")
                    }
                }

                setPiState(PiState.DOWNLOADING, "Downloading frames...")

                // ----------------------------------------------------
                // Step 3: Download & Process exactly 6 frames sequentially
                // ----------------------------------------------------
                val allFrameDetections = mutableListOf<List<Detection>>()
                var totalInferenceTimeMs = 0L
                var totalDownloadTimeMs = 0L

                for (frameIndex in 0 until TOTAL_PI_FRAMES) {
                    val frameNumber = frameIndex + 1
                    val frameName = String.format("frame_%02d.jpg", frameIndex)
                    val frameUrl = "$formattedBaseUrl/$frameName?t=${System.currentTimeMillis()}"

                    setPiState(
                        PiState.DOWNLOADING,
                        "Raspberry Pi: Downloading $frameName ($frameNumber/$TOTAL_PI_FRAMES)"
                    )
                    Log.i(TAG, "[PI] Downloading $frameName")

                    val dlStart = SystemClock.elapsedRealtime()
                    val frameBitmap = downloadBitmap(frameUrl)
                        ?: throw Exception("Failed to download $frameName from $frameUrl")
                    val dlTime = SystemClock.elapsedRealtime() - dlStart
                    totalDownloadTimeMs += dlTime

                    try {
                        setPiState(
                            PiState.DETECTING,
                            "Raspberry Pi: Processing frame $frameNumber/$TOTAL_PI_FRAMES"
                        )
                        Log.i(TAG, "[DETECT] Processing frame $frameNumber/$TOTAL_PI_FRAMES")

                        withContext(Dispatchers.Main) {
                            frameView.setImageBitmap(frameBitmap)
                        }

                        // Preprocess 1920x1080 -> 640x640 CHW
                        val preprocessResult = imagePreprocessor?.preprocess(frameBitmap)
                            ?: throw Exception("Image preprocessing failed for $frameName")

                        // ONNX Inference on all model classes
                        val detectionResult = objectDetector?.detect(preprocessResult)
                            ?: throw Exception("ONNX detection returned null for $frameName")

                        val detections = detectionResult.detections
                        allFrameDetections.add(detections)
                        totalInferenceTimeMs += detectionResult.inferenceTimeMs

                        Log.i(TAG, "[DETECT] Inference = ${detectionResult.inferenceTimeMs} ms (${detections.size} objects)")

                        val frameSummary = buildCountsSummary(detections)
                        val ramMb = calculateRamUsageMb()

                        withContext(Dispatchers.Main) {
                            detectionOverlay.setDetections(detections, frameBitmap.width, frameBitmap.height)
                            frameStatsText.text = "Frame $frameNumber / $TOTAL_PI_FRAMES"
                            countText.text = "Frame $frameNumber Objects: ${detections.size}"
                            breakdownText.text = "Frame $frameNumber: $frameSummary"
                            inferenceText.text = "Inference: ${detectionResult.inferenceTimeMs}ms | RAM: ${ramMb}MB"
                            ramText.text = "RAM: ${ramMb}MB"
                        }
                    } finally {
                        // Release temporary bitmap memory for intermediate frames to prevent high RAM consumption
                        if (frameIndex != TOTAL_PI_FRAMES - 1) {
                            frameBitmap.recycle()
                        }
                    }
                }

                Log.i(TAG, "[DETECT] Cycle complete. Aggregating results across $TOTAL_PI_FRAMES frames...")

                // ----------------------------------------------------
                // Step 4: Aggregate detections across all 6 frames
                // ----------------------------------------------------
                val finalAggregatedCounts = aggregatePiDetections(allFrameDetections)
                val totalCycleTimeMs = SystemClock.elapsedRealtime() - cycleStartTime

                Log.i(TAG, "[DETECT] FINAL AGGREGATED COUNTS = $finalAggregatedCounts (Total cycle time: ${totalCycleTimeMs}ms)")

                // ----------------------------------------------------
                // Step 5: Format TTS announcement
                // ----------------------------------------------------
                val announcementPhrase = buildFinalAnnouncementPhrase(finalAggregatedCounts)

                // ----------------------------------------------------
                // Step 6: Update UI with final scene summary
                // ----------------------------------------------------
                withContext(Dispatchers.Main) {
                    setPiState(PiState.FINISHED, "Raspberry Pi: Detection Complete")

                    frameStatsText.text = "Frames: $TOTAL_PI_FRAMES / $TOTAL_PI_FRAMES"
                    inferenceText.text = "Inference Avg: ${totalInferenceTimeMs / TOTAL_PI_FRAMES}ms | Total: ${totalCycleTimeMs}ms"
                    ramText.text = "RAM: ${calculateRamUsageMb()}MB"

                    if (finalAggregatedCounts.isEmpty()) {
                        countText.text = "Final Objects: 0"
                        breakdownText.text = "Final Counts: None"
                    } else {
                        val totalCount = finalAggregatedCounts.values.sum()
                        countText.text = "Final Objects: $totalCount"
                        val dynamicList = finalAggregatedCounts.entries.joinToString(" | ") { (label, count) ->
                            "${label.capitalizeWords()}: $count"
                        }
                        breakdownText.text = "Final Counts:\n$dynamicList"
                    }

                    // ------------------------------------------------
                    // Step 7: Announce final result once via TTS
                    // ------------------------------------------------
                    speakAnnouncement(announcementPhrase)

                    // Re-enable Object Recognition button & return to IDLE
                    setPiState(PiState.IDLE, "Raspberry Pi: Ready for next cycle")
                }

            } catch (e: Exception) {
                Log.e(TAG, "[PI] Detection cycle error: ${e.message}", e)
                withContext(Dispatchers.Main) {
                    val userErrorMsg = when {
                        e.message?.contains("connect", ignoreCase = true) == true -> "Cannot connect to Raspberry Pi"
                        e.message?.contains("download", ignoreCase = true) == true -> e.message ?: "Download failed"
                        e.message?.contains("timeout", ignoreCase = true) == true -> "Connection timed out"
                        else -> e.message ?: "Unknown error occurred"
                    }
                    setPiState(PiState.ERROR, "Raspberry Pi: $userErrorMsg", true)
                    ttsStatusText.text = "TTS: Error"
                }
            } finally {
                piDetectionJob = null
            }
        }
    }

    private fun cancelPiJob() {
        piDetectionJob?.cancel()
        piDetectionJob = null
        setPiState(PiState.IDLE)
    }

    // ============================================================
    // HTTP NETWORKING
    // ============================================================

    private fun fetchPiStatus(baseUrl: String): String {
        val urlString = "$baseUrl/status?t=${System.currentTimeMillis()}"
        var connection: HttpURLConnection? = null
        return try {
            val url = URL(urlString)
            connection = (url.openConnection() as HttpURLConnection).apply {
                connectTimeout = 3000
                readTimeout = 3000
                requestMethod = "GET"
                useCaches = false
            }

            if (connection.responseCode != HttpURLConnection.HTTP_OK) {
                Log.w(TAG, "[PI] /status returned HTTP ${connection.responseCode}")
                return "unknown"
            }

            val responseBody = connection.inputStream.bufferedReader().use { it.readText() }
            val json = JSONObject(responseBody)
            json.optString("status", "unknown")
        } catch (e: Exception) {
            Log.d(TAG, "[PI] /status check failed: ${e.message}")
            "unknown"
        } finally {
            connection?.disconnect()
        }
    }

    private fun downloadBitmap(urlString: String): Bitmap? {
        var connection: HttpURLConnection? = null
        return try {
            val url = URL(urlString)
            connection = (url.openConnection() as HttpURLConnection).apply {
                connectTimeout = 5000
                readTimeout = 10000
                requestMethod = "GET"
                useCaches = false
            }

            if (connection.responseCode != HttpURLConnection.HTTP_OK) {
                Log.e(TAG, "[PI] Frame request returned HTTP ${connection.responseCode}")
                return null
            }

            connection.inputStream.use { stream ->
                BitmapFactory.decodeStream(stream)
            }
        } catch (e: Exception) {
            Log.e(TAG, "[PI] Frame download error: ${e.message}", e)
            null
        } finally {
            connection?.disconnect()
        }
    }

    private fun formatPiBaseUrl(input: String): String {
        var trimmed = input.trim().removeSuffix("/")
        if (!trimmed.startsWith("http://") && !trimmed.startsWith("https://")) {
            trimmed = "http://$trimmed"
        }
        val uriWithoutScheme = trimmed.substringAfter("://")
        if (!uriWithoutScheme.contains(":")) {
            trimmed = "$trimmed:$DEFAULT_PI_PORT"
        }
        return trimmed
    }

    private fun extractHost(input: String): String {
        var clean = input.trim()
        if (clean.contains("://")) {
            clean = clean.substringAfter("://")
        }
        if (clean.contains(":")) {
            clean = clean.substringBefore(":")
        }
        return clean.removeSuffix("/")
    }

    // ============================================================
    // DETECTIONS AGGREGATION & NATURAL LANGUAGE TTS
    // ============================================================

    /**
     * Aggregates detections from all 6 frames.
     * Uses modal (most frequent) observed count per detected class to prevent multiplying
     * objects simply because they appear across consecutive frames.
     */
    private fun aggregatePiDetections(allDetections: List<List<Detection>>): Map<String, Int> {
        if (allDetections.isEmpty()) return emptyMap()

        // 1. Map each frame to its class counts
        val frameClassCounts = allDetections.map { detectionsInFrame ->
            val counts = mutableMapOf<String, Int>()
            for (det in detectionsInFrame) {
                counts[det.label] = (counts[det.label] ?: 0) + 1
            }
            counts
        }

        // 2. Collect all unique classes detected across the 6 frames
        val detectedClasses = frameClassCounts.flatMap { it.keys }.toSet()
        val finalResults = mutableMapOf<String, Int>()

        for (label in detectedClasses) {
            val countsPerFrame = frameClassCounts.map { it[label] ?: 0 }
            
            // Calculate frequency of each observed count
            val frequencyMap = countsPerFrame.groupingBy { it }.eachCount()

            // Find the modal count
            val modalCount = frequencyMap.maxByOrNull { it.value }?.key ?: 0

            if (modalCount > 0) {
                finalResults[label] = modalCount
            } else {
                // If modal is 0 (e.g. object seen in 2 or 3 frames), check if it was seen in multiple frames
                val nonZeroCounts = countsPerFrame.filter { it > 0 }
                if (nonZeroCounts.size >= 2) {
                    val stableCount = nonZeroCounts.groupingBy { it }.eachCount().maxByOrNull { it.value }?.key ?: 1
                    finalResults[label] = stableCount
                }
            }
        }

        return finalResults
    }

    /**
     * Dynamically builds a natural English sentence for all detected classes:
     * Examples:
     * - "One person detected."
     * - "One person and two chairs detected."
     * - "One person, two chairs, and one car detected."
     * - "No objects detected."
     */
    private fun buildFinalAnnouncementPhrase(counts: Map<String, Int>): String {
        if (counts.isEmpty()) {
            return "No objects detected."
        }

        val numberWords = arrayOf(
            "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
            "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen",
            "eighteen", "nineteen", "twenty"
        )

        val parts = mutableListOf<String>()

        for ((label, count) in counts) {
            val countWord = if (count in numberWords.indices) numberWords[count] else count.toString()
            val formattedLabel = formatClassLabelPlural(label, count)
            parts.add("$countWord $formattedLabel")
        }

        val joined = when (parts.size) {
            0 -> "No objects"
            1 -> parts[0]
            2 -> "${parts[0]} and ${parts[1]}"
            else -> {
                val head = parts.dropLast(1).joinToString(", ")
                val tail = parts.last()
                "$head, and $tail"
            }
        }

        return "${joined.replaceFirstChar { if (it.isLowerCase()) it.titlecase(Locale.US) else it.toString() }} detected."
    }

    private fun formatClassLabelPlural(label: String, count: Int): String {
        val cleanLabel = label.replace("-", " ").trim()
        if (count <= 1) return cleanLabel

        return when (cleanLabel.lowercase()) {
            "person" -> "people"
            "bus" -> "buses"
            "auto" -> "autos"
            "car" -> "cars"
            "chair" -> "chairs"
            "laptop" -> "laptops"
            "bicycle" -> "bicycles"
            "truck" -> "trucks"
            "accident" -> "accidents"
            "two wheeler" -> "two wheelers"
            "emergency responder" -> "emergency responders"
            "emergency vehicle" -> "emergency vehicles"
            "wrecked vehicle" -> "wrecked vehicles"
            else -> {
                if (cleanLabel.endsWith("s") || cleanLabel.endsWith("x") || cleanLabel.endsWith("ch") || cleanLabel.endsWith("sh")) {
                    "${cleanLabel}es"
                } else {
                    "${cleanLabel}s"
                }
            }
        }
    }

    private fun buildCountsSummary(detections: List<Detection>): String {
        if (detections.isEmpty()) return "None"
        val counts = mutableMapOf<String, Int>()
        for (detection in detections) {
            counts[detection.label] = (counts[detection.label] ?: 0) + 1
        }
        return counts.entries.joinToString(" | ") { "${it.key.capitalizeWords()}: ${it.value}" }
    }

    private fun String.capitalizeWords(): String =
        split(" ").joinToString(" ") { word ->
            word.replaceFirstChar { if (it.isLowerCase()) it.titlecase(Locale.US) else it.toString() }
        }

    // ============================================================
    // METRICS & SYSTEM HELPERS
    // ============================================================

    private fun calculateFps() {
        frameCount++
        val now = SystemClock.uptimeMillis()
        if (lastFpsTimestamp == 0L) {
            lastFpsTimestamp = now
            return
        }
        val elapsed = now - lastFpsTimestamp
        if (elapsed >= 1000) {
            currentFps = (frameCount * 1000.0) / elapsed
            frameCount = 0
            lastFpsTimestamp = now
        }
    }

    private fun calculateRamUsageMb(): Long {
        val runtime = Runtime.getRuntime()
        return (runtime.totalMemory() - runtime.freeMemory()) / (1024 * 1024)
    }

    // ============================================================
    // CLEANUP & LIFECYCLE
    // ============================================================

    override fun onDestroy() {
        cancelPiJob()
        stopCamera()
        cameraExecutor.shutdown()

        imagePreprocessor?.release()
        objectDetector?.close()

        try {
            androidTts?.stop()
            androidTts?.shutdown()
            androidTts = null
        } catch (e: Exception) {
            Log.e(TAG, "[TTS] Error shutting down Android TextToSpeech", e)
        }

        piperTTS?.close()
        announcementManager?.reset()

        super.onDestroy()
    }
}