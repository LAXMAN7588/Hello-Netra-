package com.example.objectdetection
import java.util.concurrent.atomic.AtomicBoolean
import java.io.File
import java.io.FileOutputStream
import android.app.Activity
import android.Manifest
import android.content.BroadcastReceiver
import android.content.Context
import android.content.IntentFilter
import android.content.Intent
import android.content.SharedPreferences
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Color
import android.os.Build
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
import java.util.Locale
import com.hellonetra.ocr.SpeechTextFormatter
import com.hellonetra.ocr.PiperTTS as OcrPiperTTS
import com.paddle.ocr.EngineConfig
import com.paddle.ocr.PaddleOCR
import com.paddle.ocr.PaddleOCRConfig
import com.paddle.ocr.util.OpenCVUtils
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity(), TextToSpeech.OnInitListener {

    companion object {
        private const val TAG = "MainActivity"
        private const val PREFS_NAME = "object_detection_prefs"
        private const val PREF_KEY_PI_IP = "saved_pi_ip"
        private const val DEFAULT_PI_IP = "10.138.117.190"
        private const val PI_UDP_PORT = 5000
    }

    enum class PiState {
        IDLE,
        STARTING,
        OBJECT_RECOGNITION,
        OCR,
        SOS,
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
    private lateinit var ocrStatusText: TextView
    private lateinit var ocrResultText: TextView

    private lateinit var btnPhoneCamera: Button
    private lateinit var btnOcrMode: Button
    private lateinit var btnSos: Button
    private lateinit var btnPiMode: Button
    private lateinit var piControlPanel: LinearLayout
    private lateinit var piIpAddress: EditText
    private lateinit var btnDetectPi: Button
    private lateinit var btnPiOcr: Button
    private lateinit var piStatusText: TextView

    private lateinit var sosControlPanel: LinearLayout
    private lateinit var btnAddEmergencyContact: Button
    private lateinit var btnSendSos: Button
    private lateinit var sosStatusText: TextView

    private lateinit var sosManager: SosManager

    // ------------------------------------------------------------
    // ML Models & Audio Engines
    // ------------------------------------------------------------
    private var objectDetector: ObjectDetector? = null
    private var imagePreprocessor: ImagePreprocessor? = null
    private var androidTts: TextToSpeech? = null
    private var isAndroidTtsReady = false
    private var piperTTS: PiperTTS? = null
    private var announcementManager: AnnouncementManager? = null

    // OCR
    private var ocrEngine: PaddleOCR? = null
    private var ocrTts: OcrPiperTTS? = null
    private val ocrProcessing = AtomicBoolean(false)
    private var lastOcrTimeMs = 0L
    private var ocrInitialized = false
    @Volatile
    private var debugFrameSaved = false

    private val piSignalReceiver = object : BroadcastReceiver() {
        override fun onReceive(context: Context?, intent: Intent?) {
            val command = intent?.getStringExtra("command") ?: return
            Log.i(TAG, "NETRA_SIGNAL: MainActivity received signal '$command'")
            when (command.uppercase(Locale.ROOT)) {
                "OBJECT" -> {
                    switchToPiMode()
                    startPiDetectionCycle()
                }
                "OCR" -> {
                    switchToPiMode()
                    startPiOcrCycle()
                }
                "SOS" -> {
                    switchToSosMode()
                    sosManager.startSos(this@MainActivity)
                }
            }
        }
    }


    @Volatile
    private var ocrGeneration = 0L
    private val ocrSpeaking = AtomicBoolean(false)
    private var ocrNextAllowedTimeMs = 0L

    // ------------------------------------------------------------
    // OCR INITIALIZATION
    // ------------------------------------------------------------

    private fun initializeOcr() {
        if (ocrInitialized) return

        lifecycleScope.launch(Dispatchers.IO) {
            try {
                Log.d(TAG, "NETRA_OCR Initializing OCR engine")

                OpenCVUtils.init(this@MainActivity)

                val ocrConfig = PaddleOCRConfig(
                    detImgMode = "BGR",
                    detLimitSideLen = 64,
                    detLimitType = "min",
                    detMaxSideLimit = 4000,
                    detThresh = 0.3f,
                    detBoxThresh = 0.6f,
                    detUnclipRatio = 1.5f,
                    detMaxCandidates = 3000,
                    detUseDilation = false,
                    detScoreMode = "fast",
                    detBoxType = "quad",
                    recScoreThresh = 0.0f,
                    recBatchSize = 1
                )

                val engineConfig = EngineConfig(
                    numThreads = 4
                )

                val engine = PaddleOCR.create(
                    context = this@MainActivity,
                    config = ocrConfig,
                    engineConfig = engineConfig,
                    detModelAssetPath = "det/ch_PP-OCRv5_det_mobile.onnx",
                    recModelAssetPath = "rec/en_PP-OCRv5_rec_mobile.onnx",
                    recConfigAssetPath = "rec/ppocrv5_en_rec.yml"
                )

                val tts = OcrPiperTTS(this@MainActivity)

                withContext(Dispatchers.Main) {
                    ocrEngine = engine
                    ocrTts = tts
                    ocrInitialized = true

                    ocrStatusText.text = "OCR: Ready"
                    Log.d(TAG, "NETRA_OCR OCR engine ready")
                }

            } catch (e: Exception) {
                Log.e(TAG, "NETRA_OCR Initialization failed", e)

                withContext(Dispatchers.Main) {
                    ocrStatusText.text = "OCR: Initialization failed"
                    Toast.makeText(
                        this@MainActivity,
                        "OCR initialization failed: ${e.message}",
                        Toast.LENGTH_LONG
                    ).show()
                }
            }
        }
    }

    private fun stopOcrMode() {
        ocrGeneration++
        ocrTts?.stop()
        ocrProcessing.set(false)
        ocrSpeaking.set(false)
        ocrNextAllowedTimeMs = 0L
    }

    // ------------------------------------------------------------
    // State & Background Management
    // ------------------------------------------------------------
    private val cameraExecutor: ExecutorService = Executors.newSingleThreadExecutor()
    private var isPiMode = false
    private var piDetectionJob: Job? = null
    private var currentState = PiState.IDLE
    private lateinit var sharedPreferences: SharedPreferences
    // Raspberry Pi UDP camera
    private var piVideoReceiver: PiVideoReceiver? = null

    // Prevent multiple ML inferences from running simultaneously.
    private val piFrameProcessing = AtomicBoolean(false)

    private var piFrameCount = 0
    private var piLastInferenceTimeMs = 0L
    // Phone Camera Metrics
    private var frameCount = 0
    private var lastFpsTimestamp = 0L
    private var currentFps = 0.0

    // ------------------------------------------------------------
    // Permissions
    // ------------------------------------------------------------
    
    private val contactPickerLauncher =
        registerForActivityResult(ActivityResultContracts.StartActivityForResult()) { result ->
            if (result.resultCode != Activity.RESULT_OK) return@registerForActivityResult
            val uri = result.data?.data ?: return@registerForActivityResult

            try {
                val projection = arrayOf(android.provider.ContactsContract.CommonDataKinds.Phone.NUMBER)
                contentResolver.query(uri, projection, null, null, null)?.use { cursor ->
                    if (cursor.moveToFirst()) {
                        val numberIndex = cursor.getColumnIndex(android.provider.ContactsContract.CommonDataKinds.Phone.NUMBER)
                        if (numberIndex >= 0) {
                            val number = cursor.getString(numberIndex)
                            if (!number.isNullOrBlank()) {
                                sosManager.addEmergencyContact(number)
                                Toast.makeText(this, "Added Emergency Contact: $number", Toast.LENGTH_SHORT).show()
                            }
                        }
                    }
                }
            } catch (e: Exception) {
                Log.e(TAG, "Error selecting contact", e)
                Toast.makeText(this, "Failed to pick contact: ${e.message}", Toast.LENGTH_SHORT).show()
            }
        }

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

        sosManager = SosManager(this) { status ->
            runOnUiThread {
                sosStatusText.text = status
            }
        }

        initializeModels()
        setupListeners()
        startCaneBackgroundService()

        val filter = IntentFilter("com.example.objectdetection.PI_SIGNAL")
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            registerReceiver(piSignalReceiver, filter, Context.RECEIVER_NOT_EXPORTED)
        } else {
            registerReceiver(piSignalReceiver, filter)
        }
    

        // Initialize OCR engine in the background.
        initializeOcr()

        // Load saved IP address
        val savedIp = sharedPreferences.getString(PREF_KEY_PI_IP, DEFAULT_PI_IP) ?: DEFAULT_PI_IP
        piIpAddress.setText(savedIp)

        // Default to Phone Camera mode
        switchToPhoneCameraMode()
    }

    private fun startCaneBackgroundService() {
        try {
            val serviceIntent = Intent(this, CaneBackgroundService::class.java)
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                ContextCompat.startForegroundService(this, serviceIntent)
            } else {
                startService(serviceIntent)
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to start CaneBackgroundService", e)
        }
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
        ocrStatusText = findViewById(R.id.ocrStatusText)
        ocrResultText = findViewById(R.id.ocrResultText)

        btnPhoneCamera = findViewById(R.id.btnPhoneCamera)
        btnOcrMode = findViewById(R.id.btnOcrMode)
        btnSos = findViewById(R.id.btnSos)
        btnPiMode = findViewById(R.id.btnPiMode)
        piControlPanel = findViewById(R.id.piControlPanel)
        piIpAddress = findViewById(R.id.piIpAddress)
        btnDetectPi = findViewById(R.id.btnDetectPi)
        btnPiOcr = findViewById(R.id.btnPiOcr)
        piStatusText = findViewById(R.id.piStatusText)
        sosControlPanel = findViewById(R.id.sosPanel)
        btnAddEmergencyContact = findViewById(R.id.btnAddEmergencyContact)
        btnSendSos = findViewById(R.id.btnSendSos)
        sosStatusText = findViewById(R.id.sosStatusText)
    }

    private fun setupListeners() {
        btnPhoneCamera.setOnClickListener {
            switchToPhoneCameraMode()
        }

        btnOcrMode.setOnClickListener {
            switchToOcrMode()
        }
        btnSos.setOnClickListener {
            switchToSosMode()
        }

        btnAddEmergencyContact.setOnClickListener {
            val intent = sosManager.createContactPickerIntent()
            contactPickerLauncher.launch(intent)
        }

        btnSendSos.setOnClickListener {
            sosManager.startSos(
                activity = this,
                onLocationRequired = {
                    Toast.makeText(this, "Location permission required for SOS", Toast.LENGTH_LONG).show()
                },
                onCallPermissionRequired = {
                    Toast.makeText(this, "Call permission required for SOS", Toast.LENGTH_LONG).show()
                }
            )
        }


        btnPiMode.setOnClickListener {
            switchToPiMode()
        }

        btnDetectPi.setOnClickListener {
            startPiDetectionCycle()
        }
        btnPiOcr.setOnClickListener {
            startPiOcrCycle()
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

    
        private fun switchToSosMode() {
        Log.i(TAG, "[NETRA_SOS] Switching to SOS mode")
        
        isPiMode = false
        cancelPiJob()
        stopCamera()
        stopOcrMode()
        announcementManager?.reset()
        piperTTS?.stopCurrentAudio()

        currentState = PiState.SOS

        detectionOverlay.clearDetections()
        previewView.visibility = View.GONE
        frameView.visibility = View.GONE
        piControlPanel.visibility = View.GONE
        sosControlPanel.visibility = View.VISIBLE

        btnPhoneCamera.setBackgroundColor(Color.TRANSPARENT)
        btnOcrMode.setBackgroundColor(Color.TRANSPARENT)
        btnPiMode.setBackgroundColor(Color.TRANSPARENT)
        btnSos.setBackgroundColor(Color.DKGRAY)

        connectionStatusText.text = "Mode: SOS"
        connectionStatusText.setTextColor(Color.RED)
    }

    private fun switchToOcrMode() {
        Log.i(TAG, "NETRA_OCR: switchToOcrMode entered")

        cancelPiJob()
        piVideoReceiver?.stop()
        piVideoReceiver = null
        isPiMode = false

        // Stop Object Recognition TTS & announcements
        announcementManager?.reset()
        piperTTS?.stopCurrentAudio()

        stopOcrMode()
        stopCamera()

        currentState = PiState.OCR

        // Clear Object Recognition Overlay
        detectionOverlay.clearDetections()
        countText.text = "Objects Detected: 0"
        breakdownText.text = "Detections: None"
        inferenceText.text = "Inference: -- ms | Total: -- ms"

        previewView.visibility = View.VISIBLE
        frameView.visibility = View.GONE
        piControlPanel.visibility = View.GONE
        sosControlPanel.visibility = View.GONE

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
            startCamera()
        } else {
            cameraPermissionLauncher.launch(Manifest.permission.CAMERA)
        }

        ocrStatusText.text = if (ocrInitialized) "OCR: Ready" else "OCR: Initializing..."
        ocrResultText.text = ""

        btnPhoneCamera.setBackgroundColor(Color.TRANSPARENT)
        btnOcrMode.setBackgroundColor(Color.DKGRAY)
        btnPiMode.setBackgroundColor(Color.TRANSPARENT)
        btnSos.setBackgroundColor(Color.TRANSPARENT)

        Log.i(TAG, "NETRA_OCR: OCR mode active")
    }

    private fun switchToPhoneCameraMode() {
        Log.i(TAG, "NETRA_OBJECT: Switching to Phone Camera mode")
        isPiMode = false
        cancelPiJob()
        stopOcrMode()
        stopCamera()

        currentState = PiState.OBJECT_RECOGNITION

        detectionOverlay.clearDetections()
        countText.text = "Objects Detected: 0"
        breakdownText.text = "Detections: None"

        previewView.visibility = View.VISIBLE
        frameView.visibility = View.GONE
        piControlPanel.visibility = View.GONE
        sosControlPanel.visibility = View.GONE

        btnPhoneCamera.setBackgroundColor(Color.DKGRAY)
        btnOcrMode.setBackgroundColor(Color.TRANSPARENT)
        btnPiMode.setBackgroundColor(Color.TRANSPARENT)
        btnSos.setBackgroundColor(Color.TRANSPARENT)

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
        stopOcrMode()

        previewView.visibility = View.GONE
        frameView.visibility = View.VISIBLE
        piControlPanel.visibility = View.VISIBLE
        sosControlPanel.visibility = View.GONE

        btnPhoneCamera.setBackgroundColor(Color.TRANSPARENT)
        btnPiMode.setBackgroundColor(Color.DKGRAY)

        connectionStatusText.text = "Mode: Raspberry Pi"
        connectionStatusText.setTextColor(Color.CYAN)

        setPiState(PiState.IDLE, "Raspberry Pi: Ready")

        detectionOverlay.clearDetections()
        frameStatsText.text = "Frames: 0"
        countText.text = "Objects Detected: 0"
        breakdownText.text = "Detections: None"
        fpsText.text = "FPS: 0.0"
        inferenceText.text = "Inference: -- ms"
        ttsStatusText.text = "TTS: Ready"
    }

    // ============================================================
    // STATE MACHINE
    // ============================================================

    private fun setPiState(
        state: PiState,
        message: String? = null,
        isError: Boolean = false
    ) {
        currentState = state

        runOnUiThread {
            btnDetectPi.isEnabled =
                state == PiState.IDLE ||
                state == PiState.ERROR
            btnPiOcr.isEnabled = btnDetectPi.isEnabled

            val displayMsg =
                message ?: when (state) {
                    PiState.IDLE ->
                        "Raspberry Pi: Ready"

                    PiState.STARTING ->
                        "Raspberry Pi: Connecting..."

                    PiState.OBJECT_RECOGNITION ->
                        "Raspberry Pi: Receiving camera stream"
                    PiState.OCR ->
                        "OCR: Active"
                    PiState.SOS ->
                        "SOS: Active"
                    PiState.ERROR ->
                        "Raspberry Pi: Error occurred"
                }

            piStatusText.text = displayMsg

            piStatusText.setTextColor(
                when {
                    isError -> Color.RED
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
        Log.i(TAG, "NETRA_OCR: OCR camera binding started (currentState=$currentState)")
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            try {
                val cameraProvider = cameraProviderFuture.get()
                Log.i(TAG, "[NETRA_OBJECT] CameraX provider obtained")
                val preview = Preview.Builder().build().also {
                    it.surfaceProvider = previewView.surfaceProvider
                }

                val imageAnalysis = ImageAnalysis.Builder()
                    .setTargetResolution(android.util.Size(1280, 720))
                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                    .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                    .build()
                    .also { analysis ->
                        analysis.setAnalyzer(cameraExecutor) { imageProxy ->
                            when (currentState) {
                                PiState.OCR -> processPhoneOcrFrame(imageProxy)
                                PiState.OBJECT_RECOGNITION -> processPhoneImageFrame(imageProxy)
                                else -> imageProxy.close()
                            }
                        }
                    }

                val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(this, cameraSelector, preview, imageAnalysis)
                Log.i(TAG, "NETRA_OCR: OCR camera binding successful")
            } catch (e: Exception) {
                Log.e(TAG, "Camera binding failed", e)
            }
        }, ContextCompat.getMainExecutor(this))
    }

    private fun stopCamera() {
        Log.i(TAG, "NETRA_OCR: stopCamera entered")
        try {
            val cameraProvider = ProcessCameraProvider.getInstance(this).get()
            cameraProvider.unbindAll()
            Log.i(TAG, "NETRA_OBJECT: Object Recognition camera stopped when switching to OCR")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to stop camera", e)
        }
    }

    private fun processPhoneImageFrame(imageProxy: ImageProxy) {
        if (currentState != PiState.OBJECT_RECOGNITION || isPiMode) {
            imageProxy.close()
            return
        }

        val detector = objectDetector
        val preprocessor = imagePreprocessor

        if (detector == null || preprocessor == null) {
            imageProxy.close()
            return
        }

        try {
            calculateFps()
            val preprocessResult = preprocessor.preprocess(imageProxy)
            val detectionResult = detector.detect(preprocessResult)
            val detections = detectionResult.detections

            if (currentState == PiState.OBJECT_RECOGNITION) {
                announcementManager?.onDetections(detections)

                val ramUsageMb = calculateRamUsageMb()
                val countsSummary = buildCountsSummary(detections)

                runOnUiThread {
                    if (currentState != PiState.OBJECT_RECOGNITION) return@runOnUiThread

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
            }
        } catch (e: Exception) {
            Log.e(TAG, "Phone frame processing error", e)
        } finally {
            imageProxy.close()
        }
    }

    private fun processPhoneOcrFrame(imageProxy: ImageProxy) {
        if (currentState != PiState.OCR || isPiMode) {
            imageProxy.close()
            return
        }

        val now = SystemClock.elapsedRealtime()

        if (ocrSpeaking.get() || now < ocrNextAllowedTimeMs) {
            imageProxy.close()
            return
        }

        if (now - lastOcrTimeMs < 1200L) {
            imageProxy.close()
            return
        }

        if (!ocrProcessing.compareAndSet(false, true)) {
            imageProxy.close()
            return
        }

        lastOcrTimeMs = now
        val currentGen = ocrGeneration
        val engine = ocrEngine

        if (engine == null || !ocrInitialized || currentState != PiState.OCR) {
            ocrProcessing.set(false)
            imageProxy.close()
            return
        }

        try {
            Log.i(TAG, "NETRA_OCR: OCR ImageProxy received: ${imageProxy.width}x${imageProxy.height} rot=${imageProxy.imageInfo.rotationDegrees}")

            val bitmap = imageProxyToBitmap(imageProxy)
            Log.i(TAG, "NETRA_OCR: OCR frame converted to Bitmap")

            lifecycleScope.launch(Dispatchers.IO) {
                try {
                    Log.i(TAG, "NETRA_OCR: OCR recognize() started")
                    val result = engine.recognize(bitmap)
                    Log.i(TAG, "NETRA_OCR: OCR recognize() completed, lines=${result.lineCount}")

                    if (currentGen != ocrGeneration) return@launch

                    val rawText = result.results.joinToString("\n") { it.text }
                    val speechText = SpeechTextFormatter.format(result.results)

                    withContext(Dispatchers.Main) {
                        if (currentState != PiState.OCR || currentGen != ocrGeneration) return@withContext

                        ocrStatusText.text = "OCR: ${result.lineCount} lines | ${result.totalTimeMs} ms"
                        ocrResultText.text = if (rawText.isNotBlank()) rawText else "No readable text detected"

                        Log.d(TAG, "[NETRA_OCR] number of OCR results: ${result.results.size} lines=${result.lineCount} text=$rawText")

                        if (speechText.isNotBlank()) {
                            ocrSpeaking.set(true)
                            ocrTts?.speak(speechText) {
                                if (currentGen == ocrGeneration) {
                                    ocrSpeaking.set(false)
                                    ocrNextAllowedTimeMs = SystemClock.elapsedRealtime() + 2000L
                                }
                            }
                        } else {
                            ocrSpeaking.set(false)
                            ocrNextAllowedTimeMs = SystemClock.elapsedRealtime() + 2000L
                        }
                    }
                } catch (e: Exception) {
                    Log.e(TAG, "[NETRA_OCR] OCR processing failed", e)

                    withContext(Dispatchers.Main) {
                        if (currentState == PiState.OCR && currentGen == ocrGeneration) {
                            ocrStatusText.text = "OCR: Error"
                            ocrSpeaking.set(false)
                        }
                    }
                } finally {
                    bitmap.recycle()
                    ocrProcessing.set(false)
                }
            }

        } catch (e: Exception) {
            Log.e(TAG, "[NETRA_OCR] Frame conversion failed", e)
            ocrProcessing.set(false)
        } finally {
            imageProxy.close()
        }
    }

    private fun imageProxyToBitmap(imageProxy: ImageProxy): Bitmap {
        val plane = imageProxy.planes.firstOrNull()
            ?: throw IllegalStateException("OCR frame has no image plane")

        val width = imageProxy.width
        val height = imageProxy.height
        val rowStride = plane.rowStride
        val pixelStride = plane.pixelStride
        val rotationDegrees = imageProxy.imageInfo.rotationDegrees

        Log.i(
            TAG,
            "NETRA_OCR_DIAG: format=${imageProxy.format} planes=${imageProxy.planes.size} " +
                    "w=$width h=$height rowStride=$rowStride pixelStride=$pixelStride rot=$rotationDegrees"
        )

        val buffer = plane.buffer
        buffer.rewind()

        // Extract clean pixel buffer handling rowStride padding without buffer underflows or corrupted pixel offsets
        val cleanBuffer = if (rowStride == width * pixelStride) {
            buffer
        } else {
            val packed = java.nio.ByteBuffer.allocateDirect(width * height * pixelStride)
            val row = ByteArray(width * pixelStride)
            for (y in 0 until height) {
                buffer.position(y * rowStride)
                buffer.get(row, 0, width * pixelStride)
                packed.put(row)
            }
            packed.rewind()
            packed
        }

        val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        bitmap.copyPixelsFromBuffer(cleanBuffer)

        // Apply rotation if required
        val rotatedBitmap = if (rotationDegrees != 0) {
            val matrix = android.graphics.Matrix()
            matrix.postRotate(rotationDegrees.toFloat())
            Bitmap.createBitmap(bitmap, 0, 0, width, height, matrix, true).also {
                if (it != bitmap) bitmap.recycle()
            }
        } else {
            bitmap
        }

        // Scale down high-resolution frames (e.g., 2176x2176) to max side length 1280
        // to reduce PaddleOCR inference time from 17s down to ~0.5s-1s
        val maxDim = maxOf(rotatedBitmap.width, rotatedBitmap.height)
        val finalBitmap = if (maxDim > 1280) {
            val scale = 1280.0f / maxDim
            val targetW = (rotatedBitmap.width * scale).toInt()
            val targetH = (rotatedBitmap.height * scale).toInt()
            Bitmap.createScaledBitmap(rotatedBitmap, targetW, targetH, true).also {
                if (it != rotatedBitmap) rotatedBitmap.recycle()
            }
        } else {
            rotatedBitmap
        }

        Log.i(TAG, "NETRA_OCR_DIAG: Final bitmap config=${finalBitmap.config} size=${finalBitmap.width}x${finalBitmap.height}")

        // TASK 2: Temporarily save ONE converted OCR bitmap to cacheDir/netra_ocr_debug.jpg
        if (!debugFrameSaved) {
            debugFrameSaved = true
            try {
                val debugFile = File(cacheDir, "netra_ocr_debug.jpg")
                FileOutputStream(debugFile).use { out ->
                    finalBitmap.compress(Bitmap.CompressFormat.JPEG, 90, out)
                }
                Log.i(TAG, "NETRA_OCR: Saved debug frame to ${debugFile.absolutePath}")
            } catch (e: Exception) {
                Log.e(TAG, "NETRA_OCR: Failed to save debug frame", e)
            }
        }

        return finalBitmap
    }

    // ============================================================
    // RASPBERRY PI UDP OBJECT RECOGNITION
    // ============================================================

    private fun startPiDetectionCycle() {
        if (
            currentState != PiState.IDLE &&
            currentState != PiState.ERROR
        ) {
            Log.d(TAG, "[NETRA_OBJECT] Recognition already running")
            return
        }

        val piHost = piIpAddress.text.toString().trim()

        if (piHost.isEmpty()) {
            setPiState(
                PiState.ERROR,
                "Raspberry Pi: Enter IP address",
                true
            )
            return
        }

        sharedPreferences
            .edit()
            .putString(PREF_KEY_PI_IP, piHost)
            .apply()

        stopCamera()

        detectionOverlay.clearDetections()
        frameStatsText.text = "Frames: 0"
        countText.text = "Objects Detected: 0"
        breakdownText.text = "Detections: Waiting..."
        ttsStatusText.text = "TTS: Ready"
        inferenceText.text = "Inference: -- ms"
        ramText.text = "RAM: ${calculateRamUsageMb()}MB"

        piFrameCount = 0
        piLastInferenceTimeMs = 0L
        piFrameProcessing.set(false)

        setPiState(
            PiState.STARTING,
            "Raspberry Pi: Starting UDP camera..."
        )

        Log.i(
            TAG,
            "NETRA_OBJECT starting Pi UDP recognition: $piHost:$PI_UDP_PORT"
        )

        piVideoReceiver?.stop()
        piVideoReceiver = null

        val receiver = PiVideoReceiver(
            width = 640,
            height = 480,
            port = PI_UDP_PORT,
            onVideoFrame = { bitmap ->
                processPiVideoFrame(bitmap)
            },
            onStatus = { status ->
                runOnUiThread {
                    piStatusText.text = status
                }
                Log.d(TAG, "NETRA_UDP $status")
            },
            onStats = { frameCount, fps ->
                runOnUiThread {
                    frameStatsText.text =
                        "Frames: $frameCount | FPS: ${"%.1f".format(Locale.US, fps)}"
                    fpsText.text =
                        "FPS: ${"%.1f".format(Locale.US, fps)}"
                    ramText.text =
                        "RAM: ${calculateRamUsageMb()}MB"
                }
            }
        )

        piVideoReceiver = receiver
        receiver.start()

        piDetectionJob = lifecycleScope.launch(Dispatchers.IO) {
            delay(100)

            if (!isPiMode) {
                return@launch
            }

            val started = receiver.sendStart(piHost)

            withContext(Dispatchers.Main) {
                if (!started) {
                    setPiState(
                        PiState.ERROR,
                        "Raspberry Pi: Failed to send START",
                        true
                    )

                    receiver.stop()

                    if (piVideoReceiver === receiver) {
                        piVideoReceiver = null
                    }

                    piDetectionJob = null
                    return@withContext
                }

                setPiState(
                    PiState.OBJECT_RECOGNITION,
                    "Raspberry Pi: Receiving camera stream"
                )

                Log.i(
                    TAG,
                    "NETRA_OBJECT START sent to $piHost:$PI_UDP_PORT"
                )
            }
        }
    }

    private fun processPiVideoFrame(bitmap: Bitmap) {
        if (!piFrameProcessing.compareAndSet(false, true)) {
            if (!bitmap.isRecycled) {
                bitmap.recycle()
            }
            return
        }

        lifecycleScope.launch(Dispatchers.Default) {
            try {
                val preprocessor = imagePreprocessor
                    ?: throw Exception("Image preprocessor is not initialized")

                val detector = objectDetector
                    ?: throw Exception("Object detector is not initialized")

                val preprocessResult = preprocessor.preprocess(bitmap)
                val detectionResult = detector.detect(preprocessResult)
                val detections = detectionResult.detections

                piFrameCount++
                piLastInferenceTimeMs = detectionResult.inferenceTimeMs.toLong()

                Log.d(
                    TAG,
                    "NETRA_OBJECT frame=$piFrameCount " +
                        "detections=${detections.size} " +
                        "inference=${detectionResult.inferenceTimeMs}ms"
                )

                withContext(Dispatchers.Main) {
                    if (!bitmap.isRecycled) {
                        frameView.setImageBitmap(bitmap)
                    }

                    detectionOverlay.setDetections(
                        detections,
                        bitmap.width,
                        bitmap.height
                    )

                    frameStatsText.text = "Frames: $piFrameCount"
                    countText.text = "Objects Detected: ${detections.size}"
                    breakdownText.text =
                        if (detections.isEmpty()) {
                            "Detections: None"
                        } else {
                            "Detections: ${buildCountsSummary(detections)}"
                        }

                    inferenceText.text =
                        "Inference: ${detectionResult.inferenceTimeMs} ms"

                    ramText.text =
                        "RAM: ${calculateRamUsageMb()}MB"

                    announcementManager?.onDetections(detections)
                }
            } catch (e: Exception) {
                Log.e(
                    TAG,
                    "NETRA_OBJECT frame processing failed",
                    e
                )

                if (!bitmap.isRecycled) {
                    bitmap.recycle()
                }
            } finally {
                piFrameProcessing.set(false)
            }
        }
    }

    
    private fun startPiOcrCycle() {
        if (currentState != PiState.IDLE && currentState != PiState.ERROR) {
            Log.d(TAG, "[NETRA_OCR] OCR or Recognition already running")
            return
        }

        val piHost = piIpAddress.text.toString().trim()
        if (piHost.isEmpty()) {
            setPiState(PiState.ERROR, "Raspberry Pi: Enter IP address", true)
            return
        }

        sharedPreferences.edit().putString(PREF_KEY_PI_IP, piHost).apply()

        stopCamera()
        stopOcrMode()

        detectionOverlay.clearDetections()
        frameStatsText.text = "Frames: 0"
        countText.text = "Lines Detected: 0"
        breakdownText.text = "OCR: Waiting..."
        ttsStatusText.text = "TTS: Ready"
        inferenceText.text = "Inference: -- ms"
        ocrResultText.text = ""
        ocrStatusText.text = "OCR: Active"

        piFrameCount = 0
        piLastInferenceTimeMs = 0L
        piFrameProcessing.set(false)

        setPiState(PiState.STARTING, "Raspberry Pi: Starting UDP camera (OCR)...")

        piVideoReceiver?.stop()
        piVideoReceiver = null

        val receiver = PiVideoReceiver(
            width = 640,
            height = 480,
            port = PI_UDP_PORT,
            onVideoFrame = { bitmap ->
                processPiOcrFrame(bitmap)
            },
            onStatus = { status ->
                runOnUiThread { piStatusText.text = status }
            },
            onStats = { frameCount, fps ->
                runOnUiThread {
                    frameStatsText.text = "Frames: $frameCount | FPS: ${"%.1f".format(Locale.US, fps)}"
                    fpsText.text = "FPS: ${"%.1f".format(Locale.US, fps)}"
                }
            }
        )

        piVideoReceiver = receiver
        receiver.start()

        currentState = PiState.OCR

        piDetectionJob = lifecycleScope.launch(Dispatchers.IO) {
            delay(100)
            if (!isPiMode) return@launch

            val started = receiver.sendStart(piHost)

            withContext(Dispatchers.Main) {
                if (!started) {
                    setPiState(PiState.ERROR, "Raspberry Pi: Failed to send START", true)
                    receiver.stop()
                    if (piVideoReceiver === receiver) piVideoReceiver = null
                    piDetectionJob = null
                    return@withContext
                }
                setPiState(PiState.OCR, "Raspberry Pi: OCR active")
            }
        }
    }

    private fun processPiOcrFrame(bitmap: Bitmap) {
        val now = SystemClock.elapsedRealtime()

        if (ocrSpeaking.get() || now < ocrNextAllowedTimeMs) {
            if (!bitmap.isRecycled) bitmap.recycle()
            return
        }

        if (now - lastOcrTimeMs < 1200L) {
            if (!bitmap.isRecycled) bitmap.recycle()
            return
        }

        if (!ocrProcessing.compareAndSet(false, true)) {
            if (!bitmap.isRecycled) bitmap.recycle()
            return
        }

        lastOcrTimeMs = now
        val currentGen = ocrGeneration
        val engine = ocrEngine

        if (engine == null || !ocrInitialized || currentState != PiState.OCR) {
            ocrProcessing.set(false)
            if (!bitmap.isRecycled) bitmap.recycle()
            return
        }

        lifecycleScope.launch(Dispatchers.IO) {
            try {
                val result = engine.recognize(bitmap)

                if (currentGen != ocrGeneration) return@launch

                val rawText = result.results.joinToString("\n") { it.text }
                val speechText = SpeechTextFormatter.format(result.results)

                withContext(Dispatchers.Main) {
                    if (!bitmap.isRecycled) {
                        frameView.setImageBitmap(bitmap)
                    }

                    if (currentState != PiState.OCR || currentGen != ocrGeneration) return@withContext

                    ocrStatusText.text = "OCR: ${result.lineCount} lines | ${result.totalTimeMs} ms"
                    ocrResultText.text = if (rawText.isNotBlank()) rawText else "No readable text detected"

                    if (speechText.isNotBlank()) {
                        ocrSpeaking.set(true)
                        ocrTts?.speak(speechText) {
                            if (currentGen == ocrGeneration) {
                                ocrSpeaking.set(false)
                                ocrNextAllowedTimeMs = SystemClock.elapsedRealtime() + 2000L
                            }
                        }
                    } else {
                        ocrSpeaking.set(false)
                        ocrNextAllowedTimeMs = SystemClock.elapsedRealtime() + 2000L
                    }
                }
            } catch (e: Exception) {
                Log.e(TAG, "[NETRA_OCR] Pi OCR processing failed", e)
                withContext(Dispatchers.Main) {
                    if (currentState == PiState.OCR && currentGen == ocrGeneration) {
                        ocrStatusText.text = "OCR: Error"
                        ocrSpeaking.set(false)
                    }
                    if (!bitmap.isRecycled) {
                        frameView.setImageBitmap(bitmap)
                    }
                }
            } finally {
                ocrProcessing.set(false)
            }
        }
    }


    private fun cancelPiJob() {
        piDetectionJob?.cancel()
        piDetectionJob = null

        val host = piIpAddress.text.toString().trim()

        if (host.isNotEmpty()) {
            try {
                piVideoReceiver?.sendStop(host)
            } catch (e: Exception) {
                Log.d(
                    TAG,
                    "NETRA_UDP STOP command failed: ${e.message}"
                )
            }
        }

        piVideoReceiver?.stop()
        piVideoReceiver = null

        piFrameProcessing.set(false)

        setPiState(
            PiState.IDLE,
            "Raspberry Pi: Ready"
        )

        Log.i(
            TAG,
            "NETRA_OBJECT Pi recognition stopped"
        )
    }

    // ============================================================
    // DETECTION HELPERS
    // ============================================================

    // ============================================================
    // DETECTION HELPERS
    // ============================================================

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

        try { unregisterReceiver(piSignalReceiver) } catch (_: Exception) {}
        sosManager.destroy()
        super.onDestroy()
    }
}











