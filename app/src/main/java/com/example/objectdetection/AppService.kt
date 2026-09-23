package com.example.objectdetection

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Binder
import android.os.Build
import android.os.Handler
import android.os.IBinder
import android.os.Looper
import android.os.SystemClock
import android.util.Log
import androidx.core.app.NotificationCompat
import com.example.objectdetection.safety.ObstacleAnalyzer
import com.example.objectdetection.safety.SafetyManager
import com.example.objectdetection.safety.TofFrame
import com.example.objectdetection.safety.TofSensorManager
import java.util.concurrent.CopyOnWriteArrayList
import java.util.concurrent.atomic.AtomicBoolean

/**
 * Robust Android Foreground Service that maintains the local TCP frame/control server,
 * coordinates operating modes, executes ONNX inference (Object Detection & OCR),
 * manages the independent VL53L5CX Safety Subsystem, and triggers SOS dispatch in the background.
 */
class AppService : Service() {

    companion object {
        private const val TAG = "AppService"
        private const val NOTIFICATION_CHANNEL_ID = "hello_netra_service_channel"
        private const val NOTIFICATION_ID = 1001
        private const val SERVER_PORT = 5000
        private const val CAMERA_SESSION_TIMEOUT_MS = 120_000L // 120-second automatic camera session timeout

        const val ACTION_START_SERVICE = "com.example.objectdetection.action.START_SERVICE"
        const val ACTION_STOP_SERVICE = "com.example.objectdetection.action.STOP_SERVICE"
    }

    inner class LocalBinder : Binder() {
        val service: AppService get() = this@AppService
    }

    private val binder = LocalBinder()
    private val mainHandler = Handler(Looper.getMainLooper())

    // Core Controllers & Repositories
    val modeController = ModeController()
    val safetyManager = SafetyManager()
    lateinit var contactRepository: EmergencyContactRepository
        private set
    lateinit var sosManager: SosManager
        private set

    // ONNX Vision & Speech Pipelines
    var objectDetector: ObjectDetector? = null
        private set
    var imagePreprocessor: ImagePreprocessor? = null
        private set
    var ocrRecognizer: OcrRecognizer? = null
        private set
    var piperTTS: PiperTTS? = null
        private set
    var announcementManager: AnnouncementManager? = null
        private set

    // Frame Server & Buffer
    var localFrameServer: LocalFrameServer? = null
        private set
    val frameBuffer = LatestFrameBuffer()

    // Worker Thread
    private val isProcessing = AtomicBoolean(false)
    private var processingThread: Thread? = null

    // Performance Metrics
    private var frameCount = 0
    private var lastFpsTimestamp = 0L
    var currentFps = 0.0
        private set

    // OCR debouncing
    private var lastOcrText = ""
    private var lastOcrTimestamp = 0L
    private val ocrCooldownMs = 4000L

    // Camera Session Auto-Timeout Runnable (120 seconds)
    private val sessionTimeoutRunnable = Runnable {
        Log.i(TAG, "120-second camera session expired. Returning to IDLE mode.")
        piperTTS?.speak("Session complete.")
        modeController.setMode(AppMode.IDLE)
    }

    // UI Observer Interface
    interface ServiceUiListener {
        fun onFrameProcessed(bitmap: Bitmap, detections: List<Detection>, inferenceMs: Long, ramMb: Long)
        fun onOcrProcessed(bitmap: Bitmap, text: String, confidence: Float, inferenceMs: Long, ramMb: Long)
        fun onConnectionStateChanged(state: LocalFrameServer.ConnectionState)
        fun onModeChanged(mode: AppMode)
        fun onSosStatusChanged(status: SosManager.SosStatus)
        fun onTtsStatus(message: String)
        fun onTofTelemetryUpdated(frame: TofFrame, analysis: ObstacleAnalyzer.SpatialAnalysis, updateRateHz: Double)
        fun onTofStatusChanged(status: TofSensorManager.SensorStatus)
    }

    private val uiListeners = CopyOnWriteArrayList<ServiceUiListener>()

    fun addUiListener(listener: ServiceUiListener) {
        uiListeners.add(listener)
    }

    fun removeUiListener(listener: ServiceUiListener) {
        uiListeners.remove(listener)
    }

    override fun onCreate() {
        super.onCreate()
        Log.i(TAG, "Initializing Hello-Netra AppService...")

        createNotificationChannel()
        startForeground(NOTIFICATION_ID, buildNotification("Starting Hello-Netra Assistant..."))

        contactRepository = EmergencyContactRepository(this)
        initializePipelines()
        setupModeController()
        setupSafetyManager()
        startFrameProcessingWorker()
        startLocalFrameServer()

        updateNotification("Ready for hardware commands (Mode: IDLE)")
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        if (intent?.action == ACTION_STOP_SERVICE) {
            stopSelf()
            return START_NOT_STICKY
        }
        return START_STICKY
    }

    override fun onBind(intent: Intent?): IBinder = binder

    private fun createNotificationChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val channel = NotificationChannel(
                NOTIFICATION_CHANNEL_ID,
                "Hello-Netra Background Service",
                NotificationManager.IMPORTANCE_LOW
            ).apply {
                description = "Keeps the local frame server, safety sensor, and hardware communication active."
                setShowBadge(false)
            }
            val manager = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
            manager.createNotificationChannel(channel)
        }
    }

    private fun buildNotification(statusText: String): Notification {
        val launchIntent = Intent(this, MainActivity::class.java)
        val pendingIntent = PendingIntent.getActivity(
            this,
            0,
            launchIntent,
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
        )

        return NotificationCompat.Builder(this, NOTIFICATION_CHANNEL_ID)
            .setContentTitle("Hello-Netra Assistant")
            .setContentText(statusText)
            .setSmallIcon(android.R.drawable.ic_menu_camera)
            .setContentIntent(pendingIntent)
            .setOngoing(true)
            .setPriority(NotificationCompat.PRIORITY_LOW)
            .build()
    }

    private fun updateNotification(statusText: String) {
        val manager = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
        manager.notify(NOTIFICATION_ID, buildNotification(statusText))
    }

    private fun initializePipelines() {
        try {
            // 1. YOLO Letterbox Preprocessor
            imagePreprocessor = ImagePreprocessor(targetWidth = 640, targetHeight = 640)

            // 2. YOLOv8 Object Detector (strict 60% confidence filter)
            objectDetector = ObjectDetector(
                context = this,
                confidenceThreshold = 0.60f,
                iouThreshold = 0.45f
            )

            // 3. PP-OCRv5 Recognition Engine
            ocrRecognizer = OcrRecognizer(context = this)

            // 4. Piper ONNX TTS Engine
            val tts = PiperTTS(this)
            piperTTS = tts

            // 5. Announcement Manager (Debounce & State-Change rules for Object Detection)
            announcementManager = AnnouncementManager(tts).apply {
                onAnnouncementStateChanged = { phrase ->
                    val ttsMs = tts.getLastInferenceTimeMs()
                    notifyTtsStatus("TTS (${ttsMs}ms): \"$phrase\"")
                }
            }

            // 6. SOS Manager
            sosManager = SosManager(this, contactRepository, tts).apply {
                onStatusChanged = { status ->
                    for (listener in uiListeners) {
                        listener.onSosStatusChanged(status)
                    }
                    updateNotification(status.message)
                }
            }

            Log.i(TAG, "All AI, Vision, and Safety pipelines initialized successfully.")
        } catch (e: Exception) {
            Log.e(TAG, "Error initializing AI pipelines", e)
        }
    }

    private fun setupModeController() {
        modeController.addListener(object : ModeController.ModeChangeListener {
            override fun onModeChanged(previousMode: AppMode, newMode: AppMode) {
                Log.i(TAG, "AppService handled mode switch: $previousMode -> $newMode")
                updateNotification("Active Mode: $newMode")

                // Reset session timeout timer
                mainHandler.removeCallbacks(sessionTimeoutRunnable)

                when (newMode) {
                    AppMode.IDLE -> {
                        announcementManager?.reset()
                        piperTTS?.speak("System idle.")
                    }
                    AppMode.OBJECT_RECOGNITION -> {
                        announcementManager?.reset()
                        piperTTS?.speak("Object recognition mode.")
                        // Start 120-second automatic session timer
                        mainHandler.postDelayed(sessionTimeoutRunnable, CAMERA_SESSION_TIMEOUT_MS)
                    }
                    AppMode.OCR -> {
                        lastOcrText = ""
                        lastOcrTimestamp = 0L
                        piperTTS?.speak("O C R reading mode.")
                        // Start 120-second automatic session timer
                        mainHandler.postDelayed(sessionTimeoutRunnable, CAMERA_SESSION_TIMEOUT_MS)
                    }
                    AppMode.SOS -> {
                        sosManager.triggerSos()
                    }
                }

                for (listener in uiListeners) {
                    listener.onModeChanged(newMode)
                }
            }
        })
    }

    private fun setupSafetyManager() {
        safetyManager.addObserver(object : SafetyManager.SafetyObserver {
            override fun onTofTelemetryUpdated(
                frame: TofFrame,
                analysis: ObstacleAnalyzer.SpatialAnalysis,
                updateRateHz: Double
            ) {
                for (listener in uiListeners) {
                    listener.onTofTelemetryUpdated(frame, analysis, updateRateHz)
                }
            }

            override fun onSafetyStatusChanged(status: TofSensorManager.SensorStatus) {
                for (listener in uiListeners) {
                    listener.onTofStatusChanged(status)
                }
            }
        })
    }

    private fun startLocalFrameServer() {
        localFrameServer = LocalFrameServer(
            port = SERVER_PORT,
            onConnectionStateChanged = { state ->
                for (listener in uiListeners) {
                    listener.onConnectionStateChanged(state)
                }
            },
            onControlCommandReceived = { mode ->
                modeController.setMode(mode)
            },
            onFrameReceived = { packet ->
                frameBuffer.offer(packet)
            },
            onTofBinaryReceived = { data, offset, len ->
                safetyManager.ingestBinaryData(data, offset, len)
            },
            onTofTextReceived = { text ->
                safetyManager.ingestTextData(text)
            }
        ).apply {
            start()
        }
    }

    private fun startFrameProcessingWorker() {
        isProcessing.set(true)
        processingThread = Thread {
            while (isProcessing.get()) {
                try {
                    val packet = frameBuffer.take()
                    val currentMode = modeController.currentMode

                    // If IDLE or SOS, do not run heavy neural network inference
                    if (currentMode == AppMode.IDLE || currentMode == AppMode.SOS) {
                        continue
                    }

                    val bitmap = BitmapFactory.decodeByteArray(
                        packet.jpegData,
                        0,
                        packet.jpegData.size
                    ) ?: continue

                    calculateFps()
                    val ramUsageMb = calculateRamUsageMb()

                    when (currentMode) {
                        AppMode.OBJECT_RECOGNITION -> {
                            val detector = objectDetector
                            val preprocessor = imagePreprocessor
                            val preprocessResult = preprocessor?.preprocess(bitmap)

                            val detectionResult = if (preprocessResult != null && detector != null) {
                                detector.detect(preprocessResult)
                            } else {
                                ObjectDetector.DetectionResult(emptyList(), 0)
                            }

                            val detections = detectionResult.detections
                            val inferenceMs = detectionResult.inferenceTimeMs

                            announcementManager?.onDetections(detections)

                            for (listener in uiListeners) {
                                listener.onFrameProcessed(bitmap, detections, inferenceMs, ramUsageMb)
                            }
                        }

                        AppMode.OCR -> {
                            val recognizer = ocrRecognizer
                            val ocrResult = recognizer?.recognize(bitmap) ?: OcrRecognizer.OcrResult("", 0f, 0)
                            val recognizedText = ocrResult.text
                            val inferenceMs = ocrResult.inferenceTimeMs
                            val confidence = ocrResult.confidence

                            handleOcrSpeech(recognizedText, confidence)

                            for (listener in uiListeners) {
                                listener.onOcrProcessed(bitmap, recognizedText, confidence, inferenceMs, ramUsageMb)
                            }
                        }

                        else -> {}
                    }

                } catch (_: InterruptedException) {
                    break
                } catch (e: Exception) {
                    Log.e(TAG, "Error in frame processing worker", e)
                }
            }
        }.apply {
            name = "AppServiceFrameWorker"
            isDaemon = true
            start()
        }
    }

    private fun handleOcrSpeech(text: String, confidence: Float) {
        val clean = text.trim()
        if (clean.length < 2 || confidence < 0.35f) return

        val now = System.currentTimeMillis()
        if (clean != lastOcrText && (now - lastOcrTimestamp >= ocrCooldownMs)) {
            lastOcrText = clean
            lastOcrTimestamp = now
            Log.i(TAG, "Speaking OCR text: \"$clean\" (confidence: $confidence)")
            notifyTtsStatus("TTS OCR: \"$clean\"")
            piperTTS?.speak(clean)
        }
    }

    private fun notifyTtsStatus(msg: String) {
        for (listener in uiListeners) {
            listener.onTtsStatus(msg)
        }
    }

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
        val usedMemoryBytes = runtime.totalMemory() - runtime.freeMemory()
        return usedMemoryBytes / (1024 * 1024)
    }

    override fun onDestroy() {
        super.onDestroy()
        mainHandler.removeCallbacks(sessionTimeoutRunnable)
        isProcessing.set(false)
        processingThread?.interrupt()
        localFrameServer?.release()
        safetyManager.release()
        imagePreprocessor?.release()
        objectDetector?.close()
        ocrRecognizer?.close()
        piperTTS?.close()
        announcementManager?.reset()
    }
}
