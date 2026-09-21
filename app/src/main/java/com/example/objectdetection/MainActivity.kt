package com.example.objectdetection

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.os.SystemClock
import android.util.Log
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
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {

    companion object {
        private const val TAG = "MainActivity"
    }

    // UI Components
    private lateinit var previewView: PreviewView
    private lateinit var detectionOverlay: DetectionOverlay
    private lateinit var statusText: TextView
    private lateinit var countText: TextView
    private lateinit var breakdownText: TextView
    private lateinit var fpsText: TextView
    private lateinit var inferenceText: TextView
    private lateinit var ramText: TextView
    private lateinit var ttsStatusText: TextView

    // Pipelines and Models
    private var objectDetector: ObjectDetector? = null
    private var imagePreprocessor: ImagePreprocessor? = null
    private var piperTTS: PiperTTS? = null
    private var announcementManager: AnnouncementManager? = null

    // Background Executors
    private val cameraExecutor: ExecutorService = Executors.newSingleThreadExecutor()

    // Performance Metrics Tracking
    private var frameCount = 0
    private var lastFpsTimestamp = 0L
    private var currentFps = 0.0

    // Permission launcher
    private val cameraPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { isGranted ->
        if (isGranted) {
            startCamera()
        } else {
            Toast.makeText(this, "Camera permission is required for object detection", Toast.LENGTH_LONG).show()
            statusText.text = "Status: Camera permission denied"
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        // Initialize UI Views
        previewView = findViewById(R.id.previewView)
        detectionOverlay = findViewById(R.id.detectionOverlay)
        statusText = findViewById(R.id.statusText)
        countText = findViewById(R.id.countText)
        breakdownText = findViewById(R.id.breakdownText)
        fpsText = findViewById(R.id.fpsText)
        inferenceText = findViewById(R.id.inferenceText)
        ramText = findViewById(R.id.ramText)
        ttsStatusText = findViewById(R.id.ttsStatusText)

        initializeModels()

        // Check camera permission
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
            startCamera()
        } else {
            cameraPermissionLauncher.launch(Manifest.permission.CAMERA)
        }
    }

    /**
     * Initializes ONNX models ONCE during app startup.
     */
    private fun initializeModels() {
        try {
            statusText.text = "Status: Loading ONNX models..."

            // 1. Initialize YOLO Image Preprocessor
            imagePreprocessor = ImagePreprocessor(targetWidth = 640, targetHeight = 640)

            // 2. Initialize Object Detection Model
            objectDetector = ObjectDetector(
                context = this,
                confidenceThreshold = 0.40f,
                iouThreshold = 0.45f
            )

            // 3. Initialize Piper TTS Model
            val tts = PiperTTS(this)
            piperTTS = tts

            // 4. Initialize Announcement Manager
            announcementManager = AnnouncementManager(tts).apply {
                onAnnouncementStateChanged = { phrase ->
                    runOnUiThread {
                        val ttsMs = tts.getLastInferenceTimeMs()
                        ttsStatusText.text = "TTS (${ttsMs}ms): \"$phrase\""
                    }
                }
            }

            statusText.text = "Status: Models Ready"
            statusText.setTextColor(ContextCompat.getColor(this, android.R.color.holo_green_light))
            Log.i(TAG, "All models and components successfully initialized.")
        } catch (e: Exception) {
            Log.e(TAG, "Initialization failed", e)
            statusText.text = "Status: Init Error - ${e.message}"
            statusText.setTextColor(ContextCompat.getColor(this, android.R.color.holo_red_light))
            Toast.makeText(this, "Model init error: ${e.message}", Toast.LENGTH_LONG).show()
        }
    }

    /**
     * Configures CameraX with Preview and ImageAnalysis use cases using STRATEGY_KEEP_ONLY_LATEST.
     */
    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)

        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()

            // Preview use case
            val preview = Preview.Builder()
                .build()
                .also {
                    it.surfaceProvider = previewView.surfaceProvider
                }

            // ImageAnalysis use case with non-blocking STRATEGY_KEEP_ONLY_LATEST
            val imageAnalysis = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .build()
                .also { analysis ->
                    analysis.setAnalyzer(cameraExecutor) { imageProxy ->
                        processImageFrame(imageProxy)
                    }
                }

            // Select back camera
            val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(
                    this,
                    cameraSelector,
                    preview,
                    imageAnalysis
                )
                Log.i(TAG, "Camera bound successfully to lifecycle.")
            } catch (e: Exception) {
                Log.e(TAG, "Use case binding failed", e)
            }
        }, ContextCompat.getMainExecutor(this))
    }

    /**
     * Frame-by-frame inference pipeline executed on background thread.
     */
    private fun processImageFrame(imageProxy: ImageProxy) {
        val detector = objectDetector
        val preprocessor = imagePreprocessor

        if (detector == null || preprocessor == null) {
            imageProxy.close()
            return
        }

        try {
            // 1. Measure FPS
            calculateFps()

            // 2. Preprocess frame (Letterbox 640x640 + Rotation handling)
            val preprocessResult = preprocessor.preprocess(imageProxy)

            // 3. Run Object Detection Inference & Decode
            val detectionResult = detector.detect(preprocessResult)
            val detections = detectionResult.detections
            val inferenceTimeMs = detectionResult.inferenceTimeMs

            // 4. Update Announcement Manager (runs debounce/cooldown logic)
            announcementManager?.onDetections(detections)

            // 5. Update UI on Main Thread
            val ramUsageMb = calculateRamUsageMb()
            val countsSummary = buildCountsSummary(detections)

            runOnUiThread {
                detectionOverlay.setDetections(
                    detections,
                    preprocessResult.sourceWidth,
                    preprocessResult.sourceHeight
                )

                countText.text = "Total Objects: ${detections.size}"
                breakdownText.text = "Counts: $countsSummary"
                fpsText.text = "FPS: ${"%.1f".format(currentFps)}"
                inferenceText.text = "Inference: ${inferenceTimeMs}ms"
                ramText.text = "RAM: ${ramUsageMb}MB"
            }
        } catch (e: Exception) {
            Log.e(TAG, "Frame processing error", e)
        } finally {
            imageProxy.close()
        }
    }

    private fun buildCountsSummary(detections: List<Detection>): String {
        if (detections.isEmpty()) return "None"

        val counts = mutableMapOf<String, Int>()
        for (d in detections) {
            counts[d.label] = (counts[d.label] ?: 0) + 1
        }

        return counts.entries.joinToString(" | ") { "${it.key}: ${it.value}" }
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
        cameraExecutor.shutdown()
        imagePreprocessor?.release()
        objectDetector?.close()
        piperTTS?.close()
        announcementManager?.reset()
    }
}