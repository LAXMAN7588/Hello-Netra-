package com.hellonetra.ocr

import android.Manifest
import android.app.Activity
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageFormat
import android.graphics.Matrix
import android.graphics.RectF
import android.graphics.SurfaceTexture
import android.hardware.camera2.CameraCaptureSession
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraDevice
import android.hardware.camera2.CameraManager
import android.hardware.camera2.CaptureRequest
import android.media.ExifInterface
import android.media.ImageReader
import android.os.Bundle
import android.os.Handler
import android.os.HandlerThread
import android.util.Size
import android.util.SparseIntArray
import android.view.Surface
import android.view.TextureView
import android.widget.Button
import android.widget.TextView
import com.paddle.ocr.EngineConfig
import com.paddle.ocr.PaddleOCR
import com.paddle.ocr.PaddleOCRConfig
import com.paddle.ocr.util.OpenCVUtils
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.launch
import java.io.ByteArrayInputStream
import kotlin.math.abs
import kotlin.math.max

class MainActivity : Activity() {

    companion object {
        private const val CAMERA_PERMISSION_REQUEST = 1001

        private val ORIENTATIONS = SparseIntArray().apply {
            append(Surface.ROTATION_0, 0)
            append(Surface.ROTATION_90, 90)
            append(Surface.ROTATION_180, 180)
            append(Surface.ROTATION_270, 270)
        }
    }

    private lateinit var textureView: TextureView
    private lateinit var scanButton: Button
    private lateinit var statusText: TextView
    private lateinit var resultText: TextView

    private val appScope = CoroutineScope(
        SupervisorJob() + Dispatchers.Main.immediate
    )

    private var ocr: PaddleOCR? = null

    private var cameraDevice: CameraDevice? = null
    private var cameraSession: CameraCaptureSession? = null
    private var cameraId: String? = null
    private var imageReader: ImageReader? = null
    private var cameraThread: HandlerThread? = null
    private var cameraHandler: Handler? = null
    private var previewSurface: Surface? = null
    private var cameraCharacteristics: CameraCharacteristics? = null
    private var previewSize: Size? = null
    private var captureSize: Size? = null

    private var scanInProgress = false

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        textureView = findViewById(R.id.textureView)
        scanButton = findViewById(R.id.scanButton)
        statusText = findViewById(R.id.statusText)
        resultText = findViewById(R.id.resultText)

        statusText.text = "Loading OCR models..."
        scanButton.isEnabled = false

        textureView.surfaceTextureListener = surfaceTextureListener

        scanButton.setOnClickListener {
            captureText()
        }

        loadOcr()
    }

    private fun loadOcr() {
        appScope.launch {
            try {
                statusText.text = "Initializing OpenCV..."

                val opencvReady = OpenCVUtils.init(this@MainActivity)

                if (!opencvReady) {
                    statusText.text = "OpenCV initialization failed"
                    resultText.text = "Could not load OpenCV native library."
                    return@launch
                }

                statusText.text = "Loading OCR models..."

                val engine = PaddleOCR.create(
                    context = this@MainActivity,
                    config = PaddleOCRConfig(
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
                        recBatchSize = 1,
                    ),
                    engineConfig = EngineConfig(
                        numThreads = 4,
                    ),
                    detModelAssetPath =
                        "det/ch_PP-OCRv5_det_mobile.onnx",
                    recModelAssetPath =
                        "rec/en_PP-OCRv5_rec_mobile.onnx",
                    recConfigAssetPath =
                        "rec/ppocrv5_en_rec.yml",
                )

                ocr = engine

                statusText.text = "OCR ready"

                if (checkSelfPermission(Manifest.permission.CAMERA)
                    != PackageManager.PERMISSION_GRANTED
                ) {
                    requestPermissions(
                        arrayOf(Manifest.permission.CAMERA),
                        CAMERA_PERMISSION_REQUEST
                    )
                } else {
                    if (textureView.isAvailable) {
                        openCamera(textureView.width, textureView.height)
                    }
                }

                scanButton.isEnabled = true

            } catch (t: Throwable) {
                statusText.text =
                    "OCR load failed: ${t.message ?: t.javaClass.simpleName}"
                resultText.text = ""
            }
        }
    }

    private val surfaceTextureListener = object : TextureView.SurfaceTextureListener {

        override fun onSurfaceTextureAvailable(
            surface: SurfaceTexture,
            width: Int,
            height: Int,
        ) {
            if (checkSelfPermission(Manifest.permission.CAMERA)
                == PackageManager.PERMISSION_GRANTED
            ) {
                openCamera(width, height)
            }
        }

        override fun onSurfaceTextureSizeChanged(
            surface: SurfaceTexture,
            width: Int,
            height: Int,
        ) {
            configureTextureTransform(width, height)
        }

        override fun onSurfaceTextureDestroyed(surface: SurfaceTexture): Boolean {
            return true
        }

        override fun onSurfaceTextureUpdated(surface: SurfaceTexture) {
        }
    }

    private fun startCameraThread() {
        if (cameraThread != null) return

        cameraThread = HandlerThread("HelloNetraCamera").also {
            it.start()
            cameraHandler = Handler(it.looper)
        }
    }

    private fun openCamera(viewWidth: Int, viewHeight: Int) {
        try {
            startCameraThread()

            val manager =
                getSystemService(Context.CAMERA_SERVICE) as CameraManager

            val selectedId = manager.cameraIdList.firstOrNull { id ->
                val chars = manager.getCameraCharacteristics(id)
                chars.get(CameraCharacteristics.LENS_FACING) ==
                    CameraCharacteristics.LENS_FACING_BACK
            } ?: manager.cameraIdList.firstOrNull()

            if (selectedId == null) {
                statusText.text = "No camera found"
                return
            }

            cameraId = selectedId

            val chars = manager.getCameraCharacteristics(selectedId)
            cameraCharacteristics = chars

            val map = chars.get(
                CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP
            )

            if (map == null) {
                statusText.text = "Camera configuration unavailable"
                return
            }

            val previewSizes =
                map.getOutputSizes(SurfaceTexture::class.java)

            val jpegSizes =
                map.getOutputSizes(ImageFormat.JPEG)

            previewSize = chooseSize(
                previewSizes,
                viewWidth,
                viewHeight
            )

            captureSize = chooseSize(
                jpegSizes,
                1280,
                720
            )

            val chosenPreviewSize = previewSize
                ?: Size(1280, 720)

            val chosenCaptureSize = captureSize
                ?: Size(1280, 720)

            textureView.surfaceTexture?.setDefaultBufferSize(
                chosenPreviewSize.width,
                chosenPreviewSize.height
            )

            configureTextureTransform(viewWidth, viewHeight)

            imageReader?.close()

            imageReader = ImageReader.newInstance(
                chosenCaptureSize.width,
                chosenCaptureSize.height,
                ImageFormat.JPEG,
                2
            )

            imageReader?.setOnImageAvailableListener(
                imageAvailableListener,
                cameraHandler
            )

            if (checkSelfPermission(Manifest.permission.CAMERA)
                != PackageManager.PERMISSION_GRANTED
            ) {
                return
            }

            manager.openCamera(
                selectedId,
                cameraStateCallback,
                cameraHandler
            )

            statusText.text = "Camera ready"

        } catch (t: Throwable) {
            statusText.text =
                "Camera error: ${t.message ?: t.javaClass.simpleName}"
        }
    }

    private fun chooseSize(
        sizes: Array<Size>?,
        targetWidth: Int,
        targetHeight: Int,
    ): Size? {
        if (sizes.isNullOrEmpty()) return null

        val targetRatio =
            targetWidth.toDouble() / targetHeight.toDouble()

        return sizes.minByOrNull { size ->
            val ratio =
                size.width.toDouble() / size.height.toDouble()

            val ratioError = abs(ratio - targetRatio)
            val sizeError =
                abs(size.width - targetWidth) +
                    abs(size.height - targetHeight)

            ratioError * 10000.0 + sizeError
        }
    }

    private val cameraStateCallback = object : CameraDevice.StateCallback() {

        override fun onOpened(camera: CameraDevice) {
            cameraDevice = camera
            createCameraSession()
        }

        override fun onDisconnected(camera: CameraDevice) {
            camera.close()

            if (cameraDevice === camera) {
                cameraDevice = null
            }
        }

        override fun onError(camera: CameraDevice, error: Int) {
            camera.close()

            if (cameraDevice === camera) {
                cameraDevice = null
            }

            statusText.text = "Camera error code: $error"
        }
    }

    private fun createCameraSession() {
        val camera = cameraDevice ?: return
        val texture = textureView.surfaceTexture ?: return
        val reader = imageReader ?: return

        try {
            val chosenPreviewSize =
                previewSize ?: Size(1280, 720)

            texture.setDefaultBufferSize(
                chosenPreviewSize.width,
                chosenPreviewSize.height
            )

            previewSurface?.release()

            previewSurface = Surface(texture)

            val sessionSurfaces = listOf(
                previewSurface!!,
                reader.surface
            )

            camera.createCaptureSession(
                sessionSurfaces,
                object : CameraCaptureSession.StateCallback() {

                    override fun onConfigured(
                        session: CameraCaptureSession
                    ) {
                        cameraSession = session
                        startPreview()
                    }

                    override fun onConfigureFailed(
                        session: CameraCaptureSession
                    ) {
                        statusText.text =
                            "Camera session failed"
                    }
                },
                cameraHandler
            )
        } catch (t: Throwable) {
            statusText.text =
                "Session error: ${t.message ?: t.javaClass.simpleName}"
        }
    }

    private fun startPreview() {
        val camera = cameraDevice ?: return
        val session = cameraSession ?: return
        val surface = previewSurface ?: return

        try {
            val request =
                camera.createCaptureRequest(
                    CameraDevice.TEMPLATE_PREVIEW
                ).apply {
                    addTarget(surface)

                    set(
                        CaptureRequest.CONTROL_AF_MODE,
                        CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_PICTURE
                    )

                    set(
                        CaptureRequest.CONTROL_AE_MODE,
                        CaptureRequest.CONTROL_AE_MODE_ON
                    )
                }

            session.setRepeatingRequest(
                request.build(),
                null,
                cameraHandler
            )

            statusText.text = "Point camera at text and press SCAN"
        } catch (t: Throwable) {
            statusText.text =
                "Preview error: ${t.message ?: t.javaClass.simpleName}"
        }
    }

    private fun captureText() {
        if (scanInProgress) return

        val camera = cameraDevice ?: run {
            statusText.text = "Camera not ready"
            return
        }

        val session = cameraSession ?: run {
            statusText.text = "Camera session not ready"
            return
        }

        val reader = imageReader ?: run {
            statusText.text = "Image reader not ready"
            return
        }

        scanInProgress = true
        scanButton.isEnabled = false
        statusText.text = "Capturing..."

        try {
            val request =
                camera.createCaptureRequest(
                    CameraDevice.TEMPLATE_STILL_CAPTURE
                ).apply {
                    addTarget(reader.surface)

                    set(
                        CaptureRequest.CONTROL_AF_MODE,
                        CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_PICTURE
                    )

                    set(
                        CaptureRequest.CONTROL_AE_MODE,
                        CaptureRequest.CONTROL_AE_MODE_ON
                    )

                    cameraCharacteristics?.get(
                        CameraCharacteristics.SENSOR_ORIENTATION
                    )?.let { sensorOrientation ->
                        set(
                            CaptureRequest.JPEG_ORIENTATION,
                            getJpegOrientation(sensorOrientation)
                        )
                    }
                }

            session.capture(
                request.build(),
                object : CameraCaptureSession.CaptureCallback() {
                },
                cameraHandler
            )

        } catch (t: Throwable) {
            scanInProgress = false
            scanButton.isEnabled = true

            statusText.text =
                "Capture failed: ${t.message ?: t.javaClass.simpleName}"
        }
    }

    private val imageAvailableListener =
        ImageReader.OnImageAvailableListener { reader ->

            val image = reader.acquireLatestImage()
                ?: return@OnImageAvailableListener

            try {
                val buffer = image.planes[0].buffer
                val bytes = ByteArray(buffer.remaining())

                buffer.get(bytes)

                val bitmap = decodeJpeg(bytes)

                if (bitmap == null) {
                    runOnUiThread {
                        scanInProgress = false
                        scanButton.isEnabled = true
                        statusText.text = "Could not decode camera image"
                    }
                    return@OnImageAvailableListener
                }

                appScope.launch {
                    try {
                        statusText.text = "Running OCR..."

                        val engine = ocr

                        if (engine == null) {
                            statusText.text = "OCR engine unavailable"
                            return@launch
                        }

                        val result = engine.recognize(bitmap)

                        if (result.results.isEmpty()) {
                            resultText.text = "No text detected."
                            statusText.text = "Detected 0 line(s) • ${result.totalTimeMs} ms"
                        } else {
                            resultText.text =
                                result.results.joinToString("\n") {
                                    it.text
                                }

                            statusText.text =
                                "Detected ${result.lineCount} line(s) • ${result.totalTimeMs} ms"
                        }

                    } catch (t: Throwable) {
                        resultText.text =
                            "OCR error:\n${t.message ?: t.javaClass.simpleName}"

                        statusText.text = "OCR failed"

                    } finally {
                        if (!bitmap.isRecycled) {
                            bitmap.recycle()
                        }

                        scanInProgress = false
                        scanButton.isEnabled = true
                    }
                }

            } catch (t: Throwable) {
                runOnUiThread {
                    scanInProgress = false
                    scanButton.isEnabled = true
                    statusText.text =
                        "Image processing failed: ${t.message ?: t.javaClass.simpleName}"
                }
            } finally {
                image.close()
            }
        }

    private fun decodeJpeg(bytes: ByteArray): Bitmap? {
        val bitmap =
            BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
                ?: return null

        return try {
            val exif =
                ExifInterface(ByteArrayInputStream(bytes))

            val orientation =
                exif.getAttributeInt(
                    ExifInterface.TAG_ORIENTATION,
                    ExifInterface.ORIENTATION_NORMAL
                )

            when (orientation) {
                ExifInterface.ORIENTATION_ROTATE_90 ->
                    rotateBitmap(bitmap, 90f)

                ExifInterface.ORIENTATION_ROTATE_180 ->
                    rotateBitmap(bitmap, 180f)

                ExifInterface.ORIENTATION_ROTATE_270 ->
                    rotateBitmap(bitmap, 270f)

                ExifInterface.ORIENTATION_FLIP_HORIZONTAL ->
                    flipBitmap(bitmap, horizontal = true)

                ExifInterface.ORIENTATION_FLIP_VERTICAL ->
                    flipBitmap(bitmap, horizontal = false)

                else -> bitmap
            }
        } catch (_: Throwable) {
            bitmap
        }
    }

    private fun rotateBitmap(
        bitmap: Bitmap,
        degrees: Float
    ): Bitmap {
        val matrix = Matrix().apply {
            postRotate(degrees)
        }

        val rotated = Bitmap.createBitmap(
            bitmap,
            0,
            0,
            bitmap.width,
            bitmap.height,
            matrix,
            true
        )

        if (rotated !== bitmap) {
            bitmap.recycle()
        }

        return rotated
    }

    private fun flipBitmap(
        bitmap: Bitmap,
        horizontal: Boolean
    ): Bitmap {
        val matrix = Matrix().apply {
            postScale(
                if (horizontal) -1f else 1f,
                if (horizontal) 1f else -1f
            )
        }

        val flipped = Bitmap.createBitmap(
            bitmap,
            0,
            0,
            bitmap.width,
            bitmap.height,
            matrix,
            true
        )

        if (flipped !== bitmap) {
            bitmap.recycle()
        }

        return flipped
    }

    private fun getJpegOrientation(sensorOrientation: Int): Int {
        val rotation = if (android.os.Build.VERSION.SDK_INT >= 30) {
            display?.rotation ?: Surface.ROTATION_0
        } else {
            @Suppress("DEPRECATION")
            windowManager.defaultDisplay.rotation
        }
        val deviceOrientation = ORIENTATIONS.get(rotation)
        val isFront = cameraCharacteristics?.get(CameraCharacteristics.LENS_FACING) == CameraCharacteristics.LENS_FACING_FRONT
        return if (isFront) {
            (sensorOrientation + deviceOrientation) % 360
        } else {
            (sensorOrientation - deviceOrientation + 360) % 360
        }
    }

    private fun configureTextureTransform(
        viewWidth: Int,
        viewHeight: Int
    ) {
        val size = previewSize ?: return

        if (viewWidth == 0 || viewHeight == 0) return

        val rotation = if (android.os.Build.VERSION.SDK_INT >= 30) {
            display?.rotation ?: Surface.ROTATION_0
        } else {
            @Suppress("DEPRECATION")
            windowManager.defaultDisplay.rotation
        }

        val matrix = Matrix()

        val viewRect = RectF(
            0f,
            0f,
            viewWidth.toFloat(),
            viewHeight.toFloat()
        )

        val bufferRect = RectF(
            0f,
            0f,
            size.height.toFloat(),
            size.width.toFloat()
        )

        val centerX = viewRect.centerX()
        val centerY = viewRect.centerY()

        bufferRect.offset(
            centerX - bufferRect.centerX(),
            centerY - bufferRect.centerY()
        )

        if (
            rotation == Surface.ROTATION_90 ||
            rotation == Surface.ROTATION_270
        ) {
            matrix.setRectToRect(
                viewRect,
                bufferRect,
                Matrix.ScaleToFit.FILL
            )

            val scale = max(
                viewHeight.toFloat() / size.width.toFloat(),
                viewWidth.toFloat() / size.height.toFloat()
            )

            matrix.postScale(
                scale,
                scale,
                centerX,
                centerY
            )

            matrix.postRotate(
                90f * (rotation - 2),
                centerX,
                centerY
            )

        } else if (rotation == Surface.ROTATION_180) {
            matrix.postRotate(
                180f,
                centerX,
                centerY
            )
        }

        textureView.setTransform(matrix)
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(
            requestCode,
            permissions,
            grantResults
        )

        if (requestCode == CAMERA_PERMISSION_REQUEST) {
            if (
                grantResults.isNotEmpty() &&
                grantResults[0] == PackageManager.PERMISSION_GRANTED
            ) {
                statusText.text = "Camera permission granted"

                if (textureView.isAvailable) {
                    openCamera(
                        textureView.width,
                        textureView.height
                    )
                }
            } else {
                statusText.text =
                    "Camera permission is required"
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()

        cameraSession?.close()
        cameraSession = null

        cameraDevice?.close()
        cameraDevice = null

        imageReader?.close()
        imageReader = null

        previewSurface?.release()
        previewSurface = null

        cameraThread?.quitSafely()
        cameraThread = null
        cameraHandler = null

        appScope.launch {
            try {
                ocr?.release()
            } catch (_: Throwable) {
            }
            ocr = null
        }

        appScope.cancel()
    }
}