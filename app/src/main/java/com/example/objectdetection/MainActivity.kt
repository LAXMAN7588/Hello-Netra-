package com.example.objectdetection

import android.Manifest
import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.content.ServiceConnection
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Color
import android.graphics.drawable.GradientDrawable
import android.os.Build
import android.os.Bundle
import android.os.IBinder
import android.util.Log
import android.view.Gravity
import android.view.LayoutInflater
import android.widget.Button
import android.widget.EditText
import android.widget.GridLayout
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import com.example.objectdetection.safety.ObstacleAnalyzer
import com.example.objectdetection.safety.TofFrame
import com.example.objectdetection.safety.TofSensorManager

class MainActivity : AppCompatActivity(), AppService.ServiceUiListener {

    companion object {
        private const val TAG = "MainActivity"
    }

    // UI Views
    private lateinit var frameView: ImageView
    private lateinit var detectionOverlay: DetectionOverlay
    private lateinit var connectionStatusText: TextView
    private lateinit var wifiModeText: TextView
    private lateinit var currentModeText: TextView
    private lateinit var serverAddressText: TextView
    private lateinit var frameStatsText: TextView
    private lateinit var fpsText: TextView
    private lateinit var inferenceText: TextView
    private lateinit var ramText: TextView
    private lateinit var countText: TextView
    private lateinit var breakdownText: TextView
    private lateinit var ocrStatusText: TextView
    private lateinit var sosStatusText: TextView
    private lateinit var ttsStatusText: TextView

    // ToF Safety Diagnostic Views
    private lateinit var tofStatusText: TextView
    private lateinit var tofTelemetryText: TextView
    private lateinit var tofGrid: GridLayout
    private lateinit var tofCells: Array<TextView>

    // Emergency Contacts Views
    private lateinit var contactsHeader: TextView
    private lateinit var btnAddContact: Button
    private lateinit var contactsContainer: LinearLayout

    // Background Service Binding
    private var appService: AppService? = null
    private var isBound = false

    private val serviceConnection = object : ServiceConnection {
        override fun onServiceConnected(name: ComponentName?, service: IBinder?) {
            val binder = service as? AppService.LocalBinder
            appService = binder?.service
            isBound = true
            appService?.addUiListener(this@MainActivity)

            appService?.contactRepository?.addListener(contactsChangeListener)
            refreshContactsUi()
            refreshServiceStatus()
            Log.i(TAG, "Connected and bound to AppService.")
        }

        override fun onServiceDisconnected(name: ComponentName?) {
            appService?.removeUiListener(this@MainActivity)
            appService?.contactRepository?.removeListener(contactsChangeListener)
            appService = null
            isBound = false
            Log.i(TAG, "Disconnected from AppService.")
        }
    }

    private val contactsChangeListener = object : EmergencyContactRepository.ContactChangeListener {
        override fun onContactsUpdated(contacts: List<EmergencyContact>) {
            runOnUiThread {
                refreshContactsUi()
            }
        }
    }

    // Permissions launcher
    private val permissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) { permissions ->
        val smsGranted = permissions[Manifest.permission.SEND_SMS] ?: false
        val fineLoc = permissions[Manifest.permission.ACCESS_FINE_LOCATION] ?: false
        Log.i(TAG, "Permission results: SMS=$smsGranted, Location=$fineLoc")
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        tofCells = Array(TofFrame.NUM_ZONES) {
            TextView(this)
        }

        setContentView(R.layout.activity_main)

        initViews()
        setupTofGrid()
        requestAppPermissions()
        startAndBindAppService()
    }

    private fun initViews() {
        frameView = findViewById(R.id.frameView)
        detectionOverlay = findViewById(R.id.detectionOverlay)
        connectionStatusText = findViewById(R.id.connectionStatusText)
        wifiModeText = findViewById(R.id.wifiModeText)
        currentModeText = findViewById(R.id.currentModeText)
        serverAddressText = findViewById(R.id.serverAddressText)
        frameStatsText = findViewById(R.id.frameStatsText)
        fpsText = findViewById(R.id.fpsText)
        inferenceText = findViewById(R.id.inferenceText)
        ramText = findViewById(R.id.ramText)
        countText = findViewById(R.id.countText)
        breakdownText = findViewById(R.id.breakdownText)
        ocrStatusText = findViewById(R.id.ocrStatusText)
        sosStatusText = findViewById(R.id.sosStatusText)
        ttsStatusText = findViewById(R.id.ttsStatusText)

        tofStatusText = findViewById(R.id.tofStatusText)
        tofTelemetryText = findViewById(R.id.tofTelemetryText)
        tofGrid = findViewById(R.id.tofGrid)

        contactsHeader = findViewById(R.id.contactsHeader)
        btnAddContact = findViewById(R.id.btnAddContact)
        contactsContainer = findViewById(R.id.contactsContainer)

        btnAddContact.setOnClickListener {
            showAddContactDialog()
        }
    }

    private fun setupTofGrid() {
        tofGrid.removeAllViews()

        for (i in 0 until TofFrame.NUM_ZONES) {
            val cell = TextView(this).apply {
                text = "--"
                textSize = 11f
                setTextColor(Color.WHITE)
                gravity = Gravity.CENTER
                setPadding(4, 8, 4, 8)

                val gd = GradientDrawable().apply {
                    setColor(Color.parseColor("#37474F"))
                    cornerRadius = 6f
                    setStroke(1, Color.parseColor("#546E7A"))
                }
                background = gd

                val params = GridLayout.LayoutParams().apply {
                    width = 0
                    height = GridLayout.LayoutParams.WRAP_CONTENT
                    columnSpec = GridLayout.spec(i % 4, 1f)
                    rowSpec = GridLayout.spec(i / 4)
                    setMargins(3, 3, 3, 3)
                }
                layoutParams = params
            }
            tofCells[i] = cell
            tofGrid.addView(cell)
        }
    }

    private fun requestAppPermissions() {
        val permissionsToRequest = mutableListOf(
            Manifest.permission.SEND_SMS,
            Manifest.permission.ACCESS_FINE_LOCATION,
            Manifest.permission.ACCESS_COARSE_LOCATION,
            Manifest.permission.CAMERA
        )

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            permissionsToRequest.add(Manifest.permission.POST_NOTIFICATIONS)
        }

        val missing = permissionsToRequest.filter {
            ContextCompat.checkSelfPermission(this, it) != PackageManager.PERMISSION_GRANTED
        }

        if (missing.isNotEmpty()) {
            permissionLauncher.launch(missing.toTypedArray())
        }
    }

    private fun startAndBindAppService() {
        val serviceIntent = Intent(this, AppService::class.java).apply {
            action = AppService.ACTION_START_SERVICE
        }

        ContextCompat.startForegroundService(this, serviceIntent)
        bindService(serviceIntent, serviceConnection, Context.BIND_AUTO_CREATE)
    }

    private fun refreshServiceStatus() {
        val service = appService ?: return
        val localIp = IpUtils.getLocalIpAddress()
        serverAddressText.text = "Frame Server: $localIp:5000"
        updateModeDisplay(service.modeController.currentMode)
        onTofStatusChanged(service.safetyManager.sensorManager.sensorStatus)
    }

    private fun updateModeDisplay(mode: AppMode) {
        when (mode) {
            AppMode.IDLE -> {
                currentModeText.text = "Current Mode: IDLE (Standby)"
                currentModeText.setTextColor(Color.parseColor("#76FF03"))
            }
            AppMode.OBJECT_RECOGNITION -> {
                currentModeText.text = "Current Mode: OBJECT RECOGNITION (120s Session)"
                currentModeText.setTextColor(Color.parseColor("#00E5FF"))
            }
            AppMode.OCR -> {
                currentModeText.text = "Current Mode: OCR Text Reading (120s Session)"
                currentModeText.setTextColor(Color.parseColor("#FFD600"))
            }
            AppMode.SOS -> {
                currentModeText.text = "Current Mode: SOS (Emergency Location Dispatch)"
                currentModeText.setTextColor(Color.parseColor("#FF1744"))
            }
        }
    }

    // =========================================================================
    // Emergency Contact Management UI
    // =========================================================================

    private fun refreshContactsUi() {
        val repo = appService?.contactRepository ?: return
        val contacts = repo.getContacts()

        contactsHeader.text = "Emergency Contacts (${contacts.size})"
        sosStatusText.text = "SOS: Ready (${contacts.size} Contacts)"

        contactsContainer.removeAllViews()

        if (contacts.isEmpty()) {
            val emptyTv = TextView(this).apply {
                text = "No emergency contacts configured yet.\nTap '+ Add Contact' to set up contacts for SOS."
                setTextColor(Color.parseColor("#9E9E9E"))
                textSize = 12f
                setPadding(0, 8, 0, 8)
            }
            contactsContainer.addView(emptyTv)
            return
        }

        for (contact in contacts) {
            val row = LinearLayout(this).apply {
                orientation = LinearLayout.HORIZONTAL
                gravity = Gravity.CENTER_VERTICAL
                setPadding(0, 6, 0, 6)
            }

            val infoTv = TextView(this).apply {
                layoutParams = LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f)
                text = "${contact.name}  (${contact.phoneNumber})"
                setTextColor(Color.WHITE)
                textSize = 13f
            }

            val delBtn = Button(this).apply {
                text = "Delete"
                textSize = 11f
                backgroundTintList = android.content.res.ColorStateList.valueOf(Color.parseColor("#D32F2F"))
                setTextColor(Color.WHITE)
                setOnClickListener {
                    repo.deleteContact(contact.id)
                    Toast.makeText(this@MainActivity, "Deleted ${contact.name}", Toast.LENGTH_SHORT).show()
                }
            }

            row.addView(infoTv)
            row.addView(delBtn)
            contactsContainer.addView(row)
        }
    }

    private fun showAddContactDialog() {
        val container = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(40, 20, 40, 10)
        }

        val nameEt = EditText(this).apply {
            hint = "Contact Name (e.g., Mom, John)"
            setTextColor(Color.BLACK)
        }

        val phoneEt = EditText(this).apply {
            hint = "Phone Number (e.g., +1234567890)"
            inputType = android.text.InputType.TYPE_CLASS_PHONE
            setTextColor(Color.BLACK)
        }

        container.addView(nameEt)
        container.addView(phoneEt)

        AlertDialog.Builder(this)
            .setTitle("Add Emergency Contact")
            .setView(container)
            .setPositiveButton("Save") { _, _ ->
                val name = nameEt.text.toString().trim()
                val phone = phoneEt.text.toString().trim()

                if (name.isNotEmpty() && phone.isNotEmpty()) {
                    appService?.contactRepository?.addContact(name, phone)
                    Toast.makeText(this, "Saved $name for Emergency SOS", Toast.LENGTH_SHORT).show()
                } else {
                    Toast.makeText(this, "Please enter both name and phone number", Toast.LENGTH_SHORT).show()
                }
            }
            .setNegativeButton("Cancel", null)
            .show()
    }

    // =========================================================================
    // AppService.ServiceUiListener Callbacks
    // =========================================================================

    override fun onFrameProcessed(
        bitmap: Bitmap,
        detections: List<Detection>,
        inferenceMs: Long,
        ramMb: Long
    ) {
        val countsSummary = buildCountsSummary(detections)
        val buffer = appService?.frameBuffer
        val recv = buffer?.framesReceived ?: 0
        val proc = buffer?.framesProcessed ?: 0
        val drop = buffer?.framesDropped ?: 0
        val fps = appService?.currentFps ?: 0.0

        runOnUiThread {
            frameView.setImageBitmap(bitmap)
            detectionOverlay.setDetections(detections, bitmap.width, bitmap.height)

            frameStatsText.text = "Frames: Recv: $recv | Proc: $proc | Drop: $drop"
            fpsText.text = "FPS: ${"%.1f".format(fps)}"
            inferenceText.text = "Inference: ${inferenceMs}ms"
            ramText.text = "RAM: ${ramMb}MB"
            countText.text = "Objects (>=60%): ${detections.size}"
            breakdownText.text = "Counts: $countsSummary"
        }
    }

    override fun onOcrProcessed(
        bitmap: Bitmap,
        text: String,
        confidence: Float,
        inferenceMs: Long,
        ramMb: Long
    ) {
        val buffer = appService?.frameBuffer
        val recv = buffer?.framesReceived ?: 0
        val proc = buffer?.framesProcessed ?: 0
        val drop = buffer?.framesDropped ?: 0
        val fps = appService?.currentFps ?: 0.0

        runOnUiThread {
            frameView.setImageBitmap(bitmap)
            detectionOverlay.clearDetections()

            frameStatsText.text = "Frames: Recv: $recv | Proc: $proc | Drop: $drop"
            fpsText.text = "FPS: ${"%.1f".format(fps)}"
            inferenceText.text = "Inference: ${inferenceMs}ms"
            ramText.text = "RAM: ${ramMb}MB"

            val displayText = if (text.isNotBlank()) "\"$text\" (${(confidence * 100).toInt()}%)" else "No text detected"
            ocrStatusText.text = "OCR: $displayText"
        }
    }

    override fun onConnectionStateChanged(state: LocalFrameServer.ConnectionState) {
        runOnUiThread {
            when (state) {
                is LocalFrameServer.ConnectionState.Listening -> {
                    connectionStatusText.text = "Status: WAITING FOR PI"
                    connectionStatusText.setTextColor(Color.parseColor("#FFD600"))
                    serverAddressText.text = "Frame Server: ${state.localIp}:${state.port}"
                }
                is LocalFrameServer.ConnectionState.Connected -> {
                    connectionStatusText.text = "Connection: CONNECTED"
                    connectionStatusText.setTextColor(Color.parseColor("#00E676"))
                }
                is LocalFrameServer.ConnectionState.Disconnected -> {
                    connectionStatusText.text = "Connection: DISCONNECTED"
                    connectionStatusText.setTextColor(Color.parseColor("#FF5252"))
                    detectionOverlay.clearDetections()
                }
                is LocalFrameServer.ConnectionState.Error -> {
                    connectionStatusText.text = "Error: ${state.message}"
                    connectionStatusText.setTextColor(Color.parseColor("#FF5252"))
                }
                is LocalFrameServer.ConnectionState.Stopped -> {
                    connectionStatusText.text = "Connection: STOPPED"
                    connectionStatusText.setTextColor(Color.GRAY)
                }
            }
        }
    }

    override fun onModeChanged(mode: AppMode) {
        runOnUiThread {
            updateModeDisplay(mode)
            if (mode != AppMode.OBJECT_RECOGNITION) {
                detectionOverlay.clearDetections()
            }
        }
    }

    override fun onSosStatusChanged(status: SosManager.SosStatus) {
        runOnUiThread {
            sosStatusText.text = "SOS: ${status.message}"
            sosStatusText.setTextColor(if (status.success) Color.parseColor("#00E676") else Color.parseColor("#FF1744"))
        }
    }

    override fun onTtsStatus(message: String) {
        runOnUiThread {
            ttsStatusText.text = message
        }
    }

    override fun onTofTelemetryUpdated(
        frame: TofFrame,
        analysis: ObstacleAnalyzer.SpatialAnalysis,
        updateRateHz: Double
    ) {
        val validCount = analysis.totalValidZones
        val minMm = if (analysis.overallMinDistanceMm > 0) "${analysis.overallMinDistanceMm}mm" else "--"

        runOnUiThread {
            tofTelemetryText.text = "Rate: ${"%.1f".format(updateRateHz)} Hz | Min: $minMm | Valid: $validCount/16"

            // Update 4x4 matrix display
            for (i in 0 until TofFrame.NUM_ZONES) {
                val d = frame.distances[i]
                val cell = tofCells[i]
                if (d > 0) {
                    cell.text = "$d"
                    (cell.background as? GradientDrawable)?.setColor(Color.parseColor("#1B5E20")) // Dark Green for valid
                } else {
                    cell.text = "0"
                    (cell.background as? GradientDrawable)?.setColor(Color.parseColor("#37474F")) // Dark Grey for invalid/no target
                }
            }
        }
    }

    override fun onTofStatusChanged(status: TofSensorManager.SensorStatus) {
        runOnUiThread {
            when (status) {
                TofSensorManager.SensorStatus.CONNECTED -> {
                    tofStatusText.text = "Status: CONNECTED"
                    tofStatusText.setTextColor(Color.parseColor("#00E676"))
                }
                TofSensorManager.SensorStatus.DISCONNECTED -> {
                    tofStatusText.text = "Status: DISCONNECTED"
                    tofStatusText.setTextColor(Color.parseColor("#FF5252"))
                }
                TofSensorManager.SensorStatus.WAITING -> {
                    tofStatusText.text = "Status: WAITING"
                    tofStatusText.setTextColor(Color.parseColor("#FFD600"))
                }
            }
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

    override fun onDestroy() {
        super.onDestroy()
        if (isBound) {
            appService?.removeUiListener(this)
            appService?.contactRepository?.removeListener(contactsChangeListener)
            unbindService(serviceConnection)
            isBound = false
        }
    }
}


