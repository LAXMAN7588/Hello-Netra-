package com.example.objectdetection

import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.Service
import android.content.Context
import android.content.Intent
import android.os.Build
import android.os.IBinder
import android.os.PowerManager
import android.util.Log
import androidx.core.app.NotificationCompat
import kotlinx.coroutines.*
import org.json.JSONObject
import java.io.BufferedReader
import java.io.InputStreamReader
import java.io.OutputStream
import java.net.DatagramPacket
import java.net.DatagramSocket
import java.net.ServerSocket
import java.net.Socket
import java.util.Locale

/**
 * Foreground Service for physical Pi SOS button and background triggers.
 * Handles locked screen, backgrounded app, and screen off scenarios.
 */
class CaneBackgroundService : Service() {

    private val scope = CoroutineScope(Dispatchers.IO + SupervisorJob())
    private var serverSocket: ServerSocket? = null
    private var udpSocket: DatagramSocket? = null
    private var wakeLock: PowerManager.WakeLock? = null
    private var sosManager: SosManager? = null

    companion object {
        private const val TAG = "CaneBackgroundService"
        private const val HTTP_PORT = 8080
        private const val UDP_DISCOVERY_PORT = 8888
        private const val UDP_SIGNAL_PORT = 5001
        private const val CHANNEL_ID = "netra_cane_channel"
        private const val NOTIFICATION_ID = 1001
    }

    override fun onCreate() {
        super.onCreate()
        Log.i(TAG, "NETRA_SERVICE: Starting background Cane & SOS service...")
        acquireWakeLock()
        startForegroundNotification()

        sosManager = SosManager(applicationContext) { status ->
            Log.i(TAG, "NETRA_SOS: Background status: $status")
        }

        startUdpDiscoveryResponder()
        startUdpSignalListener()
        startHttpServer()
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        return START_STICKY
    }

    private fun acquireWakeLock() {
        try {
            val pm = getSystemService(Context.POWER_SERVICE) as PowerManager
            wakeLock = pm.newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "Netra::CaneWakeLock")
            wakeLock?.acquire(24 * 60 * 60 * 1000L) // 24 hours
        } catch (e: Exception) {
            Log.e(TAG, "Failed to acquire WakeLock", e)
        }
    }

    private fun startForegroundNotification() {
        val channelName = "Netra Emergency & Cane Service"
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val chan = NotificationChannel(
                CHANNEL_ID,
                channelName,
                NotificationManager.IMPORTANCE_LOW
            )
            val manager = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
            manager.createNotificationChannel(chan)
        }

        val notification = NotificationCompat.Builder(this, CHANNEL_ID)
            .setContentTitle("Netra Safety System Active")
            .setContentText("Monitoring physical SOS button & background signals...")
            .setSmallIcon(android.R.drawable.stat_notify_sync)
            .setOngoing(true)
            .build()

        startForeground(NOTIFICATION_ID, notification)
    }

    // UDP Discovery: responds to Pi discovery ping
    private fun startUdpDiscoveryResponder() {
        scope.launch {
            try {
                udpSocket = DatagramSocket(UDP_DISCOVERY_PORT)
                val buffer = ByteArray(1024)
                while (isActive) {
                    val packet = DatagramPacket(buffer, buffer.size)
                    udpSocket?.receive(packet)
                    val msg = String(packet.data, 0, packet.length).trim()
                    if (msg == "NETRA_DISCOVERY_PING") {
                        val response = "NETRA_PHONE_ACK".toByteArray()
                        val replyPacket = DatagramPacket(response, response.size, packet.address, packet.port)
                        udpSocket?.send(replyPacket)
                    }
                }
            } catch (e: Exception) {
                Log.d(TAG, "UDP Discovery loop ended: ${e.message}")
            }
        }
    }

    // Direct UDP Signal Listener (Port 5001 for low latency physical button signals: OBJECT, OCR, SOS)
    private fun startUdpSignalListener() {
        scope.launch {
            try {
                val signalSocket = DatagramSocket(UDP_SIGNAL_PORT)
                val buffer = ByteArray(256)
                while (isActive) {
                    val packet = DatagramPacket(buffer, buffer.size)
                    signalSocket.receive(packet)
                    val signal = String(packet.data, 0, packet.length).trim()
                    handleSignalCommand(signal)
                }
            } catch (e: Exception) {
                Log.d(TAG, "UDP Signal listener ended: ${e.message}")
            }
        }
    }

    // HTTP Server Listener (Port 8080)
    private fun startHttpServer() {
        scope.launch {
            try {
                serverSocket = ServerSocket(HTTP_PORT)
                while (isActive) {
                    val socket = serverSocket?.accept() ?: break
                    launch { handleClient(socket) }
                }
            } catch (e: Exception) {
                Log.d(TAG, "HTTP server ended: ${e.message}")
            }
        }
    }

    private fun handleClient(socket: Socket) {
        var command = "UNKNOWN"

        try {
            val reader = BufferedReader(InputStreamReader(socket.getInputStream()))
            val output: OutputStream = socket.getOutputStream()

            var line: String?
            var contentLength = 0
            while (reader.readLine().also { line = it } != null) {
                if (line.isNull_or_empty()) break
                if (line!!.lowercase(Locale.ROOT).startsWith("content-length:")) {
                    contentLength = line!!.substring(15).trim().toIntOrNull() ?: 0
                }
            }

            if (contentLength > 0) {
                val body = CharArray(contentLength)
                reader.read(body, 0, contentLength)
                val json = JSONObject(String(body))
                command = json.optString("command", "UNKNOWN")
            }

            val response = "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 20\r\n\r\n{\"status\":\"received\"}"
            output.write(response.toByteArray(Charsets.UTF_8))
            output.flush()

            handleSignalCommand(command)

        } catch (e: Exception) {
            Log.e(TAG, "Error handling HTTP client", e)
        } finally {
            try { socket.close() } catch (_: Exception) {}
        }
    }

    private fun String?.isNull_or_empty(): Boolean = this == null || this.isEmpty()

    private fun handleSignalCommand(command: String) {
        val cleanCmd = command.trim().uppercase(Locale.ROOT)
        Log.i("NETRA_SIGNAL", "received $cleanCmd")

        val intent = Intent("com.example.objectdetection.PI_SIGNAL").apply {
            setPackage(packageName)
            putExtra("command", cleanCmd)
        }
        sendBroadcast(intent)

        if (cleanCmd == "SOS") {
            Log.i("NETRA_SOS", "Triggering background SOS workflow...")
            sosManager?.startSosFromBackground()
        }
    }

    override fun onDestroy() {
        scope.cancel()
        try { serverSocket?.close() } catch (_: Exception) {}
        try { udpSocket?.close() } catch (_: Exception) {}
        wakeLock?.let { if (it.isHeld) it.release() }
        super.onDestroy()
    }

    override fun onBind(intent: Intent?): IBinder? = null
}
