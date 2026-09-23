package com.example.objectdetection.safety

import android.os.Handler
import android.os.Looper
import android.os.SystemClock
import android.util.Log
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.concurrent.CopyOnWriteArrayList

/**
 * Manages ingestion, packet parsing, validation, and connection monitoring for the VL53L5CX ToF sensor.
 *
 * Binary Protocol ("PITF"):
 * [0..3]   MAGIC: "PITF" (0x50, 0x49, 0x54, 0x46)
 * [4]      VERSION: 0x01
 * [5..12]  TIMESTAMP_MS: int64 (Big Endian)
 * [13..44] 16 x DISTANCES: uint16 (Big Endian) in mm
 * Total size: 45 bytes
 *
 * Text Protocol (Development / CLI):
 * "TOF:820,760,690,710,950,840,720,680,1200,1100,900,850,0,0,0,0\n"
 */
class TofSensorManager {

    companion object {
        private const val TAG = "TofSensorManager"
        const val BINARY_PACKET_SIZE = 45
        val MAGIC_BYTES = byteArrayOf('P'.code.toByte(), 'I'.code.toByte(), 'T'.code.toByte(), 'F'.code.toByte())
        private const val DISCONNECT_TIMEOUT_MS = 3000L
    }

    enum class SensorStatus {
        WAITING,
        CONNECTED,
        DISCONNECTED
    }

    interface TofListener {
        fun onTofFrameReceived(frame: TofFrame, updateRateHz: Double)
        fun onSensorStatusChanged(status: SensorStatus)
    }

    private val listeners = CopyOnWriteArrayList<TofListener>()
    private val mainHandler = Handler(Looper.getMainLooper())

    @Volatile
    var sensorStatus: SensorStatus = SensorStatus.WAITING
        private set

    @Volatile
    var lastPacketTimestampMs: Long = 0L
        private set

    @Volatile
    var updateRateHz: Double = 0.0
        private set

    private var packetCount = 0
    private var lastFpsCalculationTime = 0L

    private val watchdogRunnable = object : Runnable {
        override fun run() {
            val now = SystemClock.uptimeMillis()
            if (sensorStatus == SensorStatus.CONNECTED && (now - lastPacketTimestampMs > DISCONNECT_TIMEOUT_MS)) {
                Log.w(TAG, "VL53L5CX packet timeout (> ${DISCONNECT_TIMEOUT_MS}ms). Marking DISCONNECTED.")
                setSensorStatusInternal(SensorStatus.DISCONNECTED)
            }
            mainHandler.postDelayed(this, 1000L)
        }
    }

    init {
        mainHandler.postDelayed(watchdogRunnable, 1000L)
    }

    fun addListener(listener: TofListener) {
        listeners.add(listener)
    }

    fun removeListener(listener: TofListener) {
        listeners.remove(listener)
    }

    /**
     * Ingests and parses a raw binary packet received from the network.
     */
    fun parseBinaryPacket(data: ByteArray, offset: Int = 0, length: Int = data.size): Boolean {
        if (length < BINARY_PACKET_SIZE) return false

        // Check Magic "PITF"
        if (data[offset] != MAGIC_BYTES[0] ||
            data[offset + 1] != MAGIC_BYTES[1] ||
            data[offset + 2] != MAGIC_BYTES[2] ||
            data[offset + 3] != MAGIC_BYTES[3]
        ) {
            return false
        }

        val buffer = ByteBuffer.wrap(data, offset, BINARY_PACKET_SIZE).order(ByteOrder.BIG_ENDIAN)
        buffer.position(4) // Skip magic

        val version = buffer.get().toInt() and 0xFF
        if (version != 1) {
            Log.w(TAG, "Unsupported PITF version: $version")
            return false
        }

        val timestampMs = buffer.long
        val distances = IntArray(TofFrame.NUM_ZONES)

        for (i in 0 until TofFrame.NUM_ZONES) {
            val rawDist = buffer.short.toInt() and 0xFFFF
            distances[i] = sanitizeDistance(rawDist)
        }

        val frame = TofFrame(
            timestampMs = if (timestampMs > 0) timestampMs else System.currentTimeMillis(),
            distances = distances
        )

        handleValidFrame(frame)
        return true
    }

    /**
     * Ingests and parses a plain-text development string.
     */
    fun parseTextPacket(text: String): Boolean {
        var clean = text.trim()
        if (clean.startsWith("TOF:", ignoreCase = true)) {
            clean = clean.substring(4).trim()
        }

        val tokens = clean.split(",")
        if (tokens.size != TofFrame.NUM_ZONES) return false

        val distances = IntArray(TofFrame.NUM_ZONES)
        try {
            for (i in 0 until TofFrame.NUM_ZONES) {
                val raw = tokens[i].trim().toInt()
                distances[i] = sanitizeDistance(raw)
            }
        } catch (_: NumberFormatException) {
            return false
        }

        val frame = TofFrame(
            timestampMs = System.currentTimeMillis(),
            distances = distances
        )

        handleValidFrame(frame)
        return true
    }

    private fun sanitizeDistance(rawDistanceMm: Int): Int {
        return if (rawDistanceMm in TofFrame.MIN_VALID_DISTANCE_MM..TofFrame.MAX_VALID_DISTANCE_MM) {
            rawDistanceMm
        } else {
            0 // Invalid / out of range
        }
    }

    private fun handleValidFrame(frame: TofFrame) {
        lastPacketTimestampMs = SystemClock.uptimeMillis()

        if (sensorStatus != SensorStatus.CONNECTED) {
            setSensorStatusInternal(SensorStatus.CONNECTED)
        }

        calculateRate()

        for (listener in listeners) {
            try {
                listener.onTofFrameReceived(frame, updateRateHz)
            } catch (e: Exception) {
                Log.e(TAG, "Error notifying ToF frame listener", e)
            }
        }
    }

    private fun calculateRate() {
        packetCount++
        val now = SystemClock.uptimeMillis()
        if (lastFpsCalculationTime == 0L) {
            lastFpsCalculationTime = now
            return
        }

        val elapsed = now - lastFpsCalculationTime
        if (elapsed >= 1000L) {
            updateRateHz = (packetCount * 1000.0) / elapsed
            packetCount = 0
            lastFpsCalculationTime = now
        }
    }

    private fun setSensorStatusInternal(newStatus: SensorStatus) {
        if (sensorStatus == newStatus) return
        sensorStatus = newStatus
        Log.i(TAG, "VL53L5CX Sensor Status -> $newStatus")

        for (listener in listeners) {
            try {
                listener.onSensorStatusChanged(newStatus)
            } catch (e: Exception) {
                Log.e(TAG, "Error notifying sensor status listener", e)
            }
        }
    }

    fun release() {
        mainHandler.removeCallbacks(watchdogRunnable)
        listeners.clear()
    }
}
