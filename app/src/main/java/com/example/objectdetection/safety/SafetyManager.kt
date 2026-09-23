package com.example.objectdetection.safety

import android.util.Log
import java.util.concurrent.CopyOnWriteArrayList

/**
 * Top-level manager for the Hello-Netra Safety Subsystem.
 *
 * Core Principles:
 * - Operates independently of the camera session lifecycle (remains active across IDLE, OBJECT, OCR, SOS).
 * - Maintains the latest valid 4x4 ToF depth frame.
 * - Does NOT continuously run expensive obstacle-analysis loops or unneeded TTS on every packet.
 * - Exposes real-time sensor metrics and spatial telemetry to UI and safety policy handlers.
 */
class SafetyManager {

    companion object {
        private const val TAG = "SafetyManager"
    }

    val sensorManager = TofSensorManager()
    val obstacleAnalyzer = ObstacleAnalyzer()

    @Volatile
    var latestFrame: TofFrame? = null
        private set

    interface SafetyObserver {
        fun onTofTelemetryUpdated(frame: TofFrame, analysis: ObstacleAnalyzer.SpatialAnalysis, updateRateHz: Double)
        fun onSafetyStatusChanged(status: TofSensorManager.SensorStatus)
    }

    private val observers = CopyOnWriteArrayList<SafetyObserver>()

    init {
        sensorManager.addListener(object : TofSensorManager.TofListener {
            override fun onTofFrameReceived(frame: TofFrame, updateRateHz: Double) {
                latestFrame = frame

                // Perform lightweight spatial sector aggregation
                val analysis = obstacleAnalyzer.analyze(frame)

                for (observer in observers) {
                    try {
                        observer.onTofTelemetryUpdated(frame, analysis, updateRateHz)
                    } catch (e: Exception) {
                        Log.e(TAG, "Error in SafetyObserver telemetry update", e)
                    }
                }
            }

            override fun onSensorStatusChanged(status: TofSensorManager.SensorStatus) {
                for (observer in observers) {
                    try {
                        observer.onSafetyStatusChanged(status)
                    } catch (e: Exception) {
                        Log.e(TAG, "Error in SafetyObserver status change", e)
                    }
                }
            }
        })
    }

    fun addObserver(observer: SafetyObserver) {
        observers.add(observer)
    }

    fun removeObserver(observer: SafetyObserver) {
        observers.remove(observer)
    }

    fun ingestBinaryData(data: ByteArray, offset: Int = 0, length: Int = data.size): Boolean {
        return sensorManager.parseBinaryPacket(data, offset, length)
    }

    fun ingestTextData(text: String): Boolean {
        return sensorManager.parseTextPacket(text)
    }

    fun release() {
        observers.clear()
        sensorManager.release()
    }
}
