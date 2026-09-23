package com.example.objectdetection

import android.util.Log

/**
 * Protocol decoder for Raspberry Pi control commands sent over TCP.
 *
 * Supported Command Formats:
 * - "MODE_OBJECT" or "OBJECT" -> AppMode.OBJECT_RECOGNITION
 * - "MODE_OCR" or "OCR" -> AppMode.OCR
 * - "MODE_STOP" or "STOP" or "IDLE" -> AppMode.IDLE
 * - "MODE_SOS" or "SOS" -> AppMode.SOS
 */
object PiControlProtocol {

    private const val TAG = "PiControlProtocol"

    fun parseCommand(rawString: String): AppMode? {
        val trimmed = rawString.trim().uppercase()

        return when {
            trimmed.contains("MODE_OBJECT") || trimmed == "OBJECT" -> AppMode.OBJECT_RECOGNITION
            trimmed.contains("MODE_OCR") || trimmed == "OCR" -> AppMode.OCR
            trimmed.contains("MODE_STOP") || trimmed == "STOP" || trimmed.contains("MODE_IDLE") || trimmed == "IDLE" -> AppMode.IDLE
            trimmed.contains("MODE_SOS") || trimmed == "SOS" -> AppMode.SOS
            else -> {
                Log.d(TAG, "Unrecognized control string: '$rawString'")
                null
            }
        }
    }
}
