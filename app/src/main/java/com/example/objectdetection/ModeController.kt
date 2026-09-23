package com.example.objectdetection

import android.util.Log
import java.util.concurrent.CopyOnWriteArrayList

/**
 * Single source of truth for the system's active operating mode.
 * Coordinates transitions between IDLE, OBJECT_RECOGNITION, OCR, and SOS.
 */
class ModeController {

    companion object {
        private const val TAG = "ModeController"
    }

    interface ModeChangeListener {
        fun onModeChanged(previousMode: AppMode, newMode: AppMode)
    }

    @Volatile
    var currentMode: AppMode = AppMode.IDLE
        private set

    private val listeners = CopyOnWriteArrayList<ModeChangeListener>()

    fun addListener(listener: ModeChangeListener) {
        listeners.add(listener)
    }

    fun removeListener(listener: ModeChangeListener) {
        listeners.remove(listener)
    }

    @Synchronized
    fun setMode(newMode: AppMode) {
        val prev = currentMode
        if (prev == newMode) {
            Log.d(TAG, "Mode is already $newMode, ignoring duplicate switch.")
            return
        }

        Log.i(TAG, ">>> MODE TRANSITION: $prev -> $newMode <<<")
        currentMode = newMode

        for (listener in listeners) {
            try {
                listener.onModeChanged(prev, newMode)
            } catch (e: Exception) {
                Log.e(TAG, "Error notifying mode change listener", e)
            }
        }
    }
}
