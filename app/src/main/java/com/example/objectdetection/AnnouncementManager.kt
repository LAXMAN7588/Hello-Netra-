package com.example.objectdetection

import android.os.Handler
import android.os.Looper
import android.util.Log

/**
 * Intelligent debounce and announcement manager for object detections.
 *
 * Rules:
 * - Announce detected objects using natural English phrases (e.g., "One person, one car.").
 * - Debounce rapidly fluctuating detections (requires counts to remain stable before speaking).
 * - Only announce when detection counts CHANGE from the last spoken state.
 * - Remain silent when counts remain unchanged.
 * - Remain silent when no objects are detected.
 * - Prevent overlapping audio queues.
 */
class AnnouncementManager(
    private val tts: PiperTTS,
    private val debounceDelayMs: Long = 1000L,
    private val cooldownMs: Long = 3000L
) {
    companion object {
        private const val TAG = "AnnouncementManager"

        private val NUMBER_WORDS = arrayOf(
            "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
            "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen",
            "eighteen", "nineteen", "twenty"
        )
    }

    private val mainHandler = Handler(Looper.getMainLooper())

    private var lastSpokenCounts: Map<String, Int> = emptyMap()
    private var pendingCounts: Map<String, Int> = emptyMap()
    private var lastSpokenTimestamp: Long = 0L

    private val debounceRunnable = Runnable {
        checkAndAnnounce()
    }

    var onAnnouncementStateChanged: ((String) -> Unit)? = null

    /**
     * Called on each frame with current detected objects.
     */
    fun onDetections(detections: List<Detection>) {
        if (detections.isEmpty()) {
            // Cancel any pending announcement when view is clear
            mainHandler.removeCallbacks(debounceRunnable)
            pendingCounts = emptyMap()
            return
        }

        // Aggregate counts by class label
        val currentCounts = mutableMapOf<String, Int>()
        for (detection in detections) {
            currentCounts[detection.label] = (currentCounts[detection.label] ?: 0) + 1
        }

        // If counts are the same as last spoken, do not trigger speech
        if (currentCounts == lastSpokenCounts) {
            mainHandler.removeCallbacks(debounceRunnable)
            return
        }

        // If pending counts changed, reset the debounce timer
        if (currentCounts != pendingCounts) {
            pendingCounts = currentCounts
            mainHandler.removeCallbacks(debounceRunnable)
            mainHandler.postDelayed(debounceRunnable, debounceDelayMs)
        }
    }

    private fun checkAndAnnounce() {
        val now = System.currentTimeMillis()
        if (now - lastSpokenTimestamp < cooldownMs) {
            // Re-schedule after cooldown expires
            val remainingCooldown = cooldownMs - (now - lastSpokenTimestamp)
            mainHandler.postDelayed(debounceRunnable, remainingCooldown)
            return
        }

        val countsToAnnounce = pendingCounts
        if (countsToAnnounce.isEmpty() || countsToAnnounce == lastSpokenCounts) {
            return
        }

        val phrase = buildAnnouncementPhrase(countsToAnnounce)
        if (phrase.isNotEmpty()) {
            lastSpokenCounts = countsToAnnounce
            lastSpokenTimestamp = now

            Log.i(TAG, "Speaking announcement: \"$phrase\"")
            onAnnouncementStateChanged?.invoke(phrase)

            tts.speak(phrase) {
                // Completed speaking
            }
        }
    }

    /**
     * Formats object counts into natural English:
     * - 1 person -> "One person."
     * - 1 person, 1 car -> "One person, one car."
     * - 2 persons, 1 car -> "Two persons, one car."
     */
    fun buildAnnouncementPhrase(counts: Map<String, Int>): String {
        if (counts.isEmpty()) return ""

        val parts = mutableListOf<String>()

        for ((label, count) in counts) {
            val countWord = if (count in NUMBER_WORDS.indices) NUMBER_WORDS[count] else count.toString()
            val formattedLabel = formatClassLabel(label, count)
            parts.add("$countWord $formattedLabel")
        }

        return parts.joinToString(", ") + "."
    }

    private fun formatClassLabel(label: String, count: Int): String {
        val isPlural = count > 1

        return when (label.lowercase()) {
            "accident" -> if (isPlural) "accidents" else "accident"
            "auto" -> if (isPlural) "autos" else "auto"
            "bicycle" -> if (isPlural) "bicycles" else "bicycle"
            "bus" -> if (isPlural) "buses" else "bus"
            "car" -> if (isPlural) "cars" else "car"
            "chair" -> if (isPlural) "chairs" else "chair"
            "emergency-responder" -> if (isPlural) "emergency responders" else "emergency responder"
            "emergency-vehicle" -> if (isPlural) "emergency vehicles" else "emergency vehicle"
            "laptop" -> if (isPlural) "laptops" else "laptop"
            "person" -> if (isPlural) "persons" else "person"
            "truck" -> if (isPlural) "trucks" else "truck"
            "two wheeler" -> if (isPlural) "two wheelers" else "two wheeler"
            "wrecked-vehicle" -> if (isPlural) "wrecked vehicles" else "wrecked vehicle"
            else -> if (isPlural) "${label}s" else label
        }
    }

    fun reset() {
        mainHandler.removeCallbacks(debounceRunnable)
        lastSpokenCounts = emptyMap()
        pendingCounts = emptyMap()
    }
}
