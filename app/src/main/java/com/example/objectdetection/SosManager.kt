package com.example.objectdetection

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.location.Geocoder
import android.location.Location
import android.location.LocationListener
import android.location.LocationManager
import android.os.Build
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.telephony.SmsManager
import android.util.Log
import androidx.core.content.ContextCompat
import java.util.Locale
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

/**
 * Manages the emergency SOS workflow.
 *
 * Requirements:
 * - Operates without internet (GPS satellites provide raw coordinates).
 * - Queries GPS and Network location providers.
 * - Formats coordinate message with optional offline reverse geocoding and Google Maps link.
 * - Sends direct SMS using Android SmsManager to ALL saved emergency contacts.
 * - No user confirmation screen, no composer, no manual Send button.
 */
class SosManager(
    private val context: Context,
    private val contactRepository: EmergencyContactRepository,
    private val tts: PiperTTS? = null
) {
    companion object {
        private const val TAG = "SosManager"
        private const val LOCATION_TIMEOUT_MS = 8000L
    }

    private val locationManager = context.getSystemService(Context.LOCATION_SERVICE) as? LocationManager
    private val executor: ExecutorService = Executors.newSingleThreadExecutor()
    private val mainHandler = Handler(Looper.getMainLooper())

    data class SosStatus(
        val success: Boolean,
        val message: String,
        val contactCount: Int,
        val locationSummary: String
    )

    var onStatusChanged: ((SosStatus) -> Unit)? = null

    /**
     * Executes the complete automatic SOS trigger workflow.
     */
    fun triggerSos() {
        Log.i(TAG, ">>> EMERGENCY SOS TRIGGERED BY HARDWARE <<<")

        executor.execute {
            val contacts = contactRepository.getContacts()
            if (contacts.isEmpty()) {
                val errorMsg = "SOS FAILED: No emergency contacts saved in app."
                Log.e(TAG, errorMsg)
                postStatus(SosStatus(false, errorMsg, 0, "No contacts"))
                tts?.speak("Emergency SOS failed. No emergency contacts are configured.")
                return@execute
            }

            // Check SMS permission
            val hasSmsPermission = ContextCompat.checkSelfPermission(
                context,
                Manifest.permission.SEND_SMS
            ) == PackageManager.PERMISSION_GRANTED

            if (!hasSmsPermission) {
                val errorMsg = "SOS FAILED: SMS permission not granted on phone."
                Log.e(TAG, errorMsg)
                postStatus(SosStatus(false, errorMsg, contacts.size, "SMS permission missing"))
                tts?.speak("Emergency SOS failed. SMS permission not granted.")
                return@execute
            }

            // Acquire location
            val location = acquireBestLocation()
            val locationStr = formatLocationDetails(location)

            // Compose emergency message
            val smsBody = buildEmergencyMessage(location)
            Log.i(TAG, "Emergency SMS Content:\n$smsBody")

            // Send SMS to all saved contacts
            var sentCount = 0
            val smsManager = getSmsManager()

            for (contact in contacts) {
                try {
                    val parts = smsManager.divideMessage(smsBody)
                    if (parts.size > 1) {
                        smsManager.sendMultipartTextMessage(
                            contact.phoneNumber,
                            null,
                            parts,
                            null,
                            null
                        )
                    } else {
                        smsManager.sendTextMessage(
                            contact.phoneNumber,
                            null,
                            smsBody,
                            null,
                            null
                        )
                    }
                    sentCount++
                    Log.i(TAG, "Sent emergency SMS to: ${contact.name} (${contact.phoneNumber})")
                } catch (e: Exception) {
                    Log.e(TAG, "Failed to send SMS to ${contact.name} (${contact.phoneNumber})", e)
                }
            }

            val success = sentCount > 0
            val statusMsg = if (success) {
                "SOS SUCCESS: Emergency SMS sent to $sentCount of ${contacts.size} contacts."
            } else {
                "SOS FAILED: Unable to dispatch SMS."
            }

            postStatus(SosStatus(success, statusMsg, contacts.size, locationStr))

            // Piper TTS Announcement
            if (success) {
                tts?.speak("Emergency SOS sent to $sentCount saved contacts.")
            } else {
                tts?.speak("Emergency SOS failed to send message.")
            }
        }
    }

    private fun acquireBestLocation(): Location? {
        val lm = locationManager ?: return null

        val hasFineLoc = ContextCompat.checkSelfPermission(
            context,
            Manifest.permission.ACCESS_FINE_LOCATION
        ) == PackageManager.PERMISSION_GRANTED
        val hasCoarseLoc = ContextCompat.checkSelfPermission(
            context,
            Manifest.permission.ACCESS_COARSE_LOCATION
        ) == PackageManager.PERMISSION_GRANTED

        if (!hasFineLoc && !hasCoarseLoc) {
            Log.w(TAG, "Location permissions not granted.")
            return null
        }

        var bestLocation: Location? = null

        // 1. Check last known locations
        try {
            if (lm.isProviderEnabled(LocationManager.GPS_PROVIDER)) {
                val loc = lm.getLastKnownLocation(LocationManager.GPS_PROVIDER)
                if (loc != null && isBetterLocation(loc, bestLocation)) {
                    bestLocation = loc
                }
            }
            if (lm.isProviderEnabled(LocationManager.NETWORK_PROVIDER)) {
                val loc = lm.getLastKnownLocation(LocationManager.NETWORK_PROVIDER)
                if (loc != null && isBetterLocation(loc, bestLocation)) {
                    bestLocation = loc
                }
            }
            if (lm.isProviderEnabled(LocationManager.PASSIVE_PROVIDER)) {
                val loc = lm.getLastKnownLocation(LocationManager.PASSIVE_PROVIDER)
                if (loc != null && isBetterLocation(loc, bestLocation)) {
                    bestLocation = loc
                }
            }
        } catch (e: Exception) {
            Log.w(TAG, "Error querying last known location", e)
        }

        // 2. Request a single fresh update if available
        val freshLocation = requestFreshLocation(lm)
        if (freshLocation != null && isBetterLocation(freshLocation, bestLocation)) {
            bestLocation = freshLocation
        }

        return bestLocation
    }

    private fun requestFreshLocation(lm: LocationManager): Location? {
        val resultHolder = java.util.concurrent.atomic.AtomicReference<Location?>(null)
        val latch = java.util.concurrent.CountDownLatch(1)

        val listener = object : LocationListener {
            override fun onLocationChanged(location: Location) {
                resultHolder.set(location)
                latch.countDown()
                try {
                    lm.removeUpdates(this)
                } catch (_: Exception) {}
            }
            override fun onProviderEnabled(provider: String) {}
            override fun onProviderDisabled(provider: String) {}
            @Deprecated("Deprecated in Java")
            override fun onStatusChanged(provider: String?, status: Int, extras: Bundle?) {}
        }

        mainHandler.post {
            try {
                if (lm.isProviderEnabled(LocationManager.GPS_PROVIDER)) {
                    lm.requestSingleUpdate(LocationManager.GPS_PROVIDER, listener, Looper.getMainLooper())
                } else if (lm.isProviderEnabled(LocationManager.NETWORK_PROVIDER)) {
                    lm.requestSingleUpdate(LocationManager.NETWORK_PROVIDER, listener, Looper.getMainLooper())
                } else {
                    latch.countDown()
                }
            } catch (e: Exception) {
                Log.w(TAG, "Could not request fresh location update", e)
                latch.countDown()
            }
        }

        try {
            latch.await(3000, java.util.concurrent.TimeUnit.MILLISECONDS)
        } catch (_: InterruptedException) {}

        return resultHolder.get()
    }

    private fun isBetterLocation(location: Location, currentBest: Location?): Boolean {
        if (currentBest == null) return true
        val timeDiff = location.time - currentBest.time
        val isSignificantlyNewer = timeDiff > 120000L
        val isSignificantlyOlder = timeDiff < -120000L
        val isNewer = timeDiff > 0

        if (isSignificantlyNewer) return true
        if (isSignificantlyOlder) return false

        val accuracyDiff = (location.accuracy - currentBest.accuracy).toInt()
        val isMoreAccurate = accuracyDiff < 0
        val isSameProvider = location.provider == currentBest.provider

        return if (isMoreAccurate) true else if (isNewer && isSameProvider) true else false
    }

    private fun buildEmergencyMessage(location: Location?): String {
        return if (location != null) {
            val lat = "%.6f".format(Locale.US, location.latitude)
            val lon = "%.6f".format(Locale.US, location.longitude)
            val acc = location.accuracy.toInt()
            val mapsUrl = "https://maps.google.com/?q=$lat,$lon"
            val address = reverseGeocode(location)

            val addressPart = if (address.isNotEmpty()) "\nNear: $address" else ""
            "EMERGENCY SOS: I need help!\nLocation: $mapsUrl\nCoordinates: Lat $lat, Lon $lon (±${acc}m)$addressPart"
        } else {
            "EMERGENCY SOS: I need help! Please contact me immediately. (GPS location was unavailable at the moment of trigger)."
        }
    }

    private fun reverseGeocode(location: Location): String {
        return try {
            if (Geocoder.isPresent()) {
                val geocoder = Geocoder(context, Locale.getDefault())
                @Suppress("DEPRECATION")
                val list = geocoder.getFromLocation(location.latitude, location.longitude, 1)
                if (!list.isNullOrEmpty()) {
                    val addr = list[0]
                    val parts = mutableListOf<String>()
                    addr.thoroughfare?.let { parts.add(it) }
                    addr.subLocality?.let { parts.add(it) }
                    addr.locality?.let { parts.add(it) }
                    parts.joinToString(", ")
                } else ""
            } else ""
        } catch (e: Exception) {
            ""
        }
    }

    private fun formatLocationDetails(location: Location?): String {
        if (location == null) return "Coordinates unavailable"
        val lat = "%.5f".format(Locale.US, location.latitude)
        val lon = "%.5f".format(Locale.US, location.longitude)
        val acc = location.accuracy.toInt()
        return "Lat: $lat, Lon: $lon (±${acc}m)"
    }

    private fun getSmsManager(): SmsManager {
        return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            context.getSystemService(SmsManager::class.java)
        } else {
            @Suppress("DEPRECATION")
            SmsManager.getDefault()
        }
    }

    private fun postStatus(status: SosStatus) {
        mainHandler.post {
            onStatusChanged?.invoke(status)
        }
    }
}
