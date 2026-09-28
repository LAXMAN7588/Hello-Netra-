package com.example.objectdetection

import android.Manifest
import android.app.Activity
import android.app.PendingIntent
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.content.pm.PackageManager
import android.location.Location
import android.location.LocationListener
import android.location.LocationManager
import android.net.ConnectivityManager
import android.net.NetworkCapabilities
import android.os.Build
import android.provider.ContactsContract
import android.telephony.SmsManager
import androidx.core.content.ContextCompat
import com.google.android.gms.location.CurrentLocationRequest
import com.google.android.gms.location.FusedLocationProviderClient
import com.google.android.gms.location.LocationServices
import com.google.android.gms.location.Priority
import com.google.android.gms.common.api.ResolvableApiException
import com.google.android.gms.location.LocationSettingsRequest
import com.google.android.gms.location.LocationRequest
import java.util.Locale

class SosManager(
    private val context: Context,
    private val statusCallback: ((String) -> Unit)? = null
) {

    data class EmergencyContact(
        val phone: String,
        val priority: Int
    )

    companion object {
        private const val CONTACTS_KEY = "contacts"
        private const val PREFS_NAME = "emergency_contacts"

        private const val ACTION_SMS_SENT =
            "com.example.objectdetection.SMS_SENT"

        private const val ACTION_SMS_DELIVERED =
            "com.example.objectdetection.SMS_DELIVERED"

        private const val EXTRA_PHONE = "phone"
        private const val EXTRA_PART = "part"
    }

    private val fusedLocationClient: FusedLocationProviderClient =
        LocationServices.getFusedLocationProviderClient(context)

    private val locationManager: LocationManager =
        context.getSystemService(Context.LOCATION_SERVICE) as LocationManager

    private val preferences =
        context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)

    private val emergencyContacts =
        mutableListOf<EmergencyContact>()

    private var receiversRegistered = false

    private val smsSentReceiver = object : BroadcastReceiver() {
        override fun onReceive(context: Context?, intent: Intent?) {
            val phone = intent?.getStringExtra(EXTRA_PHONE) ?: return

            when (resultCode) {
                android.app.Activity.RESULT_OK -> {
                    statusCallback?.invoke("SMS accepted for sending to $phone")
                }

                else -> {
                    statusCallback?.invoke(
                        "SMS failed for $phone. Error code: $resultCode"
                    )
                }
            }
        }
    }

    private val smsDeliveredReceiver = object : BroadcastReceiver() {
        override fun onReceive(context: Context?, intent: Intent?) {
            val phone = intent?.getStringExtra(EXTRA_PHONE) ?: return

            when (resultCode) {
                android.app.Activity.RESULT_OK -> {
                    statusCallback?.invoke("SMS delivered to $phone")
                }

                else -> {
                    statusCallback?.invoke(
                        "SMS delivery was not confirmed for $phone"
                    )
                }
            }
        }
    }

    init {
        loadEmergencyContacts()
        registerSmsReceivers()
    }

    // =========================================================
    // PERMISSIONS
    // =========================================================

    fun hasLocationPermission(): Boolean {
        val fine =
            ContextCompat.checkSelfPermission(
                context,
                Manifest.permission.ACCESS_FINE_LOCATION
            ) == PackageManager.PERMISSION_GRANTED

        val coarse =
            ContextCompat.checkSelfPermission(
                context,
                Manifest.permission.ACCESS_COARSE_LOCATION
            ) == PackageManager.PERMISSION_GRANTED

        return fine || coarse
    }

    fun hasSmsPermission(): Boolean {
        return ContextCompat.checkSelfPermission(
            context,
            Manifest.permission.SEND_SMS
        ) == PackageManager.PERMISSION_GRANTED
    }

    fun hasCallPermission(): Boolean {
        return ContextCompat.checkSelfPermission(
            context,
            Manifest.permission.CALL_PHONE
        ) == PackageManager.PERMISSION_GRANTED
    }

    fun hasContactsPermission(): Boolean {
        return ContextCompat.checkSelfPermission(
            context,
            Manifest.permission.READ_CONTACTS
        ) == PackageManager.PERMISSION_GRANTED
    }

    fun hasInternet(): Boolean {
        val connectivityManager =
            context.getSystemService(
                Context.CONNECTIVITY_SERVICE
            ) as ConnectivityManager

        val network =
            connectivityManager.activeNetwork ?: return false

        val capabilities =
            connectivityManager.getNetworkCapabilities(network)
                ?: return false

        return capabilities.hasCapability(
            NetworkCapabilities.NET_CAPABILITY_INTERNET
        )
    }

    // =========================================================
    // CONTACTS
    // =========================================================

    fun getEmergencyContacts(): List<EmergencyContact> {
        return emergencyContacts
            .sortedBy { it.priority }
            .toList()
    }

    fun addEmergencyContact(number: String): Boolean {
        val cleanedNumber = number.trim()

        if (cleanedNumber.isBlank()) {
            return false
        }

        if (
            emergencyContacts.any {
                it.phone == cleanedNumber
            }
        ) {
            statusCallback?.invoke(
                "This contact is already saved"
            )
            return false
        }

        val nextPriority = getNextAvailablePriority()

        emergencyContacts.add(
            EmergencyContact(
                phone = cleanedNumber,
                priority = nextPriority
            )
        )

        saveEmergencyContacts()

        statusCallback?.invoke(
            "Emergency contact added"
        )

        return true
    }

    fun deleteEmergencyContact(contact: EmergencyContact) {
        emergencyContacts.remove(contact)
        saveEmergencyContacts()

        statusCallback?.invoke(
            "Contact deleted"
        )
    }

    private fun getNextAvailablePriority(): Int {
        var priority = 1

        while (
            emergencyContacts.any {
                it.priority == priority
            }
        ) {
            priority++
        }

        return priority
    }

    private fun saveEmergencyContacts() {
        val encodedContacts =
            emergencyContacts
                .map {
                    "${it.phone}|${it.priority}"
                }
                .toSet()

        preferences
            .edit()
            .putStringSet(
                CONTACTS_KEY,
                encodedContacts
            )
            .apply()
    }

    private fun loadEmergencyContacts() {
        emergencyContacts.clear()

        val saved =
            preferences.getStringSet(
                CONTACTS_KEY,
                emptySet()
            )

        if (saved.isNullOrEmpty()) {
            return
        }

        saved.forEach { value ->

            val separator =
                value.lastIndexOf("|")

            if (separator > 0) {

                val phone =
                    value.substring(
                        0,
                        separator
                    )

                val priority =
                    value.substring(
                        separator + 1
                    ).toIntOrNull()

                if (
                    phone.isNotBlank() &&
                    priority != null
                ) {
                    emergencyContacts.add(
                        EmergencyContact(
                            phone = phone,
                            priority = priority
                        )
                    )
                }

            } else {

                // Migration from the old phone-only format.
                if (value.isNotBlank()) {
                    emergencyContacts.add(
                        EmergencyContact(
                            phone = value,
                            priority =
                                getNextAvailablePriority()
                        )
                    )
                }
            }
        }
    }

    // =========================================================
    // CONTACT PICKER
    // =========================================================

    fun createContactPickerIntent(): Intent {
        return Intent(
            Intent.ACTION_PICK,
            ContactsContract.CommonDataKinds.Phone.CONTENT_URI
        )
    }

    // =========================================================
    // SOS ENTRY POINT
    // =========================================================

    
    fun startSosFromBackground() {
        if (emergencyContacts.isEmpty()) {
            statusCallback?.invoke("No emergency contacts saved")
            return
        }

        if (!hasSmsPermission() || !hasLocationPermission()) {
            statusCallback?.invoke("SMS or Location permission missing")
            return
        }

        statusCallback?.invoke("Getting location for background SOS...")

        getSosLocation(
            activity = null,
            onCallPermissionRequired = {}
        )
    }

    fun startSos(
        activity: Activity,
        onLocationRequired: () -> Unit = {},
        onSmsPermissionRequired: () -> Unit = {},
        onCallPermissionRequired: () -> Unit = {}
    ) {

        if (emergencyContacts.isEmpty()) {
            statusCallback?.invoke(
                "Add at least one emergency contact first"
            )
            return
        }

        if (!hasSmsPermission()) {
            statusCallback?.invoke(
                "SMS permission is required"
            )
            onSmsPermissionRequired()
            return
        }

        if (!hasLocationPermission()) {
            statusCallback?.invoke(
                "Location permission is required"
            )
            onLocationRequired()
            return
        }

        statusCallback?.invoke(
            "Getting your latest location..."
        )

        getSosLocation(
            activity = activity,
            onCallPermissionRequired = onCallPermissionRequired
        )
    }

    // =========================================================
    // SOS LOCATION
    // =========================================================

    
    fun isLocationProviderEnabled(): Boolean {
        val gpsEnabled = try { locationManager.isProviderEnabled(LocationManager.GPS_PROVIDER) } catch (_: Exception) { false }
        val networkEnabled = try { locationManager.isProviderEnabled(LocationManager.NETWORK_PROVIDER) } catch (_: Exception) { false }
        return gpsEnabled || networkEnabled
    }

    private fun promptEnableLocationServices(activity: Activity, onEnabled: () -> Unit) {
        try {
            val locationRequest = LocationRequest.Builder(
                Priority.PRIORITY_HIGH_ACCURACY, 5000
            ).build()

            val builder = LocationSettingsRequest.Builder()
                .addLocationRequest(locationRequest)
                .setAlwaysShow(true)

            val client = LocationServices.getSettingsClient(context)
            client.checkLocationSettings(builder.build())
                .addOnSuccessListener {
                    onEnabled()
                }
                .addOnFailureListener { exception ->
                    if (exception is ResolvableApiException) {
                        try {
                            exception.startResolutionForResult(activity, 1001)
                        } catch (e: Exception) {
                            statusCallback?.invoke("Location services disabled. Please turn on Location in Settings.")
                        }
                    } else {
                        statusCallback?.invoke("Location services disabled. Please turn on Location in Settings.")
                    }
                }
        } catch (e: Exception) {
            statusCallback?.invoke("Location services disabled. Please turn on Location in Settings.")
        }
    }

    private fun getSosLocation(
        activity: Activity? = null,
        onCallPermissionRequired: () -> Unit
    ) {

        if (!hasLocationPermission()) {
            statusCallback?.invoke(
                "Location permission is required"
            )
            return
        }

        if (!isLocationProviderEnabled()) {
            statusCallback?.invoke("Location Services disabled on device.")
            if (activity != null) {
                promptEnableLocationServices(activity) {
                    getSosLocation(activity, onCallPermissionRequired)
                }
                return
            }
        }

        if (hasInternet()) {
            getFusedLocationForSos(
                activity,
                onCallPermissionRequired
            )
        } else {
            getGpsLocationForSos(
                activity,
                onCallPermissionRequired
            )
        }
    }

    private fun getFusedLocationForSos(
        activity: Activity? = null,
        onCallPermissionRequired: () -> Unit
    ) {

        if (!hasLocationPermission()) {
            return
        }

        try {

            val request =
                CurrentLocationRequest.Builder()
                    .setPriority(
                        Priority.PRIORITY_HIGH_ACCURACY
                    )
                    .setMaxUpdateAgeMillis(3000)
                    .setDurationMillis(15000)
                    .build()

            fusedLocationClient
                .getCurrentLocation(
                    request,
                    null
                )
                .addOnSuccessListener { location ->

                    if (location != null) {

                        sendSos(
                            activity,
                            location,
                            onCallPermissionRequired
                        )

                    } else {

                        getLastLocationForSos(
                            activity,
                            onCallPermissionRequired
                        )
                    }
                }
                .addOnFailureListener {

                    getLastLocationForSos(
                        activity,
                        onCallPermissionRequired
                    )
                }

        } catch (_: SecurityException) {

            statusCallback?.invoke(
                "Location permission is required"
            )
        }
    }

    private fun getLastLocationForSos(
        activity: Activity? = null,
        onCallPermissionRequired: () -> Unit
    ) {

        if (!hasLocationPermission()) {
            return
        }

        try {

            fusedLocationClient
                .lastLocation
                .addOnSuccessListener { location ->

                    if (location != null) {

                        sendSos(
                            activity,
                            location,
                            onCallPermissionRequired
                        )

                    } else {

                        statusCallback?.invoke(
                            "Unable to get a location. SOS was not sent."
                        )
                    }
                }
                .addOnFailureListener {

                    statusCallback?.invoke(
                        "Unable to get a location. SOS was not sent."
                    )
                }

        } catch (_: SecurityException) {

            statusCallback?.invoke(
                "Location permission is required"
            )
        }
    }

    private fun getGpsLocationForSos(
        activity: Activity? = null,
        onCallPermissionRequired: () -> Unit
    ) {

        if (!hasLocationPermission()) {
            return
        }

        try {

            if (
                !locationManager.isProviderEnabled(
                    LocationManager.GPS_PROVIDER
                )
            ) {

                statusCallback?.invoke(
                    "GPS is disabled. Please enable Location/GPS."
                )

                return
            }

            val listener =
                object : LocationListener {

                    override fun onLocationChanged(
                        location: Location
                    ) {

                        locationManager.removeUpdates(
                            this
                        )

                        sendSos(
                            activity,
                            location,
                            onCallPermissionRequired
                        )
                    }
                }

            locationManager.requestLocationUpdates(
                LocationManager.GPS_PROVIDER,
                1000L,
                0f,
                listener,
                android.os.Looper.getMainLooper()
            )

        } catch (_: SecurityException) {

            statusCallback?.invoke(
                "Location permission is required"
            )

        } catch (_: Exception) {

            statusCallback?.invoke(
                "Unable to get a location. SOS was not sent."
            )
        }
    }

    // =========================================================
    // BUILD AND SEND SOS
    // =========================================================

    private fun sendSos(
        activity: Activity? = null,
        location: Location,
        onCallPermissionRequired: () -> Unit
    ) {

        val latitude = location.latitude
        val longitude = location.longitude

        val mapsUrl =
            String.format(
                Locale.US,
                "https://maps.google.com/?q=%.6f,%.6f",
                latitude,
                longitude
            )

        val message =
            "SOS! My current location is:\n" +
                    "Latitude: %.6f\n".format(
                        Locale.US,
                        latitude
                    ) +
                    "Longitude: %.6f\n".format(
                        Locale.US,
                        longitude
                    ) +
                    "Google Maps: $mapsUrl"

        statusCallback?.invoke(
            "Sending SOS to emergency contacts..."
        )

        sendSmsToAllContacts(
            activity = activity,
            message = message
        )

        callPriorityContact(
            activity = activity,
            onPermissionRequired = onCallPermissionRequired
        )
    }

    private fun sendSmsToAllContacts(
        activity: Activity? = null,
        message: String
    ) {

        if (!hasSmsPermission()) {
            statusCallback?.invoke(
                "SMS permission is required"
            )
            return
        }

        if (
            !context.packageManager.hasSystemFeature(
                PackageManager.FEATURE_TELEPHONY
            )
        ) {
            statusCallback?.invoke(
                "This phone does not support SMS"
            )
            return
        }

        val smsManager =
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
                context.getSystemService(
                    SmsManager::class.java
                )
            } else {
                @Suppress("DEPRECATION")
                SmsManager.getDefault()
            }

        val sortedContacts =
            emergencyContacts.sortedBy {
                it.priority
            }

        sortedContacts.forEachIndexed {
            index,
            contact ->

            val phone = contact.phone

            try {

                if (message.length <= 160) {

                    val sentIntent =
                        Intent(
                            ACTION_SMS_SENT
                        ).apply {

                            setPackage(
                                context.packageName
                            )

                            putExtra(
                                EXTRA_PHONE,
                                phone
                            )
                        }

                    val deliveredIntent =
                        Intent(
                            ACTION_SMS_DELIVERED
                        ).apply {

                            setPackage(
                                context.packageName
                            )

                            putExtra(
                                EXTRA_PHONE,
                                phone
                            )
                        }

                    val sentPendingIntent =
                        PendingIntent.getBroadcast(
                            context,
                            index + 100,
                            sentIntent,
                            PendingIntent.FLAG_UPDATE_CURRENT or
                                    PendingIntent.FLAG_IMMUTABLE
                        )

                    val deliveredPendingIntent =
                        PendingIntent.getBroadcast(
                            context,
                            index + 200,
                            deliveredIntent,
                            PendingIntent.FLAG_UPDATE_CURRENT or
                                    PendingIntent.FLAG_IMMUTABLE
                        )

                    smsManager.sendTextMessage(
                        phone,
                        null,
                        message,
                        sentPendingIntent,
                        deliveredPendingIntent
                    )

                } else {

                    val parts =
                        smsManager.divideMessage(
                            message
                        )

                    val sentIntents =
                        ArrayList<PendingIntent>()

                    val deliveredIntents =
                        ArrayList<PendingIntent>()

                    parts.forEachIndexed {
                        partIndex,
                        _ ->

                        val partSentIntent =
                            Intent(
                                ACTION_SMS_SENT
                            ).apply {

                                setPackage(
                                    context.packageName
                                )

                                putExtra(
                                    EXTRA_PHONE,
                                    phone
                                )

                                putExtra(
                                    EXTRA_PART,
                                    partIndex
                                )
                            }

                        val partDeliveredIntent =
                            Intent(
                                ACTION_SMS_DELIVERED
                            ).apply {

                                setPackage(
                                    context.packageName
                                )

                                putExtra(
                                    EXTRA_PHONE,
                                    phone
                                )

                                putExtra(
                                    EXTRA_PART,
                                    partIndex
                                )
                            }

                        sentIntents.add(
                            PendingIntent.getBroadcast(
                                context,
                                1000 +
                                        index * 100 +
                                        partIndex,
                                partSentIntent,
                                PendingIntent.FLAG_UPDATE_CURRENT or
                                        PendingIntent.FLAG_IMMUTABLE
                            )
                        )

                        deliveredIntents.add(
                            PendingIntent.getBroadcast(
                                context,
                                2000 +
                                        index * 100 +
                                        partIndex,
                                partDeliveredIntent,
                                PendingIntent.FLAG_UPDATE_CURRENT or
                                        PendingIntent.FLAG_IMMUTABLE
                            )
                        )
                    }

                    smsManager.sendMultipartTextMessage(
                        phone,
                        null,
                        parts,
                        sentIntents,
                        deliveredIntents
                    )
                }

            } catch (_: SecurityException) {

                statusCallback?.invoke(
                    "SMS permission is required"
                )

            } catch (_: Exception) {

                statusCallback?.invoke(
                    "Could not send SMS to $phone"
                )
            }
        }

        statusCallback?.invoke(
            "SOS SMS sending started"
        )
    }

    // =========================================================
    // PRIORITY CALL
    // =========================================================

    private fun callPriorityContact(
        activity: Activity? = null,
        onPermissionRequired: () -> Unit
    ) {

        val priorityContact =
            emergencyContacts.minByOrNull {
                it.priority
            } ?: return

        if (!hasCallPermission()) {

            onPermissionRequired()
            return
        }

        try {

            val callIntent =
                Intent(
                    Intent.ACTION_CALL
                ).apply {

                    data =
                        android.net.Uri.parse(
                            "tel:${priorityContact.phone}"
                        )
                }

            if (activity != null) {
                activity.startActivity(callIntent)
            } else {
                callIntent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
                context.startActivity(callIntent)
            }

        } catch (_: SecurityException) {

            statusCallback?.invoke(
                "Phone call permission is required."
            )

        } catch (_: Exception) {

            statusCallback?.invoke(
                "Unable to start the phone call."
            )
        }
    }

    // =========================================================
    // SMS RECEIVERS
    // =========================================================

    private fun registerSmsReceivers() {

        if (receiversRegistered) {
            return
        }

        val sentFilter =
            IntentFilter(
                ACTION_SMS_SENT
            )

        val deliveredFilter =
            IntentFilter(
                ACTION_SMS_DELIVERED
            )

        if (
            Build.VERSION.SDK_INT >=
            Build.VERSION_CODES.TIRAMISU
        ) {

            context.registerReceiver(
                smsSentReceiver,
                sentFilter,
                Context.RECEIVER_EXPORTED
            )

            context.registerReceiver(
                smsDeliveredReceiver,
                deliveredFilter,
                Context.RECEIVER_EXPORTED
            )

        } else {

            @Suppress("DEPRECATION")
            context.registerReceiver(
                smsSentReceiver,
                sentFilter
            )

            @Suppress("DEPRECATION")
            context.registerReceiver(
                smsDeliveredReceiver,
                deliveredFilter
            )
        }

        receiversRegistered = true
    }

    fun destroy() {

        if (!receiversRegistered) {
            return
        }

        try {
            context.unregisterReceiver(
                smsSentReceiver
            )
        } catch (_: IllegalArgumentException) {
        }

        try {
            context.unregisterReceiver(
                smsDeliveredReceiver
            )
        } catch (_: IllegalArgumentException) {
        }

        receiversRegistered = false
    }
}
