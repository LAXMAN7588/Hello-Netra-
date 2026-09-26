package com.example.objectdetection

import android.util.Log
import java.net.Inet4Address
import java.net.NetworkInterface
import java.util.Collections

object IpUtils {
    private const val TAG = "IpUtils"

    /**
     * Finds the local IP address for the Android hotspot or Wi-Fi interface.
     */
    fun getLocalIpAddress(): String {
        val foundIps = mutableListOf<String>()

        try {
            val interfaces = Collections.list(NetworkInterface.getNetworkInterfaces())
            for (networkInterface in interfaces) {
                if (!networkInterface.isUp || networkInterface.isLoopback) continue

                val addresses = Collections.list(networkInterface.inetAddresses)
                for (address in addresses) {
                    if (!address.isLoopbackAddress && address is Inet4Address) {
                        val ip = address.hostAddress ?: continue
                        val ifName = networkInterface.name.lowercase()
                        Log.d(TAG, "Found network interface: $ifName -> $ip")

                        // Prioritize standard Android hotspot interfaces (ap0, swlan, wlan)
                        if (ifName.contains("ap") || ifName.contains("hotspot") || ip.startsWith("192.168.43.")) {
                            return ip
                        }
                        foundIps.add(ip)
                    }
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error discovering local IP address", e)
        }

        // Return first non-loopback IPv4 if available
        if (foundIps.isNotEmpty()) {
            return foundIps.first()
        }

        // Standard default Android Mobile Hotspot IP
        return "192.168.43.1"
    }
}
