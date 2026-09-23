package com.example.objectdetection

import java.util.UUID

/**
 * Represents an emergency contact configured on the device.
 */
data class EmergencyContact(
    val id: String = UUID.randomUUID().toString(),
    val name: String,
    val phoneNumber: String
)
