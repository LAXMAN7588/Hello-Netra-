package com.example.objectdetection

import android.content.Context
import android.content.SharedPreferences
import android.util.Log
import org.json.JSONArray
import org.json.JSONObject
import java.util.concurrent.CopyOnWriteArrayList

/**
 * Thread-safe persistent repository for Emergency Contacts stored in local SharedPreferences.
 * Operates 100% locally with zero internet dependency and zero cloud server requirements.
 */
class EmergencyContactRepository(context: Context) {

    companion object {
        private const val TAG = "EmergencyContactRepo"
        private const val PREFS_NAME = "emergency_contacts_prefs"
        private const val KEY_CONTACTS_JSON = "contacts_json"
    }

    private val prefs: SharedPreferences = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
    private val memoryList = CopyOnWriteArrayList<EmergencyContact>()

    interface ContactChangeListener {
        fun onContactsUpdated(contacts: List<EmergencyContact>)
    }

    private val listeners = CopyOnWriteArrayList<ContactChangeListener>()

    init {
        loadFromDisk()
    }

    fun addListener(listener: ContactChangeListener) {
        listeners.add(listener)
    }

    fun removeListener(listener: ContactChangeListener) {
        listeners.remove(listener)
    }

    @Synchronized
    fun getContacts(): List<EmergencyContact> {
        return memoryList.toList()
    }

    @Synchronized
    fun addContact(name: String, phoneNumber: String): EmergencyContact {
        val cleanName = name.trim()
        val cleanPhone = phoneNumber.trim().replace(" ", "").replace("-", "")
        val contact = EmergencyContact(name = cleanName, phoneNumber = cleanPhone)

        memoryList.add(contact)
        saveToDisk()
        notifyListeners()
        Log.i(TAG, "Added emergency contact: $cleanName ($cleanPhone). Total contacts: ${memoryList.size}")
        return contact
    }

    @Synchronized
    fun deleteContact(contactId: String): Boolean {
        val removed = memoryList.removeAll { it.id == contactId }
        if (removed) {
            saveToDisk()
            notifyListeners()
            Log.i(TAG, "Deleted emergency contact id: $contactId. Remaining: ${memoryList.size}")
        }
        return removed
    }

    @Synchronized
    fun clearAll() {
        memoryList.clear()
        saveToDisk()
        notifyListeners()
    }

    private fun loadFromDisk() {
        memoryList.clear()
        val jsonStr = prefs.getString(KEY_CONTACTS_JSON, null) ?: return

        try {
            val jsonArray = JSONArray(jsonStr)
            for (i in 0 until jsonArray.length()) {
                val obj = jsonArray.getJSONObject(i)
                val id = obj.optString("id", "")
                val name = obj.optString("name", "")
                val phone = obj.optString("phone", "")

                if (name.isNotEmpty() && phone.isNotEmpty()) {
                    memoryList.add(
                        EmergencyContact(
                            id = if (id.isNotEmpty()) id else java.util.UUID.randomUUID().toString(),
                            name = name,
                            phoneNumber = phone
                        )
                    )
                }
            }
            Log.i(TAG, "Loaded ${memoryList.size} emergency contacts from persistent storage.")
        } catch (e: Exception) {
            Log.e(TAG, "Error parsing saved contacts JSON", e)
        }
    }

    private fun saveToDisk() {
        try {
            val jsonArray = JSONArray()
            for (contact in memoryList) {
                val obj = JSONObject().apply {
                    put("id", contact.id)
                    put("name", contact.name)
                    put("phone", contact.phoneNumber)
                }
                jsonArray.put(obj)
            }
            prefs.edit().putString(KEY_CONTACTS_JSON, jsonArray.toString()).apply()
        } catch (e: Exception) {
            Log.e(TAG, "Error saving contacts to persistent storage", e)
        }
    }

    private fun notifyListeners() {
        val current = getContacts()
        for (listener in listeners) {
            try {
                listener.onContactsUpdated(current)
            } catch (e: Exception) {
                Log.e(TAG, "Error notifying contact listener", e)
            }
        }
    }
}
