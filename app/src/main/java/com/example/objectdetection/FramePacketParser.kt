package com.example.objectdetection

import android.util.Log
import java.io.ByteArrayOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * State-machine parser for the TCP Frame Packet binary protocol.
 *
 * Binary Frame Format:
 * [0..3]   MAGIC: "PIFR" (0x50, 0x49, 0x46, 0x52)
 * [4]      VERSION: 0x01
 * [5]      FORMAT: 0x01 (JPEG)
 * [6..9]   FRAME_ID: uint32 (Big Endian)
 * [10..17] TIMESTAMP: int64 (Big Endian)
 * [18..21] PAYLOAD_LENGTH: uint32 (Big Endian)
 * [22..22+PAYLOAD_LENGTH-1] JPEG Payload Data
 */
class FramePacketParser(
    private val onFrameParsed: (FramePacket) -> Unit
) {
    companion object {
        private const val TAG = "FramePacketParser"
        const val HEADER_SIZE = 22
        val MAGIC_BYTES = byteArrayOf('P'.code.toByte(), 'I'.code.toByte(), 'F'.code.toByte(), 'R'.code.toByte())
        const val MAX_PAYLOAD_SIZE = 10 * 1024 * 1024 // 10 MB sanity limit
    }

    private val streamBuffer = ByteArrayOutputStream(64 * 1024)

    @Synchronized
    fun feedData(data: ByteArray, length: Int) {
        if (length <= 0) return
        streamBuffer.write(data, 0, length)
        processBuffer()
    }

    @Synchronized
    fun reset() {
        streamBuffer.reset()
    }

    private fun processBuffer() {
        var rawBytes = streamBuffer.toByteArray()
        var offset = 0

        while (rawBytes.size - offset >= HEADER_SIZE) {
            // 1. Find Magic Bytes "PIFR"
            val magicIndex = findMagicIndex(rawBytes, offset)
            if (magicIndex == -1) {
                // Magic not found, discard all but the last 3 bytes (in case magic is split across chunks)
                val keepBytes = minOf(rawBytes.size - offset, 3)
                val newBuffer = ByteArrayOutputStream()
                if (keepBytes > 0) {
                    newBuffer.write(rawBytes, rawBytes.size - keepBytes, keepBytes)
                }
                streamBuffer.reset()
                newBuffer.writeTo(streamBuffer)
                return
            }

            offset = magicIndex
            if (rawBytes.size - offset < HEADER_SIZE) {
                // Header incomplete, wait for more data
                break
            }

            // 2. Parse Header
            val headerBuffer = ByteBuffer.wrap(rawBytes, offset, HEADER_SIZE).order(ByteOrder.BIG_ENDIAN)
            // Skip Magic (4 bytes)
            headerBuffer.position(4)
            val version = headerBuffer.get()
            val format = headerBuffer.get()
            val frameId = headerBuffer.int.toLong() and 0xFFFFFFFFL
            val timestampMs = headerBuffer.long
            val payloadLength = headerBuffer.int

            if (payloadLength < 0 || payloadLength > MAX_PAYLOAD_SIZE) {
                Log.w(TAG, "Invalid payload length: $payloadLength, recovering stream...")
                offset += 4 // Skip magic to search for next packet
                continue
            }

            val totalPacketSize = HEADER_SIZE + payloadLength
            if (rawBytes.size - offset < totalPacketSize) {
                // Incomplete payload, wait for more data
                break
            }

            // 3. Extract Payload
            val jpegData = ByteArray(payloadLength)
            System.arraycopy(rawBytes, offset + HEADER_SIZE, jpegData, 0, payloadLength)

            val packet = FramePacket(
                frameId = frameId,
                timestampMs = timestampMs,
                format = format,
                jpegData = jpegData
            )

            try {
                onFrameParsed(packet)
            } catch (e: Exception) {
                Log.e(TAG, "Error in onFrameParsed callback", e)
            }

            offset += totalPacketSize
        }

        // Keep remaining unparsed bytes in buffer
        val remainingBytes = rawBytes.size - offset
        streamBuffer.reset()
        if (remainingBytes > 0) {
            streamBuffer.write(rawBytes, offset, remainingBytes)
        }
    }

    private fun findMagicIndex(data: ByteArray, startOffset: Int): Int {
        val maxCheck = data.size - 4
        for (i in startOffset..maxCheck) {
            if (data[i] == MAGIC_BYTES[0] &&
                data[i + 1] == MAGIC_BYTES[1] &&
                data[i + 2] == MAGIC_BYTES[2] &&
                data[i + 3] == MAGIC_BYTES[3]
            ) {
                return i
            }
        }
        return -1
    }
}
