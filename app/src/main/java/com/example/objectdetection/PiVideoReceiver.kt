package com.example.objectdetection

import java.io.ByteArrayOutputStream

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.SystemClock
import android.util.Log
import java.net.DatagramPacket
import java.net.DatagramSocket
import java.net.InetSocketAddress
import java.util.TreeMap
import java.util.concurrent.atomic.AtomicBoolean

/**
 * NETRA Raspberry Pi camera receiver.
 *
 * Exact camera.py packet protocol:
 *
 * 0..3    NETR
 * 4..7    frame number, uint32 big-endian
 * 8..9    chunk number, uint16 big-endian
 * 10..11  total chunks, uint16 big-endian
 * 12..    JPEG chunk payload
 *
 * UDP port: 5000
 */
class PiVideoReceiver(
    private val width: Int = 640,
    private val height: Int = 480,
    private val port: Int = 5000,
    private val onVideoFrame: (Bitmap) -> Unit,
    private val onStatus: (String) -> Unit,
    private val onStats: (Int, Double) -> Unit
) {

    companion object {
        private const val TAG = "NETRA_UDP"

        private const val HEADER_SIZE = 12
        private const val MAX_PACKET_SIZE = 1500
        private const val MAX_FRAME_SIZE = 5 * 1024 * 1024
        private const val FRAME_TIMEOUT_MS = 500L

        private const val MAGIC_0 = 'N'.code.toByte()
        private const val MAGIC_1 = 'E'.code.toByte()
        private const val MAGIC_2 = 'T'.code.toByte()
        private const val MAGIC_3 = 'R'.code.toByte()
    }

    private data class FrameBuffer(
        val frameId: Int,
        val packetCount: Int,
        val packets: Array<ByteArray?>,
        var received: Int = 0,
        val createdAt: Long = SystemClock.elapsedRealtime()
    )

    private val running = AtomicBoolean(false)

    private var socket: DatagramSocket? = null
    private var receiveThread: Thread? = null

    private val frames = TreeMap<Int, FrameBuffer>()
    private val frameLock = Any()

    private var lastFrameId = -1

    private var frameCounter = 0
    private var fpsStartTime = SystemClock.elapsedRealtime()
    private var fpsFrameCount = 0

    fun start() {
        if (running.getAndSet(true)) {
            Log.d(TAG, "UDP receiver already running")
            return
        }

        frameCounter = 0
        fpsFrameCount = 0
        fpsStartTime = SystemClock.elapsedRealtime()
        lastFrameId = -1

        synchronized(frameLock) {
            frames.clear()
        }

        try {
            Log.i(TAG, "NETRA_UDP bind attempt: port=$port width=$width height=$height")
            socket = DatagramSocket(null).apply {
                reuseAddress = true
                receiveBufferSize = 4 * 1024 * 1024
                bind(InetSocketAddress(5000))
            }
        } catch (e: Exception) {
            running.set(false)
            socket = null

            Log.e(TAG, "Failed to bind UDP:$port", e)
            onStatus("UDP bind error: ${e.message}")
            return
        }

        onStatus("UDP listening on port $port")
        Log.i(TAG, "NETRA_UDP listening on UDP:$port")

        receiveThread = Thread {
            receiveLoop()
        }.apply {
            name = "NetraUdpReceiver"
            isDaemon = true
        }

        receiveThread?.start()
    }

    /**
     * Sends a command to the Raspberry Pi using the SAME UDP socket
     * that is bound to Android UDP port 5000.
     *
     * This is required because camera.py sends the video stream back
     * to the source IP/port of the START command.
     */
    fun sendCommand(host: String, command: String): Boolean {
        val currentSocket = socket

        if (!running.get() || currentSocket == null || currentSocket.isClosed) {
            Log.e(TAG, "Cannot send command: UDP receiver is not running")
            onStatus("UDP receiver is not ready")
            return false
        }

        return try {
            val bytes = command.trim().toByteArray(Charsets.UTF_8)

            val packet = DatagramPacket(
                bytes,
                bytes.size,
                InetSocketAddress(host.trim(), port)
            )

            currentSocket.send(packet)

            Log.i(
                TAG,
                "NETRA_UDP sent '$command' to $host:$port from local UDP:$port"
            )

            onStatus("Sent $command to $host:$port")
            true

        } catch (e: Exception) {
            Log.e(TAG, "Failed to send UDP command '$command' to $host:$port", e)
            onStatus("UDP command failed: ${e.message}")
            false
        }
    }

    fun sendStart(host: String): Boolean {
        return sendCommand(host, "START")
    }

    fun sendStop(host: String): Boolean {
        return sendCommand(host, "STOP")
    }

    private fun receiveLoop() {
        try {
            val buffer = ByteArray(MAX_PACKET_SIZE)

            while (running.get()) {
                val packet = DatagramPacket(buffer, buffer.size)

                try {
                    socket?.receive(packet)
                } catch (e: Exception) {
                    if (running.get()) {
                        Log.e(TAG, "UDP receive error", e)
                    }
                    break
                }

                if (!running.get()) {
                    break
                }

                if (packet.length >= HEADER_SIZE) {
                    processPacket(packet.data, packet.length)
                }

                cleanupOldFrames()
            }

        } catch (e: Exception) {
            if (running.get()) {
                Log.e(TAG, "UDP receiver failure", e)
                onStatus("UDP error: ${e.message}")
            }
        } finally {
            try {
                socket?.close()
            } catch (_: Exception) {
            }

            socket = null
        }
    }

    private fun processPacket(
        data: ByteArray,
        length: Int
    ) {
        if (length < HEADER_SIZE) {
            return
        }

        if (
            data[0] != MAGIC_0 ||
            data[1] != MAGIC_1 ||
            data[2] != MAGIC_2 ||
            data[3] != MAGIC_3
        ) {
            return
        }

        val frameId = readInt(data, 4)
        val chunkId = readUnsignedShort(data, 8)
        val totalChunks = readUnsignedShort(data, 10)

        if (totalChunks <= 0) {
            return
        }

        if (chunkId >= totalChunks) {
            return
        }

        val payloadSize = length - HEADER_SIZE

        if (payloadSize <= 0) {
            return
        }

        val payload = ByteArray(payloadSize)

        System.arraycopy(
            data,
            HEADER_SIZE,
            payload,
            0,
            payloadSize
        )

        synchronized(frameLock) {

            if (
                lastFrameId >= 0 &&
                frameId < lastFrameId
            ) {
                return
            }

            var frame = frames[frameId]

            if (frame == null) {
                frame = FrameBuffer(
                    frameId = frameId,
                    packetCount = totalChunks,
                    packets = arrayOfNulls(totalChunks)
                )

                frames[frameId] = frame

                Log.d(
                    TAG,
                    "NETRA_UDP: NETR packet received frame=$frameId chunks=$totalChunks"
                )
            }

            if (frame.packetCount != totalChunks) {
                frames.remove(frameId)
                return
            }

            if (frame.packets[chunkId] == null) {
                frame.packets[chunkId] = payload
                frame.received++
            }

            if (frame.received == frame.packetCount) {

                val jpegData = rebuildFrame(frame)

                frames.remove(frameId)

                lastFrameId = maxOf(
                    lastFrameId,
                    frameId
                )

                if (jpegData != null) {
                    deliverFrame(frameId, jpegData)
                }
            }
        }
    }

    private fun rebuildFrame(frame: FrameBuffer): ByteArray? {

        var totalSize = 0

        for (packet in frame.packets) {
            if (packet == null) {
                return null
            }

            totalSize += packet.size

            if (totalSize > MAX_FRAME_SIZE) {
                Log.w(
                    TAG,
                    "Frame ${frame.frameId} exceeds maximum size"
                )
                return null
            }
        }

        val output = ByteArrayOutputStream(totalSize)

        for (packet in frame.packets) {
            output.write(packet)
        }

        return output.toByteArray()
    }

    private fun deliverFrame(
        frameId: Int,
        jpegData: ByteArray
    ) {
        try {
            val bitmap = BitmapFactory.decodeByteArray(
                jpegData,
                0,
                jpegData.size
            )

            if (bitmap == null) {
                Log.w(
                    TAG,
                    "Failed to decode JPEG for frame $frameId"
                )
                return
            }

            frameCounter++
            fpsFrameCount++

            val now = SystemClock.elapsedRealtime()
            val elapsed = now - fpsStartTime

            val fps =
                if (elapsed > 0) {
                    fpsFrameCount * 1000.0 / elapsed
                } else {
                    0.0
                }

            Log.i(TAG, "NETRA_UDP: frame reconstructed frameId=$frameId")

            onStats(
                frameCounter,
                fps
            )

            onVideoFrame(bitmap)

        } catch (e: Exception) {
            Log.e(
                TAG,
                "Frame $frameId processing failed",
                e
            )
        }
    }

    private fun cleanupOldFrames() {
        val now = SystemClock.elapsedRealtime()

        synchronized(frameLock) {
            val iterator = frames.entries.iterator()

            while (iterator.hasNext()) {
                val entry = iterator.next()

                if (
                    now - entry.value.createdAt >
                    FRAME_TIMEOUT_MS
                ) {
                    Log.d(
                        TAG,
                        "Dropping incomplete frame ${entry.key}"
                    )

                    iterator.remove()
                }
            }
        }
    }

    private fun readInt(
        data: ByteArray,
        offset: Int
    ): Int {
        return (
            ((data[offset].toInt() and 0xFF) shl 24) or
            ((data[offset + 1].toInt() and 0xFF) shl 16) or
            ((data[offset + 2].toInt() and 0xFF) shl 8) or
            (data[offset + 3].toInt() and 0xFF)
        )
    }

    private fun readUnsignedShort(
        data: ByteArray,
        offset: Int
    ): Int {
        return (
            ((data[offset].toInt() and 0xFF) shl 8) or
            (data[offset + 1].toInt() and 0xFF)
        )
    }

    fun stop() {
        if (!running.getAndSet(false)) {
            return
        }

        onStatus(
            "Stopping Pi UDP receiver..."
        )

        try {
            socket?.close()
        } catch (e: Exception) {
            Log.e(TAG, "Error closing UDP socket", e)
        }

        socket = null

        try {
            receiveThread?.interrupt()
        } catch (_: Exception) {
        }

        receiveThread = null

        synchronized(frameLock) {
            frames.clear()
        }

        onStatus(
            "Pi UDP receiver stopped"
        )

        Log.i(TAG, "NETRA_UDP receiver stopped")
    }
}

